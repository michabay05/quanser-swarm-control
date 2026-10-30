# Quanser Error code source
# https://docs.quanser.com/quarc/documentation/net/html/error_codes.html

import time
import os

import numpy as np
from numpy.typing import NDArray
import shapely
from shapely import MultiPoint, Polygon
import matplotlib.pyplot as plt

from new_rbf import DensityPlot
from qvl.multi_agent import MultiAgent
from quanser.hardware import HILError
from qvl.qbot_platform import QLabsQBotPlatform
from pal.products.qbot_platform import QBotPlatformDriver
from sc_setup import setup_boilerplate

class Manager(DensityPlot):
    KEY_PREFIX = "qb"

    def __init__(self, seed=21, side_len=5, n_agents=4, ip_driver="localhost"):
        rng = np.random.default_rng(seed)
        margin = 2
        agent_pts = rng.uniform(low=margin, high=side_len - margin, size=(n_agents, 2))

        domain = shapely.box(0, 0, side_len, side_len)
        super().__init__(domain, agent_pts, epsilon=1, mult=10)

        # Quanser-related attributes
        self.spawner: MultiAgent = setup_boilerplate(agent_pts, verbose=True)
        # Wait a bit for the setup to complete fully
        time.sleep(1)

        # NOTE(mabay): might not be necessary to validate the type
        for i, qb in enumerate(self.spawner.robotActors):
            assert isinstance(qb, QLabsQBotPlatform), f"Expected self.qbots[{i}] to a QLabQBotPlatform object, instead got {type(qb)}"

        self.qbots: list[QLabsQBotPlatform] = self.spawner.robotActors

        self._qbot_pds: dict[str, QBotPlatformDriver] = {
            f"{self.KEY_PREFIX}_{rd['actorNumber']}": QBotPlatformDriver(mode=1, ip=ip_driver, driverPort=rd["driverPort"])
            for rd in self.spawner.robotsDict.values()
        }

        self.start_time = time.time()
        time.sleep(3)
        print(">>>>>>")

        # NOTE(mabay): I believe the quanser simulator runs at 60 fps
        self.frame_dt = 1 / 60
        self.vmax = 0.3
        self.wmax = np.deg2rad(90)
        self.qbp_key = lambda actor_no: f"{self.KEY_PREFIX}_{actor_no}"
        self.targets: dict[str, NDArray[np.float64] | None] = {
            self.qbp_key(actor_no): None for actor_no in self.actor_nums
        }

        self.quit = False

    @property
    def actor_nums(self) -> list[int]:
        return [int(qb.actorNumber) for qb in self.qbots]

    def qbp_get_dr(self, actor_no: int):
        """
        Retrieves the platform driver based on the QBotPlatform's ID (aka. actor number)
        """
        assert actor_no in self.actor_nums, "Unknown actor number"
        return self._qbot_pds[self.qbp_key(actor_no)]
    
    @staticmethod
    def qbp_fetch_pose_2d(qbp: QLabsQBotPlatform) -> tuple[NDArray[np.float64], float]:
        _, loc, rot, _ = qbp.get_world_transform()
        return np.array(loc[:2]), rot[2]

    def qbp_fetch_all_pos_2d(self):
        return np.array([
            Manager.qbp_fetch_pose_2d(qbot)[0] for qbot in self.qbots
        ])

    @staticmethod
    def compute_time_from_pos(
        start_pos: tuple[float, float], start_rot: float,
        target_pos: tuple[float, float], qb_v: float, qb_w: float
    ) -> tuple[float, float, float, float]:
        """Computes the amount of time required to execute a sequence
        of moves that will allow a qbot to fix its heading (step 1)
        and go to its target position (step 2)

        Parameters:
            start_pos (tuple[float, float]): qbot's starting position on the xy-plane
            start_rot (float): qbot's starting rotation about the z-axis
            target_pos (tuple[float, float]): qbot's target position on the xy-plane
            qb_v (float): constant linear velocity with which time is computed
            qb_w (float): constant angular velocity with which time is computed

        Returns:
            tuple[float, float, float, float]: time to move along heading (t_v), time to fix heading (t_w),
                remaining distance to target, and remaining d_angle to target heading
        """
        pos_xy = np.array(start_pos)
        rot_z  = start_rot
        target = np.array(target_pos)
        d_pos = target - pos_xy
        dist_to_target = np.linalg.norm(d_pos)

        # Step 1: Fix heading
        target_heading = np.arctan2(d_pos[1], d_pos[0])
        d_angle = target_heading - rot_z
        out_tw = d_angle / qb_w

        # Step 2: Travel along heading
        out_tv = dist_to_target / qb_v

        return float(out_tv), out_tw, float(dist_to_target), d_angle
    
    @staticmethod
    def compute_velocity_from_pos(
        start_pos: tuple[float, float], start_rot: float,
        target_pos: tuple[float, float], dt: float
    ) -> tuple[float, float, float, float]:
        pos_xy = np.array(start_pos)
        rot_z  = start_rot
        target = np.array(target_pos)
        d_pos = target - pos_xy

        # Step 1: Fix heading
        target_heading = np.arctan2(d_pos[1], d_pos[0])
        d_angle = target_heading - rot_z
        if np.abs(d_angle) > np.pi:
            d_angle += (-2 * np.pi) if d_angle > 0 else (2 * np.pi)

        out_w = d_angle / dt

        # Step 2: Travel along heading
        dist_to_target = float(np.linalg.norm(d_pos))
        out_v = dist_to_target / dt

        return out_v, out_w, dist_to_target, d_angle

    def update_move_cmd(self, qbp: QLabsQBotPlatform):
        actor_no: int = qbp.actorNumber # type: ignore
        key = self.qbp_key(actor_no)
        driver = self.qbp_get_dr(actor_no)        
        target = self.targets[key]
        if target is None:
            self.qbot_stop(actor_no)
            return

        start_pos, start_rot = Manager.qbp_fetch_pose_2d(qbp)
        calc_v, calc_w, dist_to_target, d_angle = Manager.compute_velocity_from_pos(
            start_pos.tolist(), start_rot, target.tolist(), self.frame_dt)

        vmax_abs, wmax_abs = np.abs(self.vmax), np.abs(self.wmax)
        new_v = np.clip(a=calc_v, a_min=-vmax_abs, a_max=vmax_abs)
        new_w = np.clip(a=calc_w, a_min=-wmax_abs, a_max=wmax_abs)
        
        dist_eps = 1e-1
        angle_eps = np.deg2rad(30)

        if np.abs(d_angle) <= angle_eps:
            new_w = 0.0
        else:
            new_v = 0.0

        print(f"start_rot_z = {start_rot:.3f}; rem = {d_angle:.3f}")
        print(f"v = {new_v:.3f}; w = {new_w:.3f}")
        print(f"dist_to_target = {dist_to_target:.3f}")
        print("--------------------")

        # Send linear and angular velocity command
        driver.read_write_std(
            timestamp=self._elapsed_time(), arm=1, commands=np.array([new_v, new_w]))
        
        if dist_to_target <= dist_eps:
            self.targets[key] = None

    def update_move_cmd_old(self, qbp: QLabsQBotPlatform, vmax: float, wmax: float):
        """Using `self.frame_dt`, this method computes the linear and angular velocity values as
        well as the required movement duration. Then, it sends a command to the relevant qbots

        Parameters:
            vmax (float): specified in m/s
            wmax (float): specified in rad/s
        """
        key = self.qbp_key(qbp.actorNumber)
        driver = self.qbp_get_dr(qbp.actorNumber)        
        target = self.targets[key]
        if target is None:
            return

        dt = self.frame_dt
        start_pos, start_rot = Manager.qbp_fetch_pose_2d(qbp)
        t_v, t_w, dist_to_target, d_angle = Manager.compute_time_from_pos(
            start_pos, start_rot, target, vmax, wmax)
        
        print(f"Moving at (v_max, w_max): t_v={t_v:.3f}, t_w={t_w:.3f}")

        # Terms:
        #   - ttfh: time to fix heading (fix heading or omega)
        #   - ttrd: time to reach destination (fix location or v)

        # First, finish fixing heading
        new_v, new_w = 0.0, 0.0
        t_eps = 0.0
        if t_w >= self.frame_dt:
            # Since ttfh is greater than frame time, for this frame rotate at w_max
            new_w = wmax

            # self.qbot_stop(qbp.actorNumber)
        elif t_w < t_eps:
            # Given that ttfh is very little (or just 0.0), the remaining rotation is negligble.
            # Move on to moving qbot to the desired location
            new_w = 0.0

            if t_v >= self.frame_dt:
                # Since ttrd is greater than frame time, for this frame rotate at w_max
                new_v = vmax
            elif t_v < t_eps:
                # Given that ttrd is very little, the remaining distance can be considered 
                # negligble. Therefore, it must mean that the qbot must have arrived at its target.
                self.targets[key] = None
                self.qbot_stop(qbp.actorNumber)
                return
            else:
                # Since ttrd is in [t_eps, self.frame_dt], the qbot has to rotate with omega < w_max
                new_v = dist_to_target / self.frame_dt
        else:
            # Since ttfh is in [t_eps, self.frame_dt], the qbot has to rotate with omega < w_max
            new_w = d_angle / self.frame_dt

            # self.qbot_stop(qbp.actorNumber)

        # Send linear and angular velocity command
        driver.read_write_std(
            timestamp=self._elapsed_time(), arm=1, commands=[new_v, new_w])

    def update_targets(self):
        self.recompute_voronoi()

        self.targets = {}
        for i, poly in enumerate(self.voronoi_polys):
            curr_qbp = None

            # Find qbot contained within this voronoi polygon
            for qbot in self.qbots:
                pos, _ = Manager.qbp_fetch_pose_2d(qbot)
                if shapely.contains_xy(poly, *pos):
                    curr_qbp = qbot
                    break
            
            assert curr_qbp is not None
            self.targets[self.qbp_key(curr_qbp.actorNumber)] = np.array(self.centroids[i])

    def add_point(self, point, index=-1, add_to_history=True):
        super().add_point(point, index, add_to_history)
        self.update_targets()

    def on_close(self, event):
        super().on_close(event)
        self.quit = True

    def _elapsed_time(self):
        return time.time() - self.start_time
    
    def terminate(self):
        time.sleep(0.1)
        for qbp_dr in self._qbot_pds.values():
            qbp_dr.terminate()

        self.spawner.qlabs.close()
        time.sleep(0.5)

    def qbot_stop(self, actor_no: int) -> None:
        qb_pd = self.qbp_get_dr(actor_no)
        delay = 0.1

        # NOTE(mabay): This is hacky but these sequence of commands seem necessary
        # to fully stop the qbot from moving in sim.
        qb_pd.read_write_std(timestamp=self._elapsed_time(),
            arm=0, commands=np.zeros(2))
        time.sleep(delay)
        qb_pd.read_write_std(timestamp=self._elapsed_time(),
            arm=1, commands=np.zeros(2))
        time.sleep(delay)
        qb_pd.read_write_std(timestamp=self._elapsed_time(),
            arm=0, commands=np.zeros(2))

    def run(self):
        mng.firstplot()
        mng.plt_update_show()

        start: float = 0.0
        while not self.quit:
            start = self._elapsed_time()

            # Update all the necessary positions so that the latest position
            # data is reflected on the matplotlib window
            self.agent_pts = self.qbp_fetch_all_pos_2d()
            self.recompute_voronoi()

            # NOTE: The target for each qbot is added when a user clicks on a point.
            for qbot in self.qbots:
                self.update_move_cmd(qbot)

            print("\n")

            self.redraw()
            self.plt_update_show()
            print("\t\t\t\t\t", self._elapsed_time() - start)
            # time.sleep(self.frame_dt)


if __name__ == "__main__":
    n_agents = 4
    side_len = 10

    mng = Manager(side_len=side_len, n_agents=n_agents)
    try:
        mng.run()
    except HILError as h:
        print(h.get_error_message())
    except KeyboardInterrupt:
        print("Detected <C-c>; quitting now...")
    except Exception as e:
        print(e)
    finally:
        mng.terminate()
        # NOTE(mabay): not sure if I need this; revisit the need for this maybe running this script on the physical qbot platform.
        # os.system("quarc_run -q -Q -t tcpip://localhost:17000 *.rt-linux_qbot_platform -d /tmp")
        print("Script has terminated.")


    
def _wait_until_target_angle(qbot, target_heading, angle_eps, ask_dt):
    while True:
        _, _, rot, _ = qbot.get_world_transform()
        if np.abs(target_heading - rot[2]) <= angle_eps:
            print("Done rotating.")
            break
            
        time.sleep(ask_dt)
        
def _wait_until_target_pos(qbot, target_pos, dist_eps, ask_dt):
    while True:
        _, loc, _, _ = qbot.get_world_transform()
        xy = loc[:2]
        if np.abs(np.linalg.norm(target_pos) - np.linalg.norm(xy)) <= dist_eps:
            print("Done moving.")
            break

        time.sleep(ask_dt)

def qbot_stop_old(qb_pd, timestamp, actor_no: int) -> None:
    qb_pd.read_write_std(timestamp, arm=1, hold=1, commands=np.zeros(2))