import time

import numpy as np
from numpy.typing import NDArray

from qvl.multi_agent import MultiAgent
from quanser.hardware import HILError
from qvl.qbot_platform import QLabsQBotPlatform
from pal.products.qbot_platform import QBotPlatformDriver
from sc_setup import setup_boilerplate

class QManager:
    KEY_PREFIX = "qb"

    def __init__(self,
        agent_pts: NDArray[np.float64], ip_driver="localhost",
        vmax=0.3, wmax=np.deg2rad(30)
    ):
        self.spawner: MultiAgent = setup_boilerplate(agent_pts, verbose=True)
        # Wait a bit for the setup to complete fully
        time.sleep(1)
        print("Spawning complete.")

        # NOTE(mabay): might not be necessary to validate the type
        for i, qb in enumerate(self.spawner.robotActors):
            assert isinstance(qb, QLabsQBotPlatform), f"Expected self.qbots[{i}] to a QLabQBotPlatform object, instead got {type(qb)}"

        self.qbots: list[QLabsQBotPlatform] = self.spawner.robotActors
        self.qbp_key = lambda actor_no: f"{self.KEY_PREFIX}_{actor_no}"
        self._qbot_pds: dict[str, QBotPlatformDriver] = {
            f"{self.qbp_key(rd['actorNumber'])}": QBotPlatformDriver(
                mode=1, ip=ip_driver, driverPort=rd["driverPort"])
            for rd in self.spawner.robotsDict.values()
        }
        self.vmax = vmax
        self.wmax = wmax
        self.start_time = time.time()
        time.sleep(3)

        print(f"qbot platform: vmax = {self.vmax}; wmax = {self.wmax}")
        print("Initialization complete.")

        self.frame_dt = 1 / 60

    @property
    def actor_nums(self) -> list[int]:
        return [int(qb.actorNumber) for qb in self.qbots]

    def qbp_get_dr(self, actor_no: int):
        """
        Retrieves the platform driver based on the QBotPlatform's ID (aka. actor number)
        """
        assert actor_no in self.actor_nums, "Unknown actor number"
        return self._qbot_pds[self.qbp_key(actor_no)]
        
    def _elapsed_time(self):
        return time.time() - self.start_time

    @staticmethod
    def qbp_fetch_pose_2d(qbp: QLabsQBotPlatform) -> NDArray[np.float64]:
        _, loc, rot, _ = qbp.get_world_transform()
        return np.array([loc[0], loc[1], rot[2]]) # [x, y, rot_z]

    def qbp_fetch_all_pos_2d(self):
        return np.array([
            self.qbp_fetch_pose_2d(qbot)[0] for qbot in self.qbots
        ])
    
    def qbp_fetch_all_pose_2d(self):
        return np.array([
            self.qbp_fetch_pose_2d(qbot) for qbot in self.qbots
        ])
 
    @staticmethod
    def compute_velocity_from_pos(
        start_pos: tuple[float, float], start_rot: float,
        target_pos: tuple[float, float], dt: float
    ) -> tuple[float, float, float, float]:
        """Computes the linear and angular velocity required to execute a sequence of
        moves that will allow a qbot to fix its heading (step 1) and go to its target
        position (step 2)

        Parameters:
            start_pos (tuple[float, float]): qbot's starting position on the xy-plane
            start_rot (float): qbot's starting rotation about the z-axis
            target_pos (tuple[float, float]): qbot's target position on the xy-plane
            dt (float): time used to compute the required linear and angular velocity

        Returns:
            tuple[float, float, float, float]: linear vel (v), angular vel (omega),
                remaining distance to target, and remaining d_angle to target heading
        """
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

    def update_move_cmd_np(self,
        positions: NDArray[np.float64], rotations: NDArray[np.float64],
        targets: NDArray[np.float64], dist_eps=0.1, angle_eps=np.deg2rad(30)
    ):
        """
        Params:
            positions (np.array): (n, 2)
            rotations (np.array): (n, 1)
            targets (np.array): (n, 2)
        """
        dt = self.frame_dt
        
        d_pos = targets - positions
        dist = np.linalg.norm(d_pos, axis=1)
        if not np.any(dist > dist_eps):
            return

        target_heading = np.arctan2(d_pos[:, 1], d_pos[:, 0])
        d_angle = target_heading - rotations
        d_angle = np.where(np.abs(d_angle) > np.pi,
                    d_angle - np.sign(d_angle) * 2 * np.pi,
                    d_angle)
        # # NOTE: The above `np.where` expression is derived from:
        # if np.abs(d_angle) > np.pi:
        #     d_angle += (-2 * np.pi) if d_angle > 0 else (2 * np.pi)

        out_w = d_angle / dt
        out_v = dist / dt

        new_w = np.clip(out_w, -np.abs(self.wmax), np.abs(self.wmax))
        new_w = np.where(np.abs(d_angle) > angle_eps, new_w, 0.0)

        new_v = np.clip(out_v, -np.abs(self.vmax), np.abs(self.vmax))
        new_v = np.where(np.isclose(new_w, 0.0, atol=0.05), new_v, 0.0)
        new_v = np.where(dist > dist_eps, new_v, 0.0)

        for i, qbot in enumerate(self.qbots):
            driver = self.qbp_get_dr(qbot.actorNumber)
            driver.read_write_std(
                timestamp=self._elapsed_time(),
                arm=1, commands=np.array([new_v[i], new_w[i]]))
            

    def terminate(self):
        time.sleep(0.1)
        for qbp_dr in self._qbot_pds.values():
            qbp_dr.terminate()

        self.spawner.qlabs.close()
        time.sleep(0.5)