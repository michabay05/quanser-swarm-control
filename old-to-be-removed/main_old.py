import time

import shapely
import numpy as np
import matplotlib.pyplot as plt

from new_rbf import DensityPlot
from sc_setup import setup_boilerplate


class MiddleMan:
    def __init__(self, side_len=5, seed=3, n_agents=4, ip_driver="localhost"):
        rng = np.random.default_rng(seed)
        self.agent_pts = rng.uniform(low=0, high=side_len, size=(n_agents, 2))

        # Density plot-related attributes
        self.domain = shapely.box(0, 0, side_len, side_len)
        self.dp = DensityPlot(
            self.domain, self.agent_pts, self.set_user_pt_callback)

        # Quanser-related attributes
        self.qbot = setup_boilerplate(locationQBotP = [0, 0, 0.1], verbose=True)
        self.qbot_pd = QBotPlatformDriver(mode=1, ip=ip_driver)
        self.start_time = time.time()
        # Time between consecutive polls to the quanser sim
        self.ask_dt = 0.05

        self._ui_init()

    def terminate(self):
        self.qbot_pd.terminate()

    def _ui_init(self):
        self.dp.firstplot()
        self.dp.update()
        self.dp.recalculate_density()
        self.dp.plt_update_show()
        plt.draw()
        plt.pause(0.1)
        plt.show(block=True)

    def _elapsed_time(self):
        return time.time() - self.start_time
    
    def set_user_pt_callback(self, user_pts):
        pass
    
    def stop_qbot(self, comm_dt: float = 0.1) -> None:
        # NOTE(mabay): This is hacky but these sequence of commands seem necessary
        # to fully stop the qbot from moving in sim.
        self.qbot.read_write_std(timestamp=self._elapsed_time(), arm=0, commands=[0, 0])
        time.sleep(comm_dt)
        self.qbot.read_write_std(timestamp=self._elapsed_time(), arm=1, commands=[0, 0])
        time.sleep(comm_dt)
        self.qbot.read_write_std(timestamp=self._elapsed_time(), arm=0, commands=[0, 0])

    def compute_time_from_pos(
        self, target_pos: tuple[float, float], qb_v: float, qb_w: float
    ) -> tuple[float, float]:
        """
        Computes the amount of time required to execute a sequence
        of moves that will allow a qbot to fix its heading (step 1)
        and go to its target position (step 2)

        Returns:
            tuple[float, float]: time to move along heading (t_v) AND time to fix heading (t_v)
        """
        _, pos, rot, _ = self.qbot.get_world_transform()
        pos_xy = np.array(pos[:2])
        rot_z  = rot[2]
        target = np.array(target_pos)
        d_pos = target - pos_xy

        # Step 1: Fix heading
        target_heading = np.arctan2(d_pos[1], d_pos[0])
        print(target_heading)
        d_angle = target_heading - rot_z
        out_tw = d_angle / qb_w

        # Step 2: Travel along heading
        out_tv = np.linalg.norm(d_pos) / qb_v

        return (out_tv, out_tw)
    
    def _wait_until_taret_angle(self, target_heading, angle_eps):
        while True:
            _, _, rot, _ = self.qbot.get_world_transform()
            if np.abs(target_heading - rot[2]) <= angle_eps:
                print("Done rotating.")
                break
                
            time.sleep(self.ask_dt)
            
    def _wait_until_target_pos(self, target_pos, dist_eps):
        while True:
            _, loc, _, _ = self.qbot.get_world_transform()
            xy = loc[:2]
            if np.abs(np.linalg.norm(target_pos) - np.linalg.norm(xy)) <= dist_eps:
                print("Done moving.")
                break

            time.sleep(self.ask_dt)


if __name__ == "__main__":
    try:
        mm = MiddleMan()
    except KeyboardInterrupt:
        print("User interrupted.")
    finally:
        mm.terminate()