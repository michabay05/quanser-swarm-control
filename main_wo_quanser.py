import time
import os

import numpy as np
from numpy.typing import NDArray
import shapely
from shapely import MultiPoint, Polygon
import matplotlib.pyplot as plt

from new_rbf import DensityPlot

class Manager(DensityPlot):
    KEY_PREFIX = "qb"

    def __init__(self, seed=21, side_len=5, n_agents=4, ip_driver="localhost"):
        rng = np.random.default_rng(seed)
        margin = 2
        agent_pts = rng.uniform(low=margin, high=side_len - margin, size=(n_agents, 2))

        domain = shapely.box(0, 0, side_len, side_len)
        super().__init__(domain, agent_pts, epsilon=5e-1, mult=1000)

        self.frame_dt = 1 / 60
        self.vmax = 3.
        self.qbp_key = lambda actor_no: f"{self.KEY_PREFIX}_{actor_no}"
        self.targets = np.empty_like(agent_pts)

        self.quit = False

    def update_move_cmd_np(self, dist_eps=1e-4):
        targets = self.targets
        d_pos = targets - self.agent_pts
        dist = np.linalg.norm(d_pos, axis=1)
        if not np.any(dist > dist_eps):
            return

        target_heading = np.arctan2(d_pos[:, 1], d_pos[:, 0])
        out_v = dist / self.frame_dt
        new_v = np.clip(out_v, -np.abs(self.vmax), np.abs(self.vmax))
        new_v = np.where(dist > dist_eps, new_v, 0.0)

        # V = [[v0]           [[cos(heading), sin(heading)],
        #      [v1]        *   [cos(heading), sin(heading)],
        #      [v2], ...]      [cos(heading), sin(heading)], ...]
        V = new_v[:, None] * np.column_stack((np.cos(target_heading), np.sin(target_heading)))
        self.agent_pts += V * self.frame_dt

    def update_targets(self):
        ax, ay = self.agent_pts[:, 0], self.agent_pts[:, 1]
        targets = np.empty_like(self.agent_pts)

        for i, poly in enumerate(self.voronoi_polys):
            mask = shapely.contains_xy(poly, ax, ay)
            idx = np.flatnonzero(mask)
            assert idx.size == 1
            targets[idx[0]] = np.array(self.centroids[i])

        self.targets = targets

    def add_point(self, point, index=-1, add_to_history=True):
        super().add_point(point, index, add_to_history)

    @property
    def should_start(self):
        return len(self.user_pts) > 0

    def on_close(self, event):
        super().on_close(event)
        self.quit = True

    def run(self):
        mng.firstplot()
        mng.plt_update_show()

        frame = 0
        while not self.quit:
            start = time.time()
            self.recompute_voronoi()
            self.update_targets()
            self.update_move_cmd_np()

            if frame % 3 == 0:
                self.redraw()
                self.plt_update_show()
                frame = 0

            # time.sleep(0.005)

            frame += 1
            print("\t\t\t\t\t", time.time() - start)
            # time.sleep(self.frame_dt)


if __name__ == "__main__":
    n_agents = 5
    side_len = 10

    mng = Manager(side_len=side_len, n_agents=n_agents)
    try:
        mng.run()
    except KeyboardInterrupt:
        print("Detected <C-c>; quitting now...")
    except Exception as e:
        print(e)
    finally:
        # NOTE(mabay): not sure if I need this; revisit the need for this maybe running this script on the physical qbot platform.
        # os.system("quarc_run -q -Q -t tcpip://localhost:17000 *.rt-linux_qbot_platform -d /tmp")
        print("Script has terminated.")

