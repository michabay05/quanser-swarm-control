import time
import os

import numpy as np
from numpy.typing import NDArray
import shapely
from shapely import MultiPoint, Polygon
import matplotlib.pyplot as plt

from new_rbf import DensityPlot
from quanser.hardware import HILError
from qmain import QManager

class Manager2(DensityPlot):
    KEY_PREFIX = "qb"

    def __init__(self, seed=21, side_len=5, n_agents=4, ip_driver="localhost"):
        rng = np.random.default_rng(seed)
        margin = 2
        agent_pts = rng.uniform(low=margin, high=side_len - margin, size=(n_agents, 2))

        domain = shapely.box(0, 0, side_len, side_len)
        super().__init__(domain, agent_pts, epsilon=5e-1, mult=1000)
        
        self.quanser = QManager(
            agent_pts=self.agent_pts, ip_driver=ip_driver,
            vmax=0.3, wmax=np.deg2rad(90)
        )

        self.targets = np.empty_like(agent_pts)
        self.quit = False

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
        if getattr(self, "quanser"):
            self.quanser.terminate()
        print("Quanser terminated in matplotlib close callback")

    def run(self):
        mng.firstplot()
        mng.plt_update_show()

        frame = 0
        while not self.quit:
            start = time.time()
            if self.should_start:
                xs, ys, rots = self.quanser.qbp_fetch_all_pose_2d().T
                self.agent_pts = np.array([xs, ys]).T
                self.recompute_voronoi()
                self.update_targets()
                
                self.quanser.update_move_cmd_np(positions=self.agent_pts,
                    rotations=rots, targets=self.targets, dist_eps=0.5)

            if frame % 4 == 0:
                self.redraw()
                self.plt_update_show()
                frame = 0

            frame += 1
            print("\t\t\t\t\t", time.time() - start)


if __name__ == "__main__":
    side_len = 10

    mng = Manager2(side_len=side_len)
    try:
        mng.run()
    except HILError as h:
        print(h.get_error_message())
    except KeyboardInterrupt:
        print("Detected <C-c>; quitting now...")
    finally:
        # mng.quanser.terminate()
        # NOTE(mabay): not sure if I need this; revisit the need for this maybe running this script on the physical qbot platform.
        # os.system("quarc_run -q -Q -t tcpip://localhost:17000 *.rt-linux_qbot_platform -d /tmp")
        print("Script has terminated.")

