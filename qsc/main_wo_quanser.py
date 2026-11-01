import time

from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np
import shapely

from .coverage import CoverageAlg
from .new_rbf import DensityPlot

class SimpleLocationMng:
    def fetch_pose(self) -> NDArray[np.float64]:
        raise NotImplementedError("fetch_pose")

    def send_waypoint(self,
        positions: NDArray[np.float64], targets: NDArray[np.float64],
        dist_eps=1e-3, vmax=3, frame_dt=1/60):
        d_pos = targets - positions
        dist = np.linalg.norm(d_pos, axis=1)
        if not np.any(dist > dist_eps):
            return

        target_heading = np.arctan2(d_pos[:, 1], d_pos[:, 0])
        out_v = dist / frame_dt
        new_v = np.clip(out_v, -np.abs(vmax), np.abs(vmax))
        new_v = np.where(dist > dist_eps, new_v, 0.0)

        # V = [[v0]           [[cos(heading), sin(heading)],
        #      [v1]        *   [cos(heading), sin(heading)],
        #      [v2], ...]      [cos(heading), sin(heading)], ...]
        V = new_v[:, None] * np.column_stack((np.cos(target_heading), np.sin(target_heading)))
        # TODO(mabay): remove this
        # This method should only set the relevant linear and angular velocity values
        positions += V * frame_dt


class Manager:
    def __init__(self,
        positions: NDArray[np.float64], location_mng: SimpleLocationMng, side_len=5):
        domain = shapely.box(0, 0, side_len, side_len)
        assert positions.shape[1] == 2, "Incorrect shape: Expected (N, 2) ~ 2D positions"
        self.positions = positions
        self.user_pts = []

        self.cov = CoverageAlg(
            domain, self.get_user_pts, self.positions, epsilon=0.5, mult=10)
        self.viz = DensityPlot(
            domain, self.cov.add_point, self.cov.remove_point,
            self.get_user_pts, self.cov.compute_density,
            self.key_callback, self.close_callback)

        assert location_mng is not None
        self.loc_mng = location_mng

        self.vmax = 3.
        # TODO(mabay): Not sure if how to deal with frame time
        self.frame_dt = 1/60
        self.targets = np.zeros_like(positions)
        self.quit = False

    def get_user_pts(self):
        return self.user_pts

    def clear_user_pts(self):
        self.user_pts.clear()

    def key_callback(self, event):
        if event.key == 'ctrl+r':
            self.viz.redraw_full(self.positions, self.cov.vpolys, self.cov.centroids)
            # self.viz.plt_update_show()
            print('redraw')
        elif event.key == 'ctrl+z':
            self.viz.undo()
            print('undo')
        elif event.key == 'R':
            self.clear_user_pts()
            self.viz.history = []
            self.viz.redraw_full(self.positions, self.cov.vpolys, self.cov.centroids)
            # self.viz.plt_update_show()
            print('reset (this cannot be undone)')

    def close_callback(self):
        self.quit = True

    def update_targets(self):
        # TODO(mabay): Currently, this is a no-op; make it do something
        # self.positions = self.loc_mng.fetch_pose(self.positions)
        ax, ay = self.positions[:, 0], self.positions[:, 1]
        targets = np.empty_like(self.positions)

        # TODO(mabay): Think about using 'ordered=True' during voronoi computation to remove the need
        # for this checking loop
        for i, poly in enumerate(self.cov.vpolys):
            mask = shapely.contains_xy(poly, ax, ay)
            idx = np.flatnonzero(mask)
            assert idx.size == 1
            targets[idx[0]] = np.array(self.cov.centroids[i])

        self.targets = targets

    def run(self):
        self.viz.firstplot(self.positions, self.cov.vpolys, self.cov.centroids)
        self.viz.plt_update_show()

        while not self.quit:
            start = time.perf_counter()
            self.cov.update()
            self.update_targets()
            self.loc_mng.send_waypoint(self.positions, self.targets)

            self.viz.redraw(self.positions, self.cov.vpolys, self.cov.centroids)
            # self.viz.plt_update_show()

            end = time.perf_counter()
            print(f">> Frame time: {(end - start)*1000:>7.2f} ms", end="\r")


if __name__ == "__main__":
    n_agents = 5
    side_len = 10

    rng = np.random.default_rng(seed=21)
    margin = 0.1 * side_len
    positions = rng.uniform(low=margin, high=side_len - 2*margin, size=(n_agents, 2))

    mng = Manager(positions, SimpleLocationMng(), side_len=side_len)
    try:
        mng.run()
    except KeyboardInterrupt:
        print("Detected <C-c>; quitting now...")

    """
    finally:
        # NOTE(mabay): not sure if I need this; revisit the need for this maybe running this script on the physical qbot platform.
        # os.system("quarc_run -q -Q -t tcpip://localhost:17000 *.rt-linux_qbot_platform -d /tmp")
        print("Script has terminated.")
    """
