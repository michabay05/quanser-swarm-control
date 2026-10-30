"""Interactive plot for visualizing RBF density functions.

You can add points by clicking on the plot, and remove points by
right-clicking on them. You can undo changes by pressing
ctrl+z. shift+R clears the plot. This cannot be undone.

DensityPlot.threed controls whether or not to plot a 3D surface.
"""

from numpy.typing import NDArray
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from matplotlib.backend_bases import MouseButton
# matplotlib.use('TkAgg')


from .coverage import CoverageAlg, mesh2coords, rbf


def mesh2d(xmin, xmax, ymin, ymax, nx, ny=None):
    if ny is None:
        ny = nx
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y

class DensityPlot:
    threed = True

    def __init__(self, domain,
        cov_add_point, cov_rm_point, compute_density, get_user_pts,
        key_callback, close_callback,
        epsilon=0.1, mult=1):
        """
        Parameters:
            domain (Polygon): shapely polygon
            agent_pts (NDArray): np array of size (n_agents, 2)
            cov_add_point ((index, point) -> None): callback
            cov_rm_point ((index) -> None): callback
            compute_density ((NDArray, NDArray) -> NDArray): callback
            key_callback ((event) -> None): callback
            close_callback (() -> None): callback
        """

        minx, miny, maxx, maxy = domain.bounds

        assert cov_add_point is not None
        self.cov_add_point = cov_add_point
        assert cov_rm_point is not None
        self.cov_rm_point = cov_rm_point
        assert compute_density is not None
        self.compute_density = compute_density
        assert get_user_pts is not None
        self.get_user_pts = get_user_pts

        assert key_callback is not None
        self.key_callback = key_callback
        assert close_callback is not None
        self.close_callback = close_callback

        self.xrange = (minx, maxx)
        self.yrange = (miny, maxy)
        self.resolution = 0.2
        self.artists = {}
        self.handlers = {}
        self.setup()
        self.attach_handlers()
        self.history = []

    def setup(self):
        size = (10, 5) if self.threed else (5.5, 5)
        self.fig = plt.figure(figsize=size)
        ax = self.fig.add_subplot(1, 2 if self.threed else 1, 1)
        ax3d = self.fig.add_subplot(1, 2, 2,
            projection='3d') if self.threed else None

        # axw = ax.twinx()
        assert isinstance(ax, plt.Axes)
        assert isinstance(ax3d, plt3d.axes3d.Axes3D)
        self.axs: tuple[plt.Axes, plt3d.axes3d.Axes3D] = (ax, ax3d)
        self.artists = {}
        plt.ion()
        self.fig.show()
        self.fig.stale_callback = None
        return self.fig, self.axs

    def set_mesh(self, resolution=None):
        resolution = resolution or self.resolution
        xmin, xmax, ymin, ymax = *self.xrange, *self.yrange
        xres = (xmax - xmin) / resolution
        yres = (ymax - ymin) / resolution
        xs = np.linspace(xmin, xmax, np.round(xres).astype(int))
        ys = np.linspace(ymin, ymax, np.round(yres).astype(int))

        self.X, self.Y = np.meshgrid(xs, ys)
        self.Z = np.zeros_like(self.X)
        return self.X, self.Y

    def recalculate_density(self):
        self.Z = self.compute_density(self.X, self.Y)

    def firstplot(self, positions, vpolys, centroids, points=None, from_scratch=True):
        if from_scratch:
            # NOTE: Should only happen when the user add/rms points
            self.set_mesh()
            self.recalculate_density()

        ar = self.artists
        ar['2dimg'] = self.axs[0].imshow(
            self.Z, extent=(*self.xrange, *self.yrange), origin='lower', zorder=0)
        x, y = np.asarray(points).T if points else (np.empty((0)), np.empty((0)))
        ar['2dpts'] = self.axs[0].scatter(x, y, s=50, c='r', alpha=0.3, marker='.', zorder=2)
        ar['2dpts'].set_picker(True)

        if from_scratch:
            # NOTE: Should only happen when the user add/rms points
            self.new_contour()
            self.new_surface()

        ag_x, ag_y = np.asarray(positions).T
        ar['agent_pts'] = self.axs[0].scatter(
            ag_x, ag_y, s=50, c='r', alpha=1, marker='x', zorder=4, animated=True)

        for i, poly in enumerate(vpolys):
            x, y = poly.exterior.xy
            ar[f'vp_{i}'] = self.axs[0].fill(x, y, alpha=0.4, edgecolor="black", animated=True)

        ct_x, ct_y = np.asarray(centroids).T if centroids else (np.empty((0)), np.empty((0)))
        ar['cntds'] = self.axs[0].scatter(
            ct_x, ct_y, s=50, c='b', alpha=1, marker='o', zorder=3, animated=True)

        self.cache_background()

    def new_contour(self):
        if '2dcont' in self.artists:
            self.artists['2dcont'].remove()
        self.artists['2dcont'] = self.axs[0].contour(self.X, self.Y, self.Z, zorder=1)

    def add_point(self, point, index=-1, add_to_history=True):
        user_pts = self.get_user_pts()

        if index < 0:
            index = len(user_pts) - index + 1
        if add_to_history:
            self.history.append(('add_point', index, len(user_pts)))

        self.cov_add_point(index, point)
        self.recalculate_density()
        self.update()

    def remove_point(self, index, add_to_history=True, update=True):
        user_pts = self.get_user_pts()
        point = self.cov_rm_point(index)
        if add_to_history:
            self.history.append(('del_point', point, index))
        if update:
            self.recalculate_density()
            self.update()

    def undo(self):
        user_pts = self.get_user_pts()
        if not self.history:
            print('nothing to undo')
            return
        action, point, index = self.history.pop()
        if action == 'add_point':
            self.remove_point(index, add_to_history=False)
        elif action == 'del_point':
            self.add_point(point, index, add_to_history=False)
            self.update()
        print('undo', action, point, index)

    def update(self):
        user_pts = self.get_user_pts()
        self.artists['2dimg'].set_data(self.Z)
        self.artists['2dimg'].autoscale()  # update the image's cached norm.vmin/vmax
        self.artists['2dpts'].set_offsets(np.asarray(user_pts).reshape((-1, 2)))
        # self.axs[0].draw_artist(self.artists['2dimg'])
        self.new_contour()
        self.new_surface()
        self.cache_background()

    def relim(self, xlim=None, ylim=None):
        user_pts = self.get_user_pts()
        self.xrange = xlim or self.xrange
        self.yrange = ylim or self.yrange
        self.set_mesh()
        self.recalculate_density()
        self.artists['2dimg'].set_extent((*self.xrange, *self.yrange))
        self.update()
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

    def redraw_full(self, positions, vpolys, centroids):
        for artist in self.artists.values():
            if isinstance(artist, list):
                for el in artist:
                    el.remove()
            else:
                artist.remove()

        self.artists = {}
        self.axs[0].clear()
        self.firstplot(positions, vpolys, centroids, from_scratch=True)

    def update_dynamic_artists(self, positions, vpolys, centroids):
        ag_x, ag_y = positions[:, 0], positions[:, 1]
        self.artists['agent_pts'].set_offsets(np.column_stack((ag_x, ag_y)))

        ct_x, ct_y = np.asarray(centroids).T
        self.artists['cntds'].set_offsets(np.column_stack((ct_x, ct_y)))

        for i, poly in enumerate(vpolys):
            x, y = poly.exterior.xy
            self.artists[f'vp_{i}'][0].set_xy(np.column_stack((x, y)))

    def cache_background(self):
        self.fig.canvas.draw()
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
    
    def blit(self, vpolys):
        c = self.fig.canvas
        c.restore_region(self._bg)
        for a in (self.artists['agent_pts'], self.artists['cntds'],
                  *(self.artists[f'vp_{i}'][0] for i in range(len(vpolys)))):
            self.axs[0].draw_artist(a)
        c.blit(self.fig.bbox)
        c.flush_events()
    
    def redraw(self, positions, vpolys, centroids):
        self.update_dynamic_artists(positions, vpolys, centroids)
        self.blit(vpolys)

    def plt_update_show(self):
        if self.fig:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def attach_handlers(self):
        self.handlers['on_click'] = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.handlers['on_key'] = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.handlers['on_pick'] = self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.handlers["on_close"] = self.fig.canvas.mpl_connect('close_event', self.on_close)

    def detach_handlers(self):
        for handler in self.handlers.values():
            self.fig.canvas.mpl_disconnect(handler)
        self.handlers = {}

    def on_pick(self, event):
        if event.mouseevent.button == MouseButton.RIGHT and event.artist is self.artists['2dpts']:
            print('delete point', event)
            self.remove_point(self.get_user_pts(), event.ind[0])

    def on_click(self, event):
        if event.button == MouseButton.LEFT and event.inaxes is self.axs[0]:
            print('add point', event)
            self.add_point((event.xdata, event.ydata))

    def on_close(self, event):
        self.close_callback()
        print("Received close event")
        print(event)

    def on_key(self, event):
        self.key_callback(event)

    def new_surface(self):
        if not self.threed:
            return
        if '3dsurf' in self.artists:
            self.artists['3dsurf'].remove()
        self.artists['3dsurf'] = self.axs[1].plot_surface(
            self.X, self.Y, self.Z, cmap=cm.coolwarm, antialiased=True)

