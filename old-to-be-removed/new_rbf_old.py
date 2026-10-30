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
from matplotlib.backend_bases import MouseButton
matplotlib.use('TkAgg')


from qsc.coverage import CoverageAlg, gaussian_kernel, rbf


def mesh2d(xmin, xmax, ymin, ymax, nx, ny=None):
    if ny is None:
        ny = nx
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y

def mesh2coords(*args):
    return np.vstack([x.ravel() for x in args]).T


class DensityPlot:
    kernel = staticmethod(gaussian_kernel)
    threed = True

    def __init__(self, domain, agent_pts, epsilon=0.1, mult=1):
        """
        Parameters:
            domain (Polygon): shapely polygon
            agent_pts (NDArray): np array of size (n_agents, 2)
        """

        self.cov = CoverageAlg(domain, agent_pts, epsilon, mult)

        minx, miny, maxx, maxy = self.cov.domain.bounds

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
        self.axs = [ax, ax3d]
        self.artists = {}
        self.fig.show()
        plt.ion()
        return self.fig, self.axs

    def set_mesh(self, resolution=None):
        resolution = resolution or self.resolution
        xmin, xmax, ymin, ymax = *self.xrange, *self.yrange
        xres = (xmax - xmin) / resolution
        yres = (ymax - ymin) / resolution
        xs = np.linspace(xmin, xmax, np.round(xres).astype(int))
        ys = np.linspace(ymin, ymax, np.round(yres).astype(int))

        self.X, self.Y = np.meshgrid(xs, ys)
        return self.X, self.Y

    def recalculate_density(self):
        self.Z = self.cov.compute_density(self.X, self.Y)

    def firstplot(self, points=None, from_scratch=True):
        if points is not None:
            self.user_pts = points

        if from_scratch:
            # NOTE: Should only happen when the user add/rms points
            self.set_mesh()
            self.recalculate_density()

        ar = self.artists
        ar['2dimg'] = self.axs[0].imshow(self.Z, extent=(*self.xrange, *self.yrange), origin='lower', zorder=0)
        x, y = np.asarray(self.user_pts).T if self.user_pts else (np.empty((0)), np.empty((0)))
        ar['2dpts'] = self.axs[0].scatter(x, y, s=50, c='r', alpha=0.3, marker='.', zorder=2)
        ar['2dpts'].set_picker(True)

        if from_scratch:
            # NOTE: Should only happen when the user add/rms points
            self.new_contour()
            self.new_surface()

        ag_x, ag_y = np.asarray(self.cov.agents).T
        ar['agent_pts'] = self.axs[0].scatter(ag_x, ag_y, s=50, c='r', alpha=1, marker='x', zorder=4)

        for i, poly in enumerate(self.cov.vpolys):
            x, y = poly.exterior.xy
            ar[f'vp_{i}'] = self.axs[0].fill(x, y, alpha=0.4, edgecolor="black")

        ct_x, ct_y = np.asarray(self.cov.centroids).T if self.cov.centroids else (np.empty((0)), np.empty((0)))
        ar['cntds'] = self.axs[0].scatter(ct_x, ct_y, s=50, c='b', alpha=1, marker='o', zorder=3)

        plt.ion()

    def new_contour(self):
        if '2dcont' in self.artists:
            self.artists['2dcont'].remove()
        self.artists['2dcont'] = self.axs[0].contour(self.X, self.Y, self.Z, zorder=1)

    def add_point(self, point, index=-1, add_to_history=True):
        if index < 0:
            index = len(self.user_pts) - index + 1
        if add_to_history:
            self.history.append(('add_point', index, len(self.user_pts)))

        self.cov.add_point(index, point)
        self.recalculate_density()
        self.update()
        self.plt_update_show()

    def remove_point(self, index, add_to_history=True, update=True):
        point = self.cov.remove_point(index)
        if add_to_history:
            self.history.append(('del_point', point, index))
        if update:
            self.recalculate_density()
            self.update()
            self.plt_update_show()

    def undo(self):
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
        self.artists['2dimg'].set_data(self.Z)
        self.artists['2dimg'].autoscale()  # update the image's cached norm.vmin/vmax
        self.artists['2dpts'].set_offsets(np.asarray(self.user_pts).reshape((-1, 2)))
        # self.axs[0].draw_artist(self.artists['2dimg'])
        self.new_contour()
        self.new_surface()

    def relim(self, xlim=None, ylim=None):
        self.xrange = xlim or self.xrange
        self.yrange = ylim or self.yrange
        self.set_mesh()
        self.recalculate_density()
        self.artists['2dimg'].set_extent((*self.xrange, *self.yrange))
        self.update()
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

    def redraw_full(self):
        for artist in self.artists.values():
            if isinstance(artist, list):
                for el in artist:
                    el.remove()
            else:
                artist.remove()

        self.artists = {}
        self.axs[0].clear()
        self.firstplot(from_scratch=True)

    def update_dynamic_artists(self):
        ag_x, ag_y = self.cov.agents[:, 0], self.cov.agents[:, 1]
        self.artists['agent_pts'].set_offsets(np.column_stack((ag_x, ag_y)))

        ct_x, ct_y = np.asarray(self.cov.centroids).T
        self.artists['cntds'].set_offsets(np.column_stack((ct_x, ct_y)))

        for i, poly in enumerate(self.cov.voronoi_polys):
            x, y = poly.exterior.xy
            self.artists[f'vp_{i}'][0].set_xy(np.column_stack((x, y)))

    def redraw(self):

        # self.artists = {}
        # self.axs[0].clear()
        # self.firstplot(from_scratch=False)

        self.update_dynamic_artists()

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
            self.remove_point(event.ind[0])

    def on_click(self, event):
        if event.button == MouseButton.LEFT and event.inaxes is self.axs[0]:
            print('add point', event)
            self.add_point((event.xdata, event.ydata))
            self.update()
            self.plt_update_show()

    def on_close(self, event):
        print("Received close event")
        print(event)

    def on_key(self, event):
        if event.key == 'ctrl+r':
            self.redraw()
            self.plt_update_show()
            print('redraw')
        elif event.key == 'ctrl+z':
            self.undo()
            print('undo')
        elif event.key == 'R':
            self.cov.clear_user_pts()
            self.history = []
            self.redraw()
            self.plt_update_show()
            print('reset (this cannot be undone)')

    def new_surface(self):
        if not self.threed:
            return
        if '3dsurf' in self.artists:
            self.artists['3dsurf'].remove()
        self.artists['3dsurf'] = self.axs[1].plot_surface(
            self.X, self.Y, self.Z, cmap=cm.coolwarm, antialiased=True)

