"""Interactive plot for visualizing RBF density functions.

You can add points by clicking on the plot, and remove points by right-clicking on them.
You can undo changes by pressing ctrl+z.
shift+R clears the plot. This cannot be undone.

DensityPlot.threed controls whether or not to plot a 3D surface.
"""

import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
matplotlib.use('TkAgg')

import shapely
from shapely import MultiPoint, Polygon
from numpy.typing import NDArray


def mesh2d(xmin, xmax, ymin, ymax, nx, ny=None):
    if ny is None:
        ny = nx
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def gaussian_kernel(r, epsilon):
    return 1+ np.exp(-(epsilon * r)**2)


def inverse_quadratic_kernel(r, epsilon):
    return 1 / (1 + np.exp((epsilon * r)**2))


def inverse_multiquadratic_kernel(r, epsilon):
    return 1 / np.sqrt(1 + (epsilon * r)**2)


def rbf(X, center, epsilon, kernel=gaussian_kernel):
    return kernel(np.linalg.norm(X - center, axis=1), epsilon=epsilon)


def mesh2coords(*args):
    return np.vstack([x.ravel() for x in args]).T


class DensityPlot:
    kernel = staticmethod(gaussian_kernel)
    threed = True

    def __init__(self, domain: Polygon, agent_pts):
        """
        Params
        ------
        domain: shapely polygon

        agent_pts: np array of size (n_agents, 2)
        """
        
        self.user_pts = []

        self.domain = domain
        minx, miny, maxx, maxy = self.domain.bounds
        self.epsilon = 0.1

        self.agent_pts = agent_pts

        self.voronoi_polys = DensityPlot.mabay_compute_voronoi_cells(agent_pts, self.domain)
        self.centroids = [
            DensityPlot.mabay_compute_centroid(poly, self._uniform_density)
            for poly in self.voronoi_polys
        ]

        self.xrange = (minx, maxx)
        self.yrange = (miny, maxy)
        self.resolution = 0.1
        self.artists = {}
        self.handlers = {}
        self.setup()
        self.attach_handlers()
        self.history = []

    def _uniform_density(self, x, y):
        assert len(x) == len(y)
        return np.ones(len(x))
    
    @staticmethod
    def mabay_compute_centroid(polygon: Polygon, density_func, grid_res=200):
        """
        Computes the centroid of a polygon given a custom density function.
        
        Parameters:
        - polygon: a shapely polygon.
        - density_func: A callable that takes two arrays (x, y) and returns a density array.
        - grid_res: Resolution of the grid for numerical integration (higher = more accurate).
        
        Returns:
        - (cx, cy): The coordinates of the weighted centroid.
        """
        
        # 1. Find the bounding box of the polygon
        min_x, min_y, max_x, max_y = polygon.bounds
        
        # 2. Create a dense grid within the bounding box
        x_grid = np.linspace(min_x, max_x, grid_res)
        y_grid = np.linspace(min_y, max_y, grid_res)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Flatten the grid into an array of (x, y) points
        points = np.vstack((X.ravel(), Y.ravel())).T
        
        # 3. Filter out points that are outside the polygon
        mask = shapely.contains_xy(polygon, points)
        inside_points = points[mask]
        
        # # Fallback to standard geometric centroid if the polygon is too small for the grid
        # if len(inside_points) == 0:
        #     return np.mean(vertices[:, 0]), np.mean(vertices[:, 1])
            
        inside_x = inside_points[:, 0]
        inside_y = inside_points[:, 1]
        
        # 4. Evaluate the density function at the interior points
        densities = density_func(inside_x, inside_y)
        
        # 5. Compute the center of mass (weighted average)
        total_mass = np.sum(densities)
        
        if total_mass == 0:
            raise ValueError("Total mass is zero. Check your density function limits.")
            
        cx = np.sum(inside_x * densities) / total_mass
        cy = np.sum(inside_y * densities) / total_mass
        
        return cx, cy

    @staticmethod
    def mabay_compute_voronoi_cells(points: NDArray[np.float64], domain: Polygon) -> list[Polygon]:
        """
        points: np array of shape (n, 2)
        domain: shapely polygon
        returns: list of voronoi cells as shapely polygons
        """
        vp = shapely.voronoi_polygons(MultiPoint(points), extend_to=domain)
        return [poly.intersection(domain) for poly in vp.geoms]

    def setup(self):
        size = (10, 5) if self.threed else (5.5, 5)
        self.fig = plt.figure(figsize=size)
        ax = self.fig.add_subplot(1, 2 if self.threed else 1, 1)
        ax3d = self.fig.add_subplot(1, 2, 2, projection='3d') if self.threed else None
        # axw = ax.twinx()
        self.axs: list[plt.Axes] = [ax, ax3d]
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

    def _user_density(self, x, y):
        if len(self.user_pts) == 0:
            return 0.0

        output = 0.0
        for point in self.user_pts:
            output += rbf(np.array([x, y]).T, point, epsilon=self.epsilon, kernel=self.kernel)

        return output

    def recalculate_density(self):
        self.Z = np.zeros(self.X.size)
        for point in self.user_pts:
            self.Z += rbf(mesh2coords(self.X, self.Y), point, epsilon=self.epsilon, kernel=self.kernel)
        self.Z = self.Z.reshape(self.X.shape)

    def recompute_voronoi(self, cells=False, centroids=False):
        if cells:
            self.voronoi_polys = DensityPlot.mabay_compute_voronoi_cells(
                self.agent_pts, self.domain)
        
        if centroids:
            density_func = self._user_density
            if len(self.user_pts) == 0:
                density_func = self._uniform_density

            self.centroids = [
                DensityPlot.mabay_compute_centroid(poly, density_func)
                for poly in self.voronoi_polys
            ]


    def firstplot(self, points=None):
        if points is not None:
            self.user_pts = points
        self.set_mesh()
        self.recalculate_density()
        ar = self.artists
        ar['2dimg'] = self.axs[0].imshow(self.Z, extent=(*self.xrange, *self.yrange), origin='lower', zorder=0)
        x, y = np.asarray(self.user_pts).T if self.user_pts else (np.empty((0)), np.empty((0)))
        ar['2dpts'] = self.axs[0].scatter(x, y, s=50, c='r', alpha=0.3, marker='.', zorder=2)
        ar['2dpts'].set_picker(True)

        self.new_contour()
        self.new_surface()

        ag_x, ag_y = np.asarray(self.agent_pts).T
        ar['agent_pts'] = self.axs[0].scatter(ag_x, ag_y, s=50, c='r', alpha=1, marker='x', zorder=4)

        for i, poly in enumerate(self.voronoi_polys):
            x, y = poly.exterior.xy
            ar[f'vp_{i}'] = self.axs[0].fill(x, y, alpha=0.4, edgecolor="black")

        ct_x, ct_y = np.asarray(self.centroids).T if self.centroids else (np.empty((0)), np.empty((0)))
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
        self.user_pts.insert(index, point)
        self.recalculate_density()

    def remove_point(self, index, add_to_history=True, update=True):
        point = self.user_pts.pop(index)
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
        self.recalculate_density(update_centroids=True)
        self.artists['2dimg'].set_extent((*self.xrange, *self.yrange))
        self.update()
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

    def redraw(self):
        for artist in self.artists.values():
            if isinstance(artist, list):
                for el in artist:
                    el.remove()
            else:
                artist.remove()

        self.artists = {}
        self.axs[0].clear()
        self.firstplot()

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
            self.user_pts = []
            self.history = []
            self.redraw()
            self.plt_update_show()
            print('reset (this cannot be undone)')

    def new_surface(self):
        if not self.threed:
            return
        if '3dsurf' in self.artists:
            self.artists['3dsurf'].remove()
        self.artists['3dsurf'] = self.axs[1].plot_surface(self.X, self.Y, self.Z, cmap=cm.coolwarm, antialiased=True)


points = [(-10, 10), (0, 0), (10, -10)]
limits = ((-15, 15), (-15, 15))


def example_density(points=points, subdivide=1000, limits=limits):
    xmin, xmax, ymin, ymax = limits
    ys = np.linspace(-xmin, xmax, subdivide)
    xs = np.linspace(-ymin, ymax, subdivide)

    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros(X.size)
    for point in points:
        Z += rbf(mesh2coords(X, Y), point, epsilon=0.3, kernel=inverse_multiquadratic_kernel)

    Z = Z.reshape(X.shape)
    return X, Y, Z


def test_plot_density():
    X, Y, Z = example_density()

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

    def on_click(event):
        if event.button == MouseButton.LEFT:
            print(event)

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.connect('button_press_event', on_click)
    plt.show()


def test_plot_density2():
    def on_click(event):
        if event.button == MouseButton.LEFT:
            print(event.xdata, event.ydata)

    X, Y, Z = example_density()

    plt.imshow(Z, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower')
    plt.contour(X, Y, Z)

    plt.connect('button_press_event', on_click)
    plt.show()


if __name__ == "__main__":
    n_agents = 6
    SIDE_LEN = 8
    domain = shapely.box(0, 0, SIDE_LEN, SIDE_LEN)

    rng = np.random.default_rng(3)
    agent_pts = rng.uniform(low=0, high=SIDE_LEN, size=(n_agents, 2))

    # test_plot_density2()
    dp = DensityPlot(domain, agent_pts)
    dp.firstplot()
    dp.update()
    dp.recalculate_density()
    dp.plt_update_show()
    plt.draw()
    plt.pause(0.1)
    plt.show(block=True)
