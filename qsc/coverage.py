from numpy.typing import NDArray
import numpy as np
import shapely
from shapely import MultiPoint, Polygon


def gaussian_kernel(r, epsilon, mult):
    return 1 + mult * np.exp(-(epsilon * r)**2)


def inverse_quadratic_kernel(r, epsilon):
    return 1 / (1 + np.exp((epsilon * r)**2))


def inverse_multiquadratic_kernel(r, epsilon):
    return 1 / np.sqrt(1 + (epsilon * r)**2)


def rbf(X, center, epsilon, mult, kernel):
    return kernel(np.linalg.norm(X - center, axis=1), epsilon=epsilon, mult=mult)

def mesh2coords(*args):
    return np.vstack([x.ravel() for x in args]).T

class CoverageAlg:
    kernel = staticmethod(gaussian_kernel)

    def __init__(self, domain: Polygon, agent_pts, epsilon: float, mult: float):
        self.user_pts = []
        self.domain = domain
        self.positions = agent_pts

        # TODO(mabay): Find optimal values for the kernel's parameter
        self.epsilon = epsilon
        self.mult = mult

        self.vpolys = self._compute_voronoi_cells(self.positions, self.domain)
        self.centroids = [
            self._compute_centroid(poly, self._uniform_density)
            for poly in self.vpolys
        ]

    def update(self):
        """ Recompute the voronoi cells based on the user density """
        self.vpolys = self._compute_voronoi_cells(
            self.positions, self.domain)

        density_func = self._user_density
        if len(self.user_pts) == 0:
            density_func = self._uniform_density

        self.centroids = [
            self._compute_centroid(poly, density_func)
            for poly in self.vpolys
        ]

    def compute_density(self, Xs, Ys):
        Zs = np.zeros(Xs.size)
        for point in self.user_pts:
            Zs += rbf(mesh2coords(Xs, Ys), point,
                          epsilon=self.epsilon, mult=self.mult,
                          kernel=self.kernel)
        Zs = Zs.reshape(Xs.shape)
        return Zs

    def add_point(self, index: int, point):
        self.user_pts.insert(index, point)

    def remove_point(self, index: int):
        return self.user_pts.pop(index)

    def clear_user_pts(self):
        self.user_pts.clear()

    def _compute_centroid(self, polygon: Polygon, density_func, grid_res=64):
        """
        Computes the centroid of a polygon given a custom density function.

        Parameters:
            polygon (Polygon): a shapely polygon.
            density_func: A callable that takes two arrays (x, y) and
            returns a density array.
            grid_res (float): Resolution of the grid for numerical
            integration (higher = more accurate).

        Returns:
            (cx, cy): The coordinates of the weighted centroid.
        """

        # Early exit
        if not self.user_pts:
            c = polygon.centroid
            return c.x, c.y

        # 1. Find the bounding box of the polygon
        min_x, min_y, max_x, max_y = polygon.bounds

        # 2. Create a dense grid within the bounding box
        x_grid, dx = np.linspace(min_x, max_x, num=grid_res, retstep=True)
        y_grid, dy = np.linspace(min_y, max_y, grid_res, retstep=True)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Flatten the grid into an array of (x, y) points
        points = np.vstack((X.ravel(), Y.ravel())).T

        # 3. Filter out points that are outside the polygon
        mask = shapely.contains_xy(polygon, points)
        inside_points = points[mask]

        # Fallback to standard geometric centroid if the polygon is too small for the grid
        if len(inside_points) == 0:
            raise ValueError("No points inside")
            return np.mean(vertices[:, 0]), np.mean(vertices[:, 1])

        inside_x, inside_y = inside_points[:, 0], inside_points[:, 1]

        # 4. Evaluate the density function at the interior points
        densities = density_func(inside_x, inside_y)

        # 5. Compute the center of mass (weighted average)
        total_mass = np.sum(densities)

        if total_mass == 0:
            raise ValueError("Total mass is zero. Check your density function limits.")

        cx = np.sum(inside_x * densities) / total_mass
        cy = np.sum(inside_y * densities) / total_mass

        return cx, cy

    def _compute_voronoi_cells(self, points: NDArray[np.float64], domain: Polygon) -> list[Polygon]:
        """
        points: np array of shape (n, 2)
        domain: shapely polygon
        returns: list of voronoi cells as shapely polygons
        """
        vp = shapely.voronoi_polygons(MultiPoint(points), extend_to=domain)
        return [poly.intersection(domain) for poly in vp.geoms]

    def _uniform_density(self, x, y):
        assert len(x) == len(y)
        return np.ones(len(x))

    def _user_density(self, x, y):
        if len(self.user_pts) == 0:
            return 0.0

        output = 0.0
        for point in self.user_pts:
            output += rbf(
                np.array([x, y]).T, point, epsilon=self.epsilon,
                mult=self.mult, kernel=self.kernel)

        return output
