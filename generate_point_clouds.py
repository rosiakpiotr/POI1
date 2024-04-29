import numpy as np
import utils

from scipy.stats import uniform


def generate_uniform_cloud(width, height, depth, points_per_dim: int):
    distribution_x = uniform(loc=-width / 2, scale=width)
    distribution_y = uniform(loc=-height / 2, scale=height)
    distribution_z = uniform(loc=-depth / 2, scale=depth)

    x = distribution_x.rvs(size=points_per_dim)
    y = distribution_y.rvs(size=points_per_dim)
    z = distribution_z.rvs(size=points_per_dim)

    return x, y, z


def generate_horizontal(width, height, depth, points_per_dim: int):
    return generate_uniform_cloud(width, height, depth, points_per_dim)


def generate_vertical(width, height, depth, points_per_dim: int):
    return generate_uniform_cloud(width, height, depth, points_per_dim)


def generate_cylinder(radius, thickness, depth, points_per_dim: int):
    distribution_radius = uniform(loc=-radius, scale=thickness)
    distribution_z = uniform(loc=-depth / 2, scale=depth)

    radius_points = distribution_radius.rvs(size=points_per_dim)
    angles = np.linspace(0, 2 * np.pi, points_per_dim)
    # Distribute linearly random radius on the circle
    x = radius_points * np.sin(angles)
    y = radius_points * np.cos(angles)
    z = distribution_z.rvs(size=points_per_dim)

    return x, y, z


if __name__ == '__main__':
    save_merged_file = True
    num_points_planes = int(1e4)
    num_points_cylinder = int(1e4)

    hpp = generate_horizontal(200, 200, 5, num_points_planes)
    vpp = generate_vertical(5, 200, 200, num_points_planes)
    cpp = generate_cylinder(100, 10, 200, num_points_cylinder)

    utils.save_point_cloud_csv('horizontal_plane.xyz', *hpp)
    utils.save_point_cloud_csv('vertical_plane.xyz', *vpp)
    utils.save_point_cloud_csv('vertical_cylinder.xyz', *cpp)
    if save_merged_file:
        concatenated = np.concatenate((hpp, vpp, cpp), axis=1)
        utils.save_point_cloud_csv('all_points.xyz', concatenated[0, :], concatenated[1, :], concatenated[2, :])

