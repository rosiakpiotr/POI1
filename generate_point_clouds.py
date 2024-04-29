import numpy as np
import utils

from scipy.stats import uniform


def generate_uniform_cloud(width, height, depth, loc_x, loc_y, loc_z, points_per_dim: int):
    distribution_x = uniform(loc=loc_x, scale=width)
    distribution_y = uniform(loc=loc_y, scale=height)
    distribution_z = uniform(loc=loc_z, scale=depth)

    x = distribution_x.rvs(size=points_per_dim)
    y = distribution_y.rvs(size=points_per_dim)
    z = distribution_z.rvs(size=points_per_dim)

    return x, y, z


def generate_horizontal(width, height, depth, loc_x, loc_y, loc_z, points_per_dim: int):
    return generate_uniform_cloud(width, height, depth, loc_x, loc_y, loc_z, points_per_dim)


def generate_vertical(width, height, depth, loc_x, loc_y, loc_z, points_per_dim: int):
    return generate_uniform_cloud(width, height, depth, loc_x, loc_y, loc_z, points_per_dim)


def generate_cylinder(radius, thickness, depth, loc_center, loc_z, points_per_dim: int):
    distribution_radius = uniform(loc=radius, scale=thickness)
    distribution_z = uniform(loc=loc_z, scale=depth)

    radius_points = distribution_radius.rvs(size=points_per_dim)
    angles = np.linspace(0, 2 * np.pi, points_per_dim)
    # Distribute linearly random radius on the circle
    x = radius_points * np.sin(angles) + loc_center
    y = radius_points * np.cos(angles) + loc_center
    z = distribution_z.rvs(size=points_per_dim)

    return x, y, z


if __name__ == '__main__':
    BASE_DIR = utils.POINT_CLOUD_BASE_DIR_REL
    save_merged_file = True
    num_points_planes = int(1e4)
    num_points_cylinder = int(1e4)

    # Definition of "dimensions" and center locations of point clouds
    # for horizontal, vertical and cylindrical surfaces.
    cases = (
        ([200, 200, 5, -200 / 2, -200 / 2, -5 / 2],  # Horizontal
         [5, 200, 200, -5 / 2, -200 / 2, -200 / 2],  # Vertical
         [100, 10, 200, -100, -200 / 2]),  # Cylindrical

        ([200, 200, 5, 200, 200, 5],  # Horizontal
         [5, 200, 200, 5, 200, 200],  # Vertical
         [100, 10, 200, 100, 200]),  # Cylindrical

        ([200, 200, 5, 0, 0, 0],  # Horizontal
         [5, 200, 200, 0, 0, 0],  # Vertical
         [100, 10, 200, 100, 0]),  # Cylindrical
    )

    for num, case in enumerate(cases):
        horizontal_dims, vertical_dims, cylindrical_dims = case

        hpp = generate_horizontal(*horizontal_dims, num_points_planes)
        vpp = generate_vertical(*vertical_dims, num_points_planes)
        cpp = generate_cylinder(*cylindrical_dims, num_points_cylinder)

        utils.save_point_cloud_csv(f'{BASE_DIR}/horizontal_plane_{num}.xyz', *hpp)
        utils.save_point_cloud_csv(f'{BASE_DIR}/vertical_plane_{num}.xyz', *vpp)
        utils.save_point_cloud_csv(f'{BASE_DIR}/vertical_cylinder_{num}.xyz', *cpp)
        if save_merged_file:
            concatenated = np.concatenate((hpp, vpp, cpp), axis=1)
            utils.save_point_cloud_csv(f'{BASE_DIR}/all_points_{num}.xyz', concatenated[0, :], concatenated[1, :], concatenated[2, :])
