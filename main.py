import numpy as np

from scipy.stats import uniform
from csv import writer


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


def save_point_cloud_csv(filename: str, x, y, z):
    points = zip(x, y, z)
    with open(filename, 'w', newline='\n') as csv_file:
        csv_writer = writer(csv_file)
        for p in points:
            csv_writer.writerow(p)


num_points_planes = int(1e4)
num_points_cylinder = int(3e4)
save_point_cloud_csv('horizontal_plane.xyz', *generate_horizontal(200, 200, 5, num_points_planes))
save_point_cloud_csv('vertical_plane.xyz', *generate_vertical(5, 200, 200, num_points_planes))
save_point_cloud_csv('vertical_cylinder.xyz', *generate_cylinder(100, 10, 200, num_points_cylinder))
