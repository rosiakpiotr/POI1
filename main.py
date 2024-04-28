from scipy.stats import uniform
from csv import writer


def generate_uniform_cloud(width, height, depth, points_per_dim: int):
    distribution_x = uniform(loc=-width/2, scale=width)
    distribution_y = uniform(loc=-height/2, scale=height)
    distribution_z = uniform(loc=-depth/2, scale=depth)

    x = distribution_x.rvs(size=points_per_dim)
    y = distribution_y.rvs(size=points_per_dim)
    z = distribution_z.rvs(size=points_per_dim)

    return x, y, z


def generate_horizontal(points_per_dim: int):
    return generate_uniform_cloud(200, 200, 5, points_per_dim)


def generate_vertical(points_per_dim: int):
    return generate_uniform_cloud(5, 200, 200, points_per_dim)


def save_point_cloud_csv(filename: str, x, y, z):
    points = zip(x, y, z)
    with open(filename, 'w', newline='\n') as csv_file:
        csv_writer = writer(csv_file)
        for p in points:
            csv_writer.writerow(p)


num_points = int(1e4)
save_point_cloud_csv('horizontal_plane.xyz', *generate_horizontal(num_points))
save_point_cloud_csv('vertical_plane.xyz', *generate_vertical(num_points))
