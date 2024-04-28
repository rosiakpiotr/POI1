import numpy as np
import csv


def point_cloud_read(filename: str):
    with open(filename) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for x, y, z in reader:
            yield float(x), float(y), float(z)


def plane_ransac(points, inlier_threshold: float, max_iter=1e2):
    model_size = 0
    plane_vec_best = 0
    plane_const_best = 0
    plane_all_distance_best = 0

    while max_iter > 0:
        # Select 3 random points
        point_a, point_b, point_c = points[
            np.random.choice(len(points), size=3, replace=False)
        ]
        # Find plane equation coefficients (fit)
        vector_a = point_a - point_c
        vector_b = point_b - point_c
        normalised_a = vector_a / np.linalg.norm(vector_a)
        normalised_b = vector_b / np.linalg.norm(vector_b)
        plane_vector = np.cross(normalised_a, normalised_b)
        plane_constant = -np.sum(np.multiply(plane_vector, point_c))
        # Compute number of inliers
        distance_all_points = np.abs((np.dot(points, plane_vector) + plane_constant) / np.linalg.norm(plane_vector))
        inliers = points[distance_all_points < inlier_threshold]
        current_model_size = len(inliers)
        if current_model_size > model_size:
            model_size = current_model_size
            plane_vec_best = plane_vector
            plane_const_best = plane_constant
            plane_all_distance_best = distance_all_points
        max_iter = max_iter - 1

    return plane_vec_best / np.linalg.norm(plane_vec_best), plane_const_best, plane_all_distance_best


if __name__ == '__main__':
    point_cloud_filenames = [
        'horizontal_plane.xyz',
        'vertical_plane.xyz',
        'vertical_cylinder.xyz'
    ]

    for filename in point_cloud_filenames:
        cloud_points = np.array(list(point_cloud_read(filename)))
        threshold = 5
        max_iterations = 1e2
        plane_vec, plane_const, points_distances = plane_ransac(cloud_points, threshold, max_iterations)


        def is_cylinder():
            return (points_distances > threshold).sum() > (len(cloud_points) * 0.5)


        def is_horizontal_plane():
            return abs(plane_vec[2]) > 0.95


        print(plane_vec,
              "powierzchnia cylindryczna" if is_cylinder() else
              ("płaszczyzna pozioma "if is_horizontal_plane() else "płaszczyzna pionowa"))
