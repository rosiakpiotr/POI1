import pyransac3d as pyrsc
import numpy as np
import utils


def compute_distance_from_plane(points, plane_eq):
    return np.abs((np.dot(points, plane_eq[:3]) + plane_eq[3]) / np.linalg.norm(plane_eq[:3]))


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
        distance_all_points = compute_distance_from_plane(points, [*plane_vector, plane_constant])
        inliers = points[distance_all_points < inlier_threshold]
        current_model_size = len(inliers)
        if current_model_size > model_size:
            model_size = current_model_size
            plane_vec_best = plane_vector
            plane_const_best = plane_constant
            plane_all_distance_best = distance_all_points
        max_iter = max_iter - 1

    return [*(plane_vec_best / np.linalg.norm(plane_vec_best)), plane_const_best], plane_all_distance_best


if __name__ == '__main__':
    point_cloud_filenames = [
        'horizontal_plane.xyz',
        'vertical_plane.xyz',
        'vertical_cylinder.xyz'
    ]

    for filename in point_cloud_filenames:
        cloud_points = np.array(list(utils.read_point_cloud_csv(filename)))
        threshold = 5
        max_iterations = int(1e2)

        best_eq, best_inliners = pyrsc.Plane().fit(cloud_points, threshold, maxIteration=max_iterations)
        plane_eq, points_distances = plane_ransac(cloud_points, threshold, max_iterations)

        def rate_and_print(title, eq, distances):
            def is_cylinder():
                return (points_distances > threshold).sum() > (len(cloud_points) * 0.5)

            def is_horizontal_plane():
                return abs(eq[2]) > 0.95

            print(title, "Ax,By,Cz,D:", f'[{(("{:.3f}, "*4).format(*eq))[:-2]}]',
                  "powierzchnia cylindryczna" if is_cylinder() else
                  ("płaszczyzna pozioma " if is_horizontal_plane() else "płaszczyzna pionowa"))


        rate_and_print("Implementacja własna:", plane_eq, points_distances)
        rate_and_print("pyransac3d:", best_eq, compute_distance_from_plane(cloud_points, best_eq))
