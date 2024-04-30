from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

import utils


def run_dbscan_on(points: np.array, eps, min_points, case_idx):
    # https://www.reneshbedre.com/blog/dbscan-python.html
    classifier = DBSCAN(eps=eps, min_samples=min_points)
    clustered = classifier.fit_predict(all_points)
    counter = Counter(clustered)
    c_count = len(counter.keys())

    fig = plt.figure(figsize=(7, 6), dpi=300)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.suptitle(f"Przypadek {case_idx + 1}")
    title = "Klastrowanie DBSCAN (eps={:.3f},min_points={:d})".format(eps, min_points)
    plt.title(title)

    def plot_scatter(data, results):
        ax.scatter(data[results, 0], data[results, 1], data[results, 2], s=0.5)

    for c in range(c_count):
        plot_scatter(points, clustered == c)

    # plt.show()
    fig.tight_layout()
    plt.savefig(f'{utils.PLOTS_BASE_DIR_REL}/dbscan_case_{case_idx + 1}')


def find_eps_for_3d(points, case_idx):
    nbrs = NearestNeighbors(n_neighbors=3 * 2 + 1).fit(points)
    neigh_dist, neigh_ind = nbrs.kneighbors(points)
    sort_neigh_dist = np.sort(neigh_dist, axis=0)

    k_dist = sort_neigh_dist[:, 6]

    plt.figure()
    plt.plot(k_dist)
    plt.ylabel("k-NN distance")
    plt.xlabel("Sorted observations (6th NN)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{utils.PLOTS_BASE_DIR_REL}/dbscan_knn_case_{case_idx + 1}')
    # plt.show()

    kneedle = KneeLocator(x=range(1, len(neigh_dist) + 1), y=k_dist, S=1.0,
                          curve="concave", direction="increasing", online=True)

    # get the estimate of knee point
    print("Optymalny epsilon: ", kneedle.knee_y)
    return kneedle.knee_y


if __name__ == '__main__':
    BASE_DIR = utils.POINT_CLOUD_BASE_DIR_REL
    for case_idx in range(3):
        all_points = np.array(list(
            utils.read_point_cloud_csv(f'{BASE_DIR}/all_points_{case_idx}.xyz'))
        )
        eps = find_eps_for_3d(all_points, case_idx)
        run_dbscan_on(all_points, eps, 3 * 2, case_idx)
