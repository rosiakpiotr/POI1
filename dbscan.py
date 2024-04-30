from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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
    plt.suptitle(f"Przypadek {case_idx+1}")
    title = "Klastrowanie DBSCAN (eps={:.3f},min_points={:d})".format(eps, min_points)
    plt.title(title)

    def plot_scatter(data, results):
        ax.scatter(data[results, 0], data[results, 1], data[results, 2], s=0.5)

    for c in range(c_count):
        plot_scatter(points, clustered == c)

    # plt.show()
    fig.tight_layout()
    plt.savefig(f'{utils.PLOTS_BASE_DIR_REL}/dbscan_case_{case_idx + 1}')


if __name__ == '__main__':
    BASE_DIR = utils.POINT_CLOUD_BASE_DIR_REL
    for case_idx in range(3):
        all_points = np.array(list(
            utils.read_point_cloud_csv(f'{BASE_DIR}/all_points_{case_idx}.xyz'))
        )
        run_dbscan_on(all_points, 8.4, 3*2, case_idx)
