from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt

import utils


def run_k_means_on(points: np.array, n_clusters: int, case_idx: int):
    classifier = KMeans(n_clusters=n_clusters)
    clustered = classifier.fit_predict(points)

    fig = plt.figure(figsize=(7, 6), dpi=300)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(f'Klastrowanie K-Średnich (k=3), przypadek {case_idx + 1}')

    def plot_clustered(data, results):
        ax.scatter(data[results, 0], data[results, 1], data[results, 2], s=0.5)

    for cluster in range(3):
        plot_clustered(all_points, clustered == cluster)
    # plt.show()
    fig.tight_layout()
    plt.savefig(f'{utils.PLOTS_BASE_DIR_REL}/k_means_case_{case_idx + 1}')


if __name__ == '__main__':
    BASE_DIR = utils.POINT_CLOUD_BASE_DIR_REL

    for case_idx in range(3):
        all_points = np.array(list(
            utils.read_point_cloud_csv(f'{BASE_DIR}/all_points_{case_idx}.xyz'))
        )
        run_k_means_on(all_points, 3, case_idx)
