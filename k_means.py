from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt

import utils

all_points = np.array(list(utils.read_point_cloud_csv('all_points.xyz')))

classifier = KMeans(n_clusters=3)
clustered = classifier.fit_predict(all_points)

# Actually nothing tells which clustered part is what
# setting variable names to 'horizontal' or 'vertical'
# is not correct as there is no way to tell.
horizontal = clustered == 0
vertical = clustered == 1
cylinder = clustered == 2


def plot_clustered(data, results, title: str):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[results, 0], data[results, 1], data[results, 2], s=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(title)
    plt.show(block=False)


def prepare_title(num, results, total_count):
    return f'Klaster {num} ({100 * results.sum() / total_count}% punktów)'


point_count = len(all_points)
plot_clustered(all_points, horizontal, prepare_title(1, horizontal, point_count))
plot_clustered(all_points, vertical, prepare_title(2, vertical, point_count))
plot_clustered(all_points, cylinder, prepare_title(3, cylinder, point_count))
plt.show()
