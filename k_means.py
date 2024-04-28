from sklearn.cluster import KMeans
import csv
import numpy as np
import matplotlib.pyplot as plt


def point_cloud_read(filename: str):
    with open(filename) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for x, y, z in reader:
            yield float(x), float(y), float(z)


horizontal_plane_points = list(point_cloud_read('horizontal_plane.xyz'))
vertical_plane_points = list(point_cloud_read('vertical_plane.xyz'))
cylinder_points = list(point_cloud_read('vertical_cylinder.xyz'))

all_points = np.array([*horizontal_plane_points, *vertical_plane_points, *cylinder_points])

classifier = KMeans(n_clusters=3)
clustered = classifier.fit_predict(all_points)

h = clustered == 0
v = clustered == 1
c = clustered == 2


def plot_clustered(data, results):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[results, 0], data[results, 1], data[results, 2], s=0.5)
    plt.show(block=False)


plot_clustered(all_points, h)
plot_clustered(all_points, v)
plot_clustered(all_points, c)
plt.show()
