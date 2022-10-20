import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class A:
    def __init__(self, ax):
        self.ax = ax

        # points
        np.random.seed(4321)  # setup seed to reproduce random values
        self.data = np.random.randint(0, 10, [11, 2])

        # init centers
        np.random.seed(0)
        self.centroids = np.random.randint(0, 10, [3, 2])

        # colors
        self.colors = ['red', 'green', 'blue']

        # functions
        self.plot_functions = [
            self.f_1,
            self.f_2,
            self.f_3,
            self.f_4,
            self.f_5,
            self.f_6,
            self.f_7,
            self.f_8
        ]

    def plot_points_links(self, points, centers):
        self.ax.scatter(points[:, 0], points[:, 1], c='black')
        colors = ['red', 'green', 'blue']
        self.ax.scatter(centers[:, 0], centers[:, 1], c=colors, marker='x')
        for i, c in enumerate(centers):
            a = np.insert(points.astype('float'), 0, c[0], axis=1)
            a = np.insert(a, 2, c[1], axis=1)
            for b in a:
                self.ax.plot(b[:2], b[2:], '-.', c=colors[i], alpha=0.5)

    def plot_points_type(self, points, dists, centers):
        min_d = np.argmin(dists, axis=1)

        nums = centers.shape[0]
        for i in range(nums):
            members = points[(min_d == i)]
            self.ax.scatter(members[:, 0], members[:, 1], c=self.colors[i])

        self.ax.scatter(centers[:, 0], centers[:, 1], c=self.colors, marker='x')

    @classmethod
    def get_distances(cls, points, centers):
        dists = None
        for c in centers:
            a = np.around(np.sqrt(np.sum(np.square(points - c), axis=1)), decimals=2).reshape(-1, 1)
            if dists is None:
                dists = a
            else:
                dists = np.concatenate((dists, a), axis=1)

        return dists

    def plot_next_centers(self, points, dists, centers):
        min_d = np.argmin(dists, axis=1)

        next_c = None
        for i in range(centers.shape[0]):
            members = points[(min_d == i)]
            plt.scatter(members[:, 0], members[:, 1], c=self.colors[i])

            a = np.average(members, axis=0)
            if next_c is None:
                next_c = a
            else:
                next_c = np.vstack((next_c, a))

        self.ax.scatter(centers[:, 0], centers[:, 1], c=self.colors, marker='x')
        self.ax.scatter(next_c[:, 0], next_c[:, 1], c=self.colors, marker='*')

        for i in range(centers.shape[0]):
            self.ax.arrow(centers[i, 0], centers[i, 1], next_c[i, 0] - centers[i, 0], next_c[i, 1] - centers[i, 1],
                          width=0.05, length_includes_head=True)

        return next_c

    def update(self, frame):
        print(f'frame = {frame}')
        self.ax.clear()
        s = self.plot_functions[frame]()
        self.ax.set_title(s)
        self.ax.set_xlim(-1, 11)
        self.ax.set_ylim(-1, 11)
        return self.ax

    def f_1(self):
        self.ax.scatter(self.data[:, 0], self.data[:, 1], c='black')

        return f'0: There are {self.data.shape[0]} Points to be Parted'

    def f_2(self):
        self.ax.scatter(self.data[:, 0], self.data[:, 1], c='black')
        self.ax.scatter(self.centroids[:, 0], self.centroids[:, 1], c=self.colors, marker='x')

        return f'1: {self.centroids.shape[0]} Random Initialized Centroids'

    def f_3(self):
        self.plot_points_links(self.data, self.centroids)

        return '2: Calculate Distances between All Points and Centroids'

    def f_4(self):
        self.distances = A.get_distances(self.data, self.centroids)
        self.plot_points_type(self.data, self.distances, self.centroids)

        return '3: Cluster Points by Minimum Distances from Centroids'

    def f_5(self):
        self.next_centers = self.plot_next_centers(self.data, self.distances, self.centroids)

        return f'4: Find the Next {self.next_centers.shape[0]} Centroids by Mean Distance of Intra-Cluster'

    def f_6(self):
        self.plot_points_links(self.data, self.next_centers)

        return '5: Recalculate Distances between All Points and the Next Centroids'

    def f_7(self):
        self.distances = A.get_distances(self.data, self.next_centers)
        self.plot_points_type(self.data, self.distances, self.next_centers)

        return '6: Cluster Points by Minimum Distances from the Next Centroids'

    def f_8(self):
        self.plot_points_type(self.data, self.get_distances(self.data, self.next_centers), self.next_centers)

        a = np.argmin(self.get_distances(self.data, self.centroids), axis=1)
        b = np.argmin(self.get_distances(self.data, self.next_centers), axis=1)
        member = self.data[a != b]

        self.ax.scatter(member[:, 0], member[:, 1], s=200, color='orange', marker='o')

        return f'7: Point ({member[:, 0][0]}, {member[:, 1][0]}) Changes the Cluster'


plt.figure()

fig, ax = plt.subplots()

a = A(ax)

ani = FuncAnimation(fig, a.update, frames=list(range(8)), repeat=False, interval=1000)

ani.save('k-means.gif')

plt.show()