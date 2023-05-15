import numpy as np

"""
Thanks to ChatGPT.

Prompt: online kmeans with dynamic size of cluster along streaming.

Performing online k-means with a dynamically changing cluster size is a more complex task. It requires adapting the traditional k-means algorithm to accommodate changes in the number of clusters as new data points arrive. One approach to achieve this is the use of a variation of k-means called "K-Means with Dynamic Cluster Creation and Deletion" (KMC2).

KMC2 is an algorithm that dynamically adjusts the cluster centroids and sizes as new data arrives. Here's an example of how you can implement KMC2 for online k-means with
"""


class DynamicKMeansMaxCluster:
    def __init__(self, threshold, max_clusters=5):
        """
        Streaming KMeans with maximum cluster size.

        Parameters
        ----------
        threshold: float, optional (default=0.1)
            Minimum threshold to consider new cluster.
        max_clusters: int, optional (default=5)
            max cluster size.
        """

        self.max_clusters = max_clusters
        self.threshold = threshold
        self.cluster_centers = []
        self.cluster_sizes = []
        self.labels = []

    def fit(self, data):
        for sample in data:
            self.streaming(sample)

    def streaming(self, sample):
        if len(self.cluster_centers) == 0:
            self.cluster_centers.append(sample)
            self.cluster_sizes.append(1)
            self.labels.append(0)
            nearest_cluster = 0
        else:
            distances = [np.linalg.norm(sample - center) for center in self.cluster_centers]
            nearest_cluster = np.argmin(distances)

            if distances[nearest_cluster] <= self.threshold:
                self.cluster_centers[nearest_cluster] = (
                    self.cluster_centers[nearest_cluster] * self.cluster_sizes[nearest_cluster] + sample) / (
                    self.cluster_sizes[nearest_cluster] + 1)
                self.cluster_sizes[nearest_cluster] += 1
            elif len(self.cluster_centers) < self.max_clusters:
                self.cluster_centers.append(sample)
                self.cluster_sizes.append(1)
            else:
                distances_to_centers = [
                    np.linalg.norm(
                        sample -
                        center) for center in self.cluster_centers]
                farthest_cluster = np.argmax(distances_to_centers)
                if self.cluster_sizes[farthest_cluster] > 1:
                    self.cluster_centers[farthest_cluster] = (
                        self.cluster_centers[farthest_cluster] * self.cluster_sizes[farthest_cluster] - sample) / (
                        self.cluster_sizes[farthest_cluster] - 1)
                    self.cluster_sizes[farthest_cluster] -= 1
                else:
                    self.cluster_centers.pop(farthest_cluster)
                    self.cluster_sizes.pop(farthest_cluster)

            distances = [np.linalg.norm(sample - center) for center in self.cluster_centers]
            nearest_cluster = np.argmin(distances)
            self.labels.append(nearest_cluster)

        return nearest_cluster


class DynamicKMeans:
    def __init__(self, threshold=0.1):
        """
        Streaming KMeans with no maximum cluster size.

        Parameters
        ----------
        threshold: float, optional (default=0.1)
            Minimum threshold to consider new cluster.
        """

        self.threshold = threshold
        self.cluster_centers = []
        self.cluster_sizes = []
        self.labels = []

    def fit(self, data):
        for sample in data:
            self.streaming(sample)

    def streaming(self, sample):
        if len(self.cluster_centers) == 0:
            self.cluster_centers.append(sample)
            self.cluster_sizes.append(1)
            self.labels.append(0)
            nearest_cluster = 0

        else:
            distances = [np.linalg.norm(sample - center) for center in self.cluster_centers]
            nearest_cluster = np.argmin(distances)

            if distances[nearest_cluster] <= self.threshold:
                self.cluster_centers[nearest_cluster] = (
                    self.cluster_centers[nearest_cluster] * self.cluster_sizes[nearest_cluster] + sample) / (
                    self.cluster_sizes[nearest_cluster] + 1)
                self.cluster_sizes[nearest_cluster] += 1
            else:
                min_size_cluster = np.argmin(self.cluster_sizes)
                if self.cluster_sizes[min_size_cluster] > 1:
                    self.cluster_centers[min_size_cluster] = (
                        self.cluster_centers[min_size_cluster] * self.cluster_sizes[min_size_cluster] - sample) / (
                        self.cluster_sizes[min_size_cluster] - 1)
                    self.cluster_sizes[min_size_cluster] -= 1
                else:
                    self.cluster_centers.append(sample)
                    self.cluster_sizes.append(1)

            distances = [np.linalg.norm(sample - center) for center in self.cluster_centers]
            nearest_cluster = np.argmin(distances)
            self.labels.append(nearest_cluster)

        return nearest_cluster
