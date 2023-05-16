"""
Thanks to ChatGPT.

Prompt: online kmeans with dynamic size of cluster along streaming.

-> Performing online k-means with a dynamically changing cluster size is a more complex task. It requires adapting the traditional k-means algorithm to accommodate changes in the number of clusters as new data points arrive. One approach to achieve this is the use of a variation of k-means called "K-Means with Dynamic Cluster Creation and Deletion" (KMC2).

-> KMC2 is an algorithm that dynamically adjusts the cluster centroids and sizes as new data arrives. Here's an example of how you can implement KMC2 for online k-means with

After I modified a bit to become streaming algorithm.
"""

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Callable


class AgglomerativeClustering:
    def __init__(
        self,
        min_clusters: int,
        max_clusters: int,
        metric: str = 'cosine',
        threshold: float = 0.25,
        method: str = 'centroid',
    ):
        """
        Load malaya-speech AgglomerativeClustering, originallly from pyannote, https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/clustering.py

        Parameters
        ----------
        min_clusters: int
            minimum cluster size, must bigger than 0
        max_clusters: int
            maximum cluster size, must equal or bigger than `min_clusters`.
            if equal to `min_clusters`, will directly fit into HMM without calculating the best cluster size.
        metric: str, optional (default='cosine')
            Only support `cosine` and `euclidean`.
        threshold: float, optional (default=0.35)
            minimum threshold to assume current iteration of cluster is the best fit.
        method: str, optional (default='centroid')
            All available methods at https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        """

        if min_clusters <= 0:
            raise ValueError('`min_clusters` must bigger than 0')

        if min_clusters > max_clusters:
            raise ValueError('`min_clusters` cannot bigger than `max_clusters`')

        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.metric = metric
        self.threshold = threshold
        self.method = method

    def fit_predict(self, X):
        """
        Fit predict.

        Parameters
        ----------
        X: np.array
            inputs with size of [batch_size, embedding size]

        Returns
        -------
        result: np.array
        """

        num_embeddings, _ = X.shape
        if num_embeddings == 1:
            return np.zeros((1,), dtype=np.int64)

        if self.metric == 'cosine' and self.method in ['centroid', 'median', 'ward']:
            with np.errstate(divide='ignore', invalid='ignore'):
                embeddings = X / np.linalg.norm(X, axis=-1, keepdims=True)
            dendrogram = linkage(
                embeddings, method=self.method, metric='euclidean'
            )
        else:
            dendrogram = linkage(
                X, method=self.method, metric=self.metric
            )

        if self.min_clusters == self.max_clusters:
            threshold = (
                dendrogram[-self.min_clusters, 2]
                if self.min_clusters < num_embeddings
                else -np.inf
            )

        else:
            max_threshold = (
                dendrogram[-self.min_clusters, 2]
                if self.min_clusters < num_embeddings
                else -np.inf
            )
            min_threshold = (
                dendrogram[-self.max_clusters, 2]
                if self.max_clusters < num_embeddings
                else -np.inf
            )
            threshold = min(max(self.threshold, min_threshold), max_threshold)

        return fcluster(dendrogram, threshold, criterion='distance') - 1


class HiddenMarkovModelClustering:
    def __init__(
        self,
        min_clusters: int,
        max_clusters: int,
        metric: str = 'cosine',
        covariance_type: str = 'diag',
        threshold: float = 0.25,
        single_cluster_detection_quantile: float = 0.05,
        single_cluster_detection_threshold: float = 1.15,
    ):
        """
        Load malaya-speech HiddenMarkovModel, originallly from pyannote, https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/clustering.py

        Parameters
        ----------
        min_clusters: int
            minimum cluster size, must bigger than 0
        max_clusters: int
            maximum cluster size, must equal or bigger than `min_clusters`.
            if equal to `min_clusters`, will directly fit into HMM without calculating the best cluster size.
        metric: str, optional (default='cosine')
            Only support `cosine` and `euclidean`.
        covariance_type: str, optional (default='diag')
            Acceptable input shape, https://hmmlearn.readthedocs.io/en/latest/api.html#gaussianhmm
        threshold: float, optional (default=0.35)
            minimum threshold to assume current iteration of cluster is the best fit.
        """

        try:
            from hmmlearn.hmm import GaussianHMM
        except BaseException:
            raise ModuleNotFoundError(
                'hmmlearn not installed. Please install it using `pip3 install hmmlearn` and try again.')

        if min_clusters <= 0:
            raise ValueError('`min_clusters` must bigger than 0')

        if min_clusters > max_clusters:
            raise ValueError('`min_clusters` cannot bigger than `max_clusters`')

        if metric not in ['euclidean', 'cosine']:
            raise ValueError("`metric` must be one of {'cosine', 'euclidean'}")

        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.metric = metric
        self.covariance_type = covariance_type
        self.threshold = threshold
        self.single_cluster_detection_quantile = single_cluster_detection_quantile
        self.single_cluster_detection_threshold = single_cluster_detection_threshold
        self._GaussianHMM = GaussianHMM

    def fit_hmm(self, n_components, train_embeddings):

        hmm = self._GaussianHMM(
            n_components=n_components,
            covariance_type=self.covariance_type,
            n_iter=100,
            random_state=42,
            implementation='log',
            verbose=False,
        )
        hmm.fit(train_embeddings)

        return hmm

    def fit_predict(self, X):
        """
        Fit predict.

        Parameters
        ----------
        X: np.array
            inputs with size of [batch_size, embedding size]

        Returns
        -------
        result: np.array
        """

        if len(X) <= self.max_clusters:
            raise ValueError('sample size must bigger than `max_cluster`')

        num_embeddings = len(X)

        if self.metric == 'cosine':
            with np.errstate(divide='ignore', invalid='ignore'):
                euclidean_embeddings = X / np.linalg.norm(
                    X, axis=-1, keepdims=True
                )
        elif self.metric == 'euclidean':
            euclidean_embeddings = X

        if self.min_clusters == self.max_clusters:
            hmm = self.fit_hmm(self.min_clusters, euclidean_embeddings)
            train_clusters = hmm.predict(euclidean_embeddings)

            return train_clusters

        min_clusters = self.min_clusters
        max_clusters = self.max_clusters

        if min_clusters == 1:
            if (
                np.quantile(
                    pdist(euclidean_embeddings, metric='euclidean'),
                    1.0 - self.single_cluster_detection_quantile,
                )
                < self.single_cluster_detection_threshold
            ):

                return np.zeros((num_embeddings,), dtype=np.int64)

            min_clusters = max(2, min_clusters)
            max_clusters = max(2, max_clusters)

        history = [-np.inf]
        patience = min(3, max_clusters - min_clusters)
        for n_components in range(min_clusters, max_clusters + 1):
            hmm = self.fit_hmm(n_components, euclidean_embeddings)
            try:
                train_clusters = hmm.predict(euclidean_embeddings)
            except ValueError:  # ValueError: startprob_ must sum to 1 (got nan)
                # stop adding states as there too many and not enough
                # training data to train it in a reliable manner.
                break
            centroids = np.vstack(
                [
                    np.mean(X[train_clusters == k], axis=0)
                    for k in range(n_components)
                ]
            )
            centroids_pdist = pdist(centroids, metric=self.metric)
            current_criterion = np.min(centroids_pdist)

            increasing = current_criterion > max(history)
            big_enough = current_criterion > self.threshold

            if increasing or big_enough:
                num_clusters = n_components

            elif n_components == num_clusters + patience:
                break

            history.append(current_criterion)

        hmm = self.fit_hmm(num_clusters, euclidean_embeddings)
        try:
            train_clusters = hmm.predict(euclidean_embeddings)
        except ValueError:
            # ValueError: startprob_ must sum to 1 (got nan)
            train_clusters = np.zeros((num_embeddings,), dtype=np.int64)

        return train_clusters


class StreamingKMeansMaxCluster:
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


class StreamingKMeans:
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


class StreamingSpeakerSimilarity:
    def __init__(self, similarity_threshold=0.8, agg_function: Callable = np.mean):
        """
        Parameters
        ----------
        similarity_threshold: float, optional (default=0.8)
            if current voice activity sample similar at least 0.8, we assumed it is from the same speaker.
        """
        self.similarity_threshold = similarity_threshold
        self.agg_function = agg_function
        self.speakers = {}

    def fit(self, data):
        for sample in data:
            self.streaming(sample)

    def streaming(self, sample):
        embedding = list(self.speakers.values())

        if len(self.speakers):
            a = np.array(embedding)
            s = ((cosine_similarity([sample], a) + 1) / 2)[0]
            where = np.where(s >= self.similarity_threshold)[0]
            if len(where):
                argsort = (np.argsort(s)[::-1]).tolist()
                argsort = [a for a in argsort if a in where]
                speaker = argsort[0]
                self.speakers[speaker] = self.agg_function([sample, self.speakers[speaker]], axis=0)
            else:
                speaker = len(embedding)
                self.speakers[speaker] = sample

        else:
            speaker = len(embedding)
            self.speakers[speaker] = sample

        return speaker
