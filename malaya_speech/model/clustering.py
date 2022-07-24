# The MIT License (MIT)
#
# Copyright (c) 2021- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import numpy as np


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
