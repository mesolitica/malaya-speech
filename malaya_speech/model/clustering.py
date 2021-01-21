from malaya_speech.utils.dist import compute_log_dist_matrix


class ClusteringAP:
    def __init__(self, damping = 0.8, preference = None, metric = 'angular'):

        self.damping = damping
        self.preference = preference
        self.metric = metric

    def apply(self, fX):
        try:
            from sklearn.cluster import AffinityPropagation
        except:
            raise ModuleNotFoundError(
                'sklearn not installed. Please install it by `pip install sklearn` and try again.'
            )

        clusterer = AffinityPropagation(
            damping = self.damping,
            max_iter = 100,
            convergence_iter = 15,
            preference = self.preference,
            affinity = 'precomputed',
        )
        distance_matrix = compute_log_dist_matrix(fX, metric = self.metric)
        cluster_labels = clusterer.fit_predict(distance_matrix)

        return cluster_labels
