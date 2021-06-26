import tensorflow as tf
import joblib
import numpy as np


class ApplyKmeans_TF(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = tf.convert_to_tensor(self.C_np)
        self.Cnorm = tf.convert_to_tensor(self.Cnorm_np)

    def __call__(self, x):
        if isinstance(x, tf.Tensor):
            dist = tf.reduce_sum(tf.math.pow(x, 2), axis=1, keep_dims=True) - 2 * \
                tf.matmul(x, self.C) + self.Cnorm
            return tf.argmin(dist, axis=1)
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def get_km_model(
    n_clusters=100,
    init='k-means++',
    max_iter=100,
    batch_size=10000,
    tol=0.0,
    max_no_improvement=100,
    n_init=20,
    reassignment_ratio=0.0,
):
    from sklearn.cluster import MiniBatchKMeans
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )
