import logging
import os
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import joblib
logging.basicConfig(level=logging.DEBUG)

# logging.basicConfig(
#     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     level=os.environ.get("LOGLEVEL", "INFO").upper(),
#     stream=sys.stdout,
# )


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
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


n_clusters = 100
init = 'k-means++'
max_iter = 1
batch_size = 10000
tol = 0.0
max_no_improvement = 5
n_init = 20
reassignment_ratio = 0.0
seed = 42

np.random.seed(seed)

feat = np.load('/home/husein/ssd1/mfcc/train_0_1.npy', mmap_mode="r")
leng_path = '/home/husein/ssd1/mfcc/train_0_1.len'
with open(leng_path, "r") as f:
    lengs = [int(line.rstrip()) for line in f]
lengs = sum(lengs[:int(len(lengs) * 0.1)])
print(lengs)
print(feat.shape)
feat = feat[:lengs]
km_model = get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
)
print(feat.shape)
km_model.fit(feat)
joblib.dump(km_model, 'train.km')
