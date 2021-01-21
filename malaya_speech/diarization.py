from malaya_speech.model.clustering import ClusteringAP
from scipy.spatial.distance import cdist
from malaya_speech.utils.dist import l2_normalize, compute_log_dist_matrix
import numpy as np
from herpetologist import check_type
from typing import Callable


@check_type
def speaker_similarity(
    vad_results,
    speaker_vector,
    similarity_threshold: float = 0.8,
    norm_function: Callable = None,
    return_embedding: bool = False,
):
    """
    Speaker diarization using L2-Norm similarity.

    Parameters
    ----------
    vad_results: List[Tuple[Frame, label]]
        results from VAD.
    speaker_vector: callable
        speaker vector object.
    speaker_change_results: List[Tuple[Frame, float]], optional (default=None)
        results from speaker change module, must in float result.
    similarity_threshold: float, optional (default=0.8)
        if current voice activity sample similar at least 80%, we assumed it is from the same speaker.
    norm_function: Callable, optional(default=None)
        normalize function for speaker vectors.
    speaker_change_threshold: float, optional (default=0.5)
        in one voice activity sample can be more than one speaker, split it using this threshold.

    Returns
    -------
    result : List[Tuple[Frame, label]]
    """
    if not 0 < similarity_threshold <= 1.0:
        raise ValueError(
            'similarity_threshold must, 0 < similarity_threshold <= 1.0'
        )

    speakers, embedding = [], []
    for result in vad_results:
        if result[1]:
            vector = speaker_vector([result[0]])[0]
            if len(embedding):
                a = np.array(embedding)
                if norm_function:
                    a = norm_function(a)
                s = 1 - cdist([vector], a, metric = 'cosine')[0]
                where = np.where(s >= similarity_threshold)[0]
                if len(where):
                    argsort = (np.argsort(s)[::-1]).tolist()
                    argsort = [a for a in argsort if a in where]
                    speakers.append(f'speaker {argsort[0]}')
                else:
                    speakers.append(f'speaker {len(embedding)}')
                    embedding.append(vector)

            else:
                speakers.append(f'speaker {len(embedding)}')
                embedding.append(vector)
        else:
            speakers.append('not a speaker')

    results = []
    for no, result in enumerate(vad_results):
        results.append((result[0], speakers[no]))

    if return_embedding:
        return results, embedding
    else:
        return results


@check_type
def affinity_propagation(
    vad_results,
    speaker_vector,
    norm_function: Callable = l2_normalize,
    log_distance_metric: str = 'cosine',
    damping: float = 0.8,
    preference: float = None,
    return_embedding = False,
):
    """
    Speaker diarization using sklearn Affinity Propagation.

    Parameters
    ----------
    vad_results: List[Tuple[Frame, label]]
        results from VAD.
    speaker_vector: callable
        speaker vector object.
    norm_function: Callable, optional(default=malaya_speech.utils.dist.l2_normalize)
        normalize function for speaker vectors.
    log_distance_metric: str, optional (default='cosine')
        post distance norm in log scale metrics.

    Returns
    -------
    result : List[Tuple[Frame, label]]
    """

    affinity = ClusteringAP(
        metric = log_distance_metric, damping = damping, preference = preference
    )
    speakers, activities, mapping = [], [], {}
    for no, result in enumerate(vad_results):
        if result[1]:
            speakers.append('got')
            mapping[len(activities)] = no
            vector = speaker_vector([result[0]])[0]
            activities.append(vector)
        else:
            speakers.append('not a speaker')

    activities = np.array(activities)
    if norm_function:
        activities = norm_function(activities)

    cluster_labels = affinity.apply(activities)

    for k, v in mapping.items():
        speakers[v] = f'speaker {cluster_labels[k]}'

    results = []
    for no, result in enumerate(vad_results):
        results.append((result[0], speakers[no]))

    if return_embedding:
        return results, activities
    else:
        return results


@check_type
def spectral_cluster(
    vad_results,
    speaker_vector,
    min_clusters: int = None,
    max_clusters: int = None,
    p_percentile: float = 0.95,
    gaussian_blur_sigma = 1.0,
    norm_function: Callable = l2_normalize,
    log_distance_metric: str = None,
    return_embedding = False,
    **kwargs,
):
    """
    Speaker diarization using SpectralCluster, https://github.com/wq2012/SpectralCluster

    Parameters
    ----------
    vad_results: List[Tuple[Frame, label]]
        results from VAD.
    speaker_vector: callable
        speaker vector object.
    min_clusters: int, optional (default=None)
        minimal number of clusters allowed (only effective if not None).
    max_clusters: int, optional (default=None)
        maximal number of clusters allowed (only effective if not None).
        can be used together with min_clusters to fix the number of clusters.
    gaussian_blur_sigma: float, optional (default=1.0)
        sigma value of the Gaussian blur operation.
    p_percentile: float, optional (default=0.95)
        the p-percentile for the row wise thresholding.
    norm_function: Callable, optional(default=malaya_speech.utils.dist.l2_normalize)
        normalize function for speaker vectors.
    log_distance_metric: str, optional (default=None)
        post distance norm in log scale metrics.

    Returns
    -------
    result : List[Tuple[Frame, label]]
    """
    try:
        from spectralcluster import SpectralClusterer

    except:
        raise ModuleNotFoundError(
            'spectralcluster not installed. Please install it by `pip install spectralcluster` and try again.'
        )

    clusterer = SpectralClusterer(
        min_clusters = min_clusters,
        max_clusters = max_clusters,
        p_percentile = p_percentile,
        gaussian_blur_sigma = gaussian_blur_sigma,
        **kwargs,
    )

    speakers, activities, mapping = [], [], {}
    for no, result in enumerate(vad_results):
        if result[1]:
            speakers.append('got')
            mapping[len(activities)] = no
            vector = speaker_vector([result[0]])[0]
            activities.append(vector)
        else:
            speakers.append('not a speaker')

    activities = np.array(activities)
    if norm_function:
        activities = norm_function(activities)

    if log_distance_metric:
        activities = compute_log_dist_matrix(activities, log_distance_metric)

    cluster_labels = clusterer.predict(activities)

    for k, v in mapping.items():
        speakers[v] = f'speaker {cluster_labels[k]}'

    results = []
    for no, result in enumerate(vad_results):
        results.append((result[0], speakers[no]))

    if return_embedding:
        return results, activities
    else:
        return results
