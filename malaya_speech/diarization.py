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
                s = 1 - cdist([vector], a, metric='cosine')[0]
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


def _group_vad(vad_results, speaker_vector, norm_function=None, log_distance_metric='cosine'):
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
    if norm_function is not None:
        activities = norm_function(activities)

    if log_distance_metric is not None:
        activities = compute_log_dist_matrix(activities, log_distance_metric)

    return speakers, activities, mapping


@check_type
def clustering(
    vad_results,
    speaker_vector,
    model,
    norm_function=l2_normalize,
    log_distance_metric: str = None,
    return_embedding: bool = False,
):
    """
    Speaker diarization using any clustering model.

    Parameters
    ----------
    vad_results: List[Tuple[Frame, label]]
        results from VAD.
    speaker_vector: callable
        speaker vector object.
    model: callable
        Any unsupervised clustering model.
        Required `fit_predict` or `apply` or `predict` method.
    norm_function: Callable, optional(default=malaya_speech.utils.dist.l2_normalize)
        normalize function for speaker vectors.
    log_distance_metric: str, optional (default=None)
        post distance norm in log scale metrics.
        this parameter is necessary for model that required square array input.
        Common value is one of ['cosine', 'angular'].

    Returns
    -------
    result : List[Tuple[Frame, label]]
    """

    if not hasattr(model, 'fit_predict') and not hasattr(model, 'apply') and not hasattr(model, 'predict'):
        raise ValueError('model must have `fit_predict` or `apply` or `predict` method.')

    speakers, activities, mapping = _group_vad(
        vad_results,
        speaker_vector=speaker_vector,
        norm_function=norm_function,
        log_distance_metric=log_distance_metric
    )

    if hasattr(model, 'fit_predict'):
        cluster_labels = model.fit_predict(activities)
    elif hasattr(model, 'predict'):
        cluster_labels = model.predict(activities)
    elif hasattr(model, 'apply'):
        cluster_labels = model.apply(activities)

    for k, v in mapping.items():
        speakers[v] = f'speaker {cluster_labels[k]}'

    results = []
    for no, result in enumerate(vad_results):
        results.append((result[0], speakers[no]))

    if return_embedding:
        return results, activities
    else:
        return results
