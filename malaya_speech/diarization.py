from malaya_speech.model.clustering import CLUSTERING_AP
from scipy.spatial.distance import cdist
from malaya_speech.utils.dist import l2_normalize
import numpy as np
from herpetologist import check_type


@check_type
def speaker_similarity(
    vad_results,
    speaker_vector,
    similarity_threshold: float = 0.8,
    return_embedding: bool = False,
):
    """
    Speaker diarization using L2-Norm similarity.

    Parameters
    ----------
    vad_results: List[Tuple[FRAME, label]]
        results from VAD.
    speaker_vector: callable
        speaker vector object.
    speaker_change_results: List[Tuple[FRAME, float]], optional (default=None)
        results from speaker change module, must in float result.
    similarity_threshold: float, optional (default=0.8)
        if current voice activity sample similar at least 80%, we assumed it is from the same speaker.
    speaker_change_threshold: float, optional (default=0.5)
        in one voice activity sample can be more than one speaker, split it using this threshold.

    Returns
    -------
    result : List[Tuple[FRAME, label]]
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
                s = (
                    1
                    - cdist([vector], np.array(embedding), metric = 'cosine')[0]
                )
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
    metric: str = 'cosine',
    damping: float = 0.8,
    preference: float = None,
    return_embedding = False,
):
    """
    Speaker diarization using sklearn Affinity Propagation.

    Parameters
    ----------
    vad_results: List[Tuple[FRAME, label]]
        results from VAD.
    speaker_vector: callable
        speaker vector object.

    Returns
    -------
    result : List[Tuple[FRAME, label]]
    """

    affinity = CLUSTERING_AP(
        metric = metric, damping = damping, preference = preference
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
    normed = l2_normalize(np.array(activities))
    cluster_labels = affinity.apply(normed)

    for k, v in mapping.items():
        speakers[v] = f'speaker {cluster_labels[k]}'

    results = []
    for no, result in enumerate(vad_results):
        results.append((result[0], speakers[no]))

    if return_embedding:
        return results, activities
    else:
        return results
