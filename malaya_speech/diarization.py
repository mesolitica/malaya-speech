from sklearn.metrics.pairwise import cosine_similarity
from malaya_speech.utils.dist import l2_normalize, compute_log_dist_matrix
import numpy as np
from herpetologist import check_type
from typing import Callable
import copy
import logging

logger = logging.getLogger(__name__)


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


def streaming_speaker_similarity(
    vector,
    speakers: dict,
    similarity_threshold: float = 0.8,
    agg_function: Callable = np.mean,
):
    """
    Speaker diarization using L2-Norm similarity streaming mode.

    Parameters
    ----------
    vector: np.array
        np.array or malaya_speech.model.frame.Frame.
    speakers: dict
        empty dictionary, it will update overtime using pass by reference.
    similarity_threshold: float, optional (default=0.8)
        if current voice activity sample similar at least 0.8, we assumed it is from the same speaker.

    Returns
    -------
    result : str
    """

    embedding = list(speakers.values())

    if len(speakers):
        a = np.array(embedding)
        s = ((cosine_similarity([vector], a) + 1) / 2)[0]
        where = np.where(s >= similarity_threshold)[0]
        if len(where):
            argsort = (np.argsort(s)[::-1]).tolist()
            argsort = [a for a in argsort if a in where]
            speaker = f'speaker {argsort[0]}'
            speakers[speaker] = agg_function([vector, speakers[speaker]], axis=0)
        else:
            speaker = f'speaker {len(embedding)}'
            speakers[speaker] = vector

    else:
        speaker = f'speaker {len(embedding)}'
        speakers[speaker] = vector

    return speaker


def speaker_similarity(
    vad_results,
    speaker_vector,
    similarity_threshold: float = 0.8,
    agg_function: Callable = np.mean,
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
        if current voice activity sample similar at least 0.8, we assumed it is from the same speaker.

    Returns
    -------
    result : List[Tuple[Frame, label]]
    """

    speakers, embedding = {}, []
    result_speakers = []
    for result in vad_results:
        if result[1]:
            vector = speaker_vector([result[0]])[0]
            speaker = streaming_speaker_similarity(
                vector=vector,
                speakers=speakers,
                similarity_threshold=similarity_threshold,
                agg_function=agg_function,
            )
            embedding.append(vector)
        else:
            speaker = 'not a speaker'

        result_speakers.append(speaker)

    results = []
    for no, result in enumerate(vad_results):
        results.append((result[0], result_speakers[no]))

    if return_embedding:
        return results, embedding
    else:
        return results


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

    if not hasattr(
            model,
            'fit_predict') and not hasattr(
            model,
            'apply') and not hasattr(
                model,
            'predict'):
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


def combine(
    list_results,
    speaker_vector,
    similarity_threshold: float = 0.8,
    agg_function: Callable = np.mean,
    sortby_pagerank: bool = True,
):
    """
    Combined multiple diarization results into single diarization results using PageRank.
    Required malaya and networkx libraries.

    Parameters
    ----------
    vad_results: List[List[Tuple[Frame, label]]]
        results from multiple diarization results.
    speaker_vector: callable
        speaker vector object.
    similarity_threshold: float, optional (default=0.8)
        if current voice activity sample similar at least 0.8, we assumed it is from the existing speakers.
    agg_function: Callable, optional (default=np.mean)
        aggregate function to aggregate when we have multiple samples for the same speaker.
    sortby_pagerank: bool, optional (default=True)
        sort speaker names using pagerank score.
        This required malaya to be installed.

    Returns
    -------
    result : List[Tuple[Frame, label]]
    """

    try:
        import networkx as nx
    except BaseException:
        raise ModuleNotFoundError(
            'networkx not installed. Please install it by `pip install networkx` and try again.'
        )

    speakers = {}
    last_timestamps = []
    for no, results in enumerate(list_results):
        if no > 0:
            last_timestamp = results[-1][0].timestamp + results[-1][0].duration
        else:
            last_timestamp = 0
        last_timestamps.append(last_timestamp)
        for result in results:
            speaker = result[1]

            if speaker == 'not a speaker':
                continue

            vector = speaker_vector([result[0]])[0]
            speaker_name = f'{no}-{speaker}'

            if speaker_name not in speakers:
                speakers[speaker_name] = vector
            else:
                speakers[speaker_name] = agg_function([vector, speakers[speaker_name]], axis=0)

    embeddings = list(speakers.values())
    list_speakers = list(speakers.keys())
    similar = (cosine_similarity(embeddings) + 1) / 2
    similar[np.diag_indices(len(similar))] = 0.0

    if sortby_pagerank:
        try:
            from malaya.graph.pagerank import pagerank
        except BaseException:
            raise ModuleNotFoundError(
                'malaya not installed. Please install it by `pip install malaya` and try again.'
            )

        scores = pagerank(similar)
        ranked = sorted(
            [
                (scores[i], s, i)
                for i, s in enumerate(list_speakers)
            ],
            reverse=False,
        )
        sorted_speakers = [r[1] for r in ranked]
    else:
        sorted_speakers = sorted(list_speakers)

    G = nx.DiGraph()
    G.add_nodes_from(list_speakers)

    for speaker in sorted_speakers:
        embeddings = list(speakers.values())
        list_speakers = list(speakers.keys())
        similar = (cosine_similarity(embeddings) + 1) / 2
        similar[np.diag_indices(len(similar))] = 0.0
        s = similar[list_speakers.index(speaker)]

        where = np.where(s >= similarity_threshold)[0]
        if len(where):
            logger.debug(f'speaker: {speaker}, where: {where}')
            argsort = (np.argsort(s)[::-1]).tolist()
            argsort = [a for a in argsort if a in where]
            speakers[list_speakers[argsort[0]]] = np.mean(
                [speakers[speaker], speakers[list_speakers[argsort[0]]]], axis=0)
            speakers.pop(speaker, None)

            G.add_edge(speaker, list_speakers[argsort[0]])

    new_list_results = []
    for no, results in enumerate(list_results):
        for result in results:
            speaker = result[1]
            speaker = f'{no}-{speaker}'
            frame = copy.deepcopy(result[0])
            frame.timestamp += last_timestamps[no]

            if 'not a speaker' not in speaker:
                traversed = list(nx.dfs_edges(G, source=speaker))
                if len(traversed):
                    new_label = traversed[-1][-1]
                else:
                    new_label = speaker
            else:
                new_label = 'not a speaker'

            new_list_results.append((frame, new_label))

    return new_list_results
