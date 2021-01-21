from scipy.stats.mstats import gmean, hmean, hdmedian
import numpy as np

dict_function = {
    'gmean': gmean,
    'hmean': hmean,
    'mean': np.mean,
    'min': np.amin,
    'max': np.amax,
    'median': hdmedian,
}

_aggregate_availability = {
    'gmean': {'Description': 'geometrical mean'},
    'hmean': {'Description': 'harmonic mean'},
    'mean': {'Description': 'mean'},
    'min': {'Description': 'minimum'},
    'max': {'Description': 'maximum'},
    'median': {'Description': 'Harrell-Davis median'},
}


def available_aggregate_function():
    from malaya.function import describe_availability

    return describe_availability(_aggregate_availability)


class Stack:
    def __init__(self, models):
        self._models = models

    def predict_proba(self, inputs, aggregate: str = 'gmean'):
        """
        Stacking for predictive models, will return probability.

        Parameters
        ----------
        inputs: List[np.array]
        aggregate : str, optional (default='gmean')
            Aggregate function supported. Allowed values:

            * ``'gmean'`` - geometrical mean.
            * ``'hmean'`` - harmonic mean.
            * ``'mean'`` - mean.
            * ``'min'`` - min.
            * ``'max'`` - max.
            * ``'median'`` - Harrell-Davis median.

        Returns
        -------
        result: np.array
        """
        aggregate = aggregate.lower()

        if aggregate not in dict_function:
            raise ValueError(
                'aggregate is not supported, please check supported functions from `malaya_speech.stack.available_aggregate_function()`.'
            )
        results = []
        for i in range(len(self._models)):
            results.append(self._models.predict_proba(inputs))

        mode = dict_function[aggregate]
        results = mode(np.array(results), axis = 0)
        return results

    def predict(self, inputs, aggregate: str = 'gmean'):
        """
        Stacking for predictive models, will return labels.

        Parameters
        ----------
        inputs: List[np.array]
        aggregate : str, optional (default='gmean')
            Aggregate function supported. Allowed values:

            * ``'gmean'`` - geometrical mean.
            * ``'hmean'`` - harmonic mean.
            * ``'mean'`` - mean.
            * ``'min'`` - min.
            * ``'max'`` - max.
            * ``'median'`` - Harrell-Davis median.

        Returns
        -------
        result: List[str]
        """
        aggregate = aggregate.lower()
        if not isinstance(models, list):
            raise ValueError('models must be a list')

        if aggregate not in dict_function:
            raise ValueError(
                'aggregate is not supported, please check supported functions from `malaya_speech.stack.available_aggregate_function()`.'
            )
        probs = np.argmax(self.predict_proba(inputs), axis = 1)
        return [self._models[0].labels[p] for p in probs]

    def __call__(self, input):
        return self.predict([input])[0]


def predict_stack(models):
    """
    Stacking for predictive models.

    Parameters
    ----------
    models: List[Callable]
        list of models.

    Returns
    -------
    result: malaya_speech.stack.Stack class
    """

    labels = None
    for i in range(len(models)):
        if not 'predict_proba' in dir(models[i]):
            raise ValueError('all models must able to `predict_proba`')
        if labels is None:
            labels = models[i].labels
        else:
            if labels != models[i].labels:
                raise ValueError('domain classification must be same!')

    return Stack(models)
