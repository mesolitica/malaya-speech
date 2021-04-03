from scipy.stats.mstats import gmean, hmean, hdmedian
import numpy as np
from typing import List, Callable


class Stack:
    def __str__(self):
        return f'<{self.__name__}: {self.__model__}>'

    def __init__(self, models):
        self._models = models
        self.__name__ = self._models[0].__name__
        self.__model__ = self._models[0].__model__

    def predict_proba(self, inputs, aggregate: Callable = gmean):
        """
        Stacking for predictive models, will return probability.

        Parameters
        ----------
        inputs: List[np.array]
        aggregate : Callable, optional (default=scipy.stats.mstats.gmean)
        Aggregate function.

        Returns
        -------
        result: np.array
        """
        results = []
        for i in range(len(self._models)):
            results.append(self._models[i].predict_proba(inputs))

        mode = aggregate
        results = mode(np.array(results), axis = 0)
        return results

    def predict(self, inputs, aggregate: Callable = gmean):
        """
        Stacking for predictive models, will return labels.

        Parameters
        ----------
        inputs: List[np.array]
        aggregate : Callable, optional (default=scipy.stats.mstats.gmean)
        Aggregate function.

        Returns
        -------
        result: List[str]
        """

        probs = np.argmax(
            self.predict_proba(inputs, aggregate = aggregate), axis = 1
        )
        return [self._models[0].labels[p] for p in probs]

    def __call__(self, input):
        return self.predict([input])[0]


def classification_stack(models):
    """
    Stacking for classification models. All models should be in the same domain classification.

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
