from malaya_speech.utils.execute import execute_graph
from malaya_speech.utils.astype import float_to_int
from typing import Callable


class Abstract:
    def __str__(self):
        return f'<{self.__name__}: {self.__model__}>'

    def _execute(self, inputs, input_labels, output_labels):
        return execute_graph(
            inputs=inputs,
            input_labels=input_labels,
            output_labels=output_labels,
            sess=self._sess,
            input_nodes=self._input_nodes,
            output_nodes=self._output_nodes,
        )

    def _torch_method(self, method, **kwargs):

        if hasattr(self, 'model'):
            if hasattr(self.model, method):
                try:
                    return getattr(self.model, method)(**kwargs)
                except:
                    raise ValueError('this model is not a PyTorch model.')

        if hasattr(self, 'hf_model'):
            if hasattr(self.hf_model, method):
                try:
                    return getattr(self.hf_model, method)(**kwargs)
                except:
                    raise ValueError('this model is not a PyTorch model.')

    def cuda(self, **kwargs):
        return self._torch_method('cuda')

    def eval(self, **kwargs):
        return self._torch_method('eval')


class TTS:
    def __init__(self, e2e=False):
        self.e2e = e2e

    def gradio(self, vocoder: Callable = None, **kwargs):
        """
        Text-to-Speech on Gradio interface.

        Parameters
        ----------
        vocoder: Callable, optional (default=None)
            vocoder object that has `predict` method, prefer from malaya_speech itself.
            Not required if using End-to-End TTS model such as VITS.

        **kwargs: keyword arguments for `predict` and `iface.launch`.
        """
        if not self.e2e and vocoder is None:
            raise ValueError('TTS model is not End-to-End, required vocoder.')

        try:
            import gradio as gr
        except BaseException:
            raise ModuleNotFoundError(
                'gradio not installed. Please install it by `pip install gradio` and try again.'
            )

        def pred(string):
            r = self.predict(string=string, **kwargs)

            if self.e2e:
                y_ = r['y']
            else:
                if 'universal' in str(vocoder):
                    o = r['universal-output']
                else:
                    o = r['mel-output']
                y_ = vocoder(o)
            data = float_to_int(y_)
            return (22050, data)

        if self.e2e:
            title = 'End-to-End Text-to-Speech'
        else:
            title = 'Text-to-Speech + Neural Vocoder'
        description = 'It will take sometime for the first time, after that, should be really fast.'

        iface = gr.Interface(pred, gr.inputs.Textbox(lines=3, label='Input Text'),
                             'audio', title=title, description=description)
        return iface.launch(**kwargs)
