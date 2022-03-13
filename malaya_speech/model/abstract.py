from malaya_speech.utils.execute import execute_graph


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
