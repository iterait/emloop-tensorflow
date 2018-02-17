from typing import List

import tensorflow as tf


class GraphTower:
    """
    ``GraphTower`` is a lightweight wrapper around a tower (TF sub-graph) in multi-GPU models.
    It allows to work with multiple copies of the same sub-graph distributed on multiple devices
    with only one set of input and output names.

    **Usage**

    .. code-block:: python
        :caption: **1** create the desired number of GraphTowers

        towers = [GraphTower(i, inputs, outputs) for i in range(4)]

    .. code-block:: python
        :caption: **2** create the TF sub-graphs in the tower environments (uses with ``tf.device(...)``)

        for tower in towers:
            with tower:
                # define the TF graph with the respective inputs and outputs

    .. code-block:: python
        :caption: **3** find the input placeholders and output variables

        for tower in towers:
            tower.find_io_tensors()


    .. code-block:: python
        :caption: **4** access the io tensors, loss etc

        towers[3]['my_input']  # my_input placeholder which is actually named 'my_input_3:0'

    .. warning:
        The sub-graphs must be defined in the order corresponding to the tower ids!

    """

    def __init__(self, id_: int, inputs: List[str], outputs: List[str], loss_name):
        """
        Create new GraphTower.

        :param id_: tower (gpu) id, towers with negative ids are placed on ``/cpu:0``
        :param inputs: tower input names
        :param outputs: tower output names
        :param loss_name: loss tensor name
        """
        self._id = id_
        self._device_name = '/cpu:0' if id_ < 0 else '/gpu:{}'.format(id_)
        self._input_names = inputs
        self._output_names = outputs
        self._inputs = {}
        self._outputs = {}
        self._loss_name = loss_name

    def _get_full_name(self, tensor_name: str) -> str:
        """
        Translate the given simple tensor name to the actual tensor name in the tower graph.

        E.g.:
        variable named ``loss`` in the 0th tower will be named ``loss:0``
        variable name ``predictions`` in the 1st tower will be name ``predictions_1:0``
        """
        return tensor_name + ('' if self._id < 1 else '_{}'.format(self._id)) + ':0'

    def _find_or_raise(self, tensor_name: str) -> tf.Tensor:
        """
        Find the tensor with the given name in the default graph or raise an exception.

        :param tensor_name: tensor name to be find
        :return: tensor with the given name
        :raise ValueError: if the tensor with the given name could not be found
        """
        full_name = self._get_full_name(tensor_name)
        try:
            return tf.get_default_graph().get_tensor_by_name(full_name)
        except (KeyError, ValueError, TypeError) as ex:
            raise ValueError('Tensor `{}` with full name `{}` was not found.'.format(tensor_name, full_name)) from ex

    def find_io_tensors(self) -> None:
        """Find the tower's input and output tensors in the default graph."""
        for input_name in self._input_names:
            self._inputs[input_name] = self._find_or_raise(input_name)
        for output_name in self._output_names:
            self._outputs[output_name] = self._find_or_raise(output_name)

    @property
    def loss(self) -> tf.Tensor:
        """Return the loss tensor."""
        return self[self._loss_name]

    @property
    def input_names(self) -> List[str]:
        """Return list of the input names."""
        return self._input_names

    @property
    def output_names(self) -> List[str]:
        """Return list of the output names."""
        return self._output_names

    def __getitem__(self, item) -> tf.Tensor:
        """
        Return input/output tensor with the given name.

        :param item: tensor name
        :return: tensor with the given name
        :raise KeyError: If the given tensor name is not listed as input/output
        """
        if item in self._outputs:
            return self._outputs[item]
        elif item in self._inputs:
            return self._inputs[item]
        else:
            raise KeyError('Tensor `{}` is not within the input/output tensors'.format(item))

    def __enter__(self) -> None:
        """Enter ``with tf.device(...):`` env."""
        self._device = tf.device(self._device_name)
        self._device.__enter__()

    def __exit__(self, *args) -> None:
        """Exit ``with tf.device(...):`` env."""
        self._device.__exit__(*args)
