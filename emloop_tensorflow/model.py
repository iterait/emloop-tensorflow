import math
import logging
from os import path
from abc import ABCMeta
from typing import List, Mapping, Optional, Dict
from glob import glob

import numpy as np
import emloop as el
import tensorflow as tf

from .third_party.tensorflow.freeze_graph import freeze_graph
from .third_party.tensorflow.average_gradients import average_gradients
from .utils import create_optimizer, Profiler
from .graph_tower import GraphTower

DEFAULT_LOSS_NAME = 'loss'
"""Default loss tensor name."""


class BaseModel(el.AbstractModel, metaclass=ABCMeta):  # pylint: disable=too-many-instance-attributes
    """
    Emloop :py:class:`AbstractModel <emloop.models.AbstractModel>` implementation for TensorFlow models.

    To define a **emloop** trainable model in TensorFlow, derive your class from :py:class:`BaseModel` and override
    :py:meth:`_create_model` method.

    See the method references for additional customization options.
    """

    TRAIN_OP_NAME = 'train_op'
    """Expected train op tensor name prefix."""

    TRAINING_FLAG_NAME = 'el_is_training'
    """Training flag variable name."""

    SIGNAL_MEAN_NAME = 'signal_mean'
    """Name of the monitored signal mean tensor/output."""

    SIGNAL_VAR_NAME = 'signal_variance'
    """Name of the monitored signal variance tensor/output."""

    def __init__(self,  # pylint: disable=too-many-arguments
                 dataset: Optional[el.AbstractDataset], log_dir: Optional[str], inputs: List[str], outputs: List[str],
                 session_config: Optional[dict] = None, n_gpus: int = 0, restore_from: Optional[str] = None,
                 optimizer: dict = None, freeze: bool = False, loss_name: str = DEFAULT_LOSS_NAME,
                 monitor: Optional[str] = None, clip_gradient: Optional[float] = None, profile: bool = False,
                 keep_profiles: int = 5, quantize: bool = False, quantize_model_name: str = 'quantized', quantize_delay: int = 0, *args, **kwargs):
        """
        Create new emloop trainable TensorFlow model.

        TF Graph, train ops etc. are constructed with the following procedure:

        #. Create ``tf.Graph`` and ``tf.Session`` with :py:meth:`_create_session`
        #. Either create or restore the model with :py:meth:`_create_model` or :py:meth:`_restore_model` respectively
        #. Find input/output tensors
        #. Create train ops with :py:meth:`_create_train_ops` unless they are already restored
        #. Find the train ops
        #. Create ``tf.Saver``

        .. note::
            In most cases, it is not required to re-define the ``__init__`` method for your models.

        .. tip::
            It is often useful to monitor signal/weights/gradients ranges, means and/or variances during the training.
            **emloop-tensorflow** base model actually provides monitoring of the feed-forward signal through the net.
            Simply set up the ``monitor`` paramater to the name of the layers to be monitored (e.g. `Conv2D` or `Relu`).
            Layer activation means and variances (named ``signal_mean`` and ``signal_variance``) will be include
            in the output.

        .. warning::
            Quantization aware training is rather experimental. You may run into many different issues among which
            using unsupported ops is the most common one. Apart from specifying the quantize flag,
            you need to register input and output tensors for the quantized graph with `_register_quantize_input` and
            `_register_quantize_output` methods called from `_create_model` method.
            For now, only TF 1.14 or 1.15 is supported. Both QA training graph with
            fake quantization nodes and fully quantized .tfile flat buffer will be saved. By the way, the inputs
            should be in [0, 255] range, otherwise they will be scaled badly. In conclusion,
            user's discretion is advised.

        :param dataset: dataset to be trained with
        :param log_dir: path to the logging directory (wherein models should be saved)
        :param inputs: model input names
        :param outputs: model output names
        :param session: TF session configuration dict, see :py:meth:`_create_session`
        :param n_gpus: number of GPUs to use
        :param restore_from: path to directory from which the model is restored
        :param restore_model_name: model name to be restored (e.g. ``model.ckpt``)
        :param optimizer: TF optimizer configuration dict
        :param freeze: freeze the graph after each save
        :param loss_name: loss tensor name
        :param monitor: monitor signal mean and variance of the tensors which names contain the specified value
        :param clip_gradient: limit the absolute value of the gradient; set to None for no clipping
        :param profile: if true, profile the speed of model inference and save profiles to the specified log_dir
        :param keep_profiles: if true, profile the speed of model inference and save profiles to the specified log_dir
        :param quantize: perform quantization-aware training from tf.contrib.quantize
        :param quantize_model_name: quantize model name
        :param quantize_delay: delay quantization-aware training by the specified number of iterations
        :param args: additional args forwarded to :py:meth:`_create_model`
        :param kwargs: additional kwargs forwarded to :py:meth:`_create_model`
        """
        if quantize and n_gpus > 1:
            raise ValueError('Quantization with n_gpus>1 is not supported at the moment.')
        if quantize and restore_from is not None:
            raise ValueError('Restoring (to be) quantized model is not supported at the moment.')
        if quantize:
            logging.warning('Quantization trainign is experimental atm, please read the warning in the docs first.')
        super().__init__(dataset=dataset, log_dir=log_dir, restore_from=restore_from)
        self._args = args
        self._session_config = session_config
        self._kwargs = kwargs
        self._quantize = quantize
        self._quantize_model_name = quantize_model_name
        self._quantize_delay = quantize_delay
        self._quantize_inputs = []
        self._quantize_outputs = []
        self._extra_outputs = []
        self._dataset = dataset
        self._log_dir = log_dir
        self._freeze_graph = freeze
        self._clip_gradient = clip_gradient
        self._loss_name = loss_name
        self._train_ops = []
        self._graph = self._saver = None
        self._towers = [GraphTower(i, inputs, outputs, loss_name) for i in range(n_gpus)]
        if n_gpus == 0:
            self._towers.append(GraphTower(-1, inputs, outputs, loss_name))
        logging.info('\tCreating TF model on %s GPU devices', n_gpus)
        self._graph = tf.Graph()
        self._session = self._create_session(session_config)

        if profile and not log_dir:
            raise ValueError('log_dir has to be specified with profile set to True')

        self._profile = profile
        if profile:
            self._profiler = Profiler(log_dir, keep_profiles, self._session)

        dependencies = []
        with self._graph.as_default():
            if restore_from is None:
                with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
                    self._is_training = tf.placeholder_with_default(tf.constant(False, tf.bool),
                                                                    shape=[],
                                                                    name=BaseModel.TRAINING_FLAG_NAME)
                    for tower in self._towers:
                        with tower:
                            self._create_model(**kwargs)
                        dependencies.append(list(self._graph.get_collection(tf.GraphKeys.UPDATE_OPS)))
            else:
                self._restore_model(restore_from=restore_from)
                self._is_training = self._graph.get_tensor_by_name(BaseModel.TRAINING_FLAG_NAME + ':0')

            if monitor:
                for protected_var_name in [BaseModel.SIGNAL_MEAN_NAME, BaseModel.SIGNAL_VAR_NAME]:
                    for io, io_name in [(inputs, 'inputs'), (outputs, 'outputs')]:
                        if protected_var_name in io:
                            raise ValueError('Variable `{}` in model {} is reserved when monitoring is turned on.'
                                             .format(protected_var_name, io_name))
                means, variances = [], []
                for op in self.graph.get_operations():
                    if monitor in op.name and 'grad' not in op.name.lower() and len(op.values()) > 0:
                        out_tensor = op.values()[0]
                        if len(out_tensor.get_shape().as_list()) > 1:
                            layer_mean, layer_var = tf.nn.moments(tf.layers.flatten(out_tensor), axes=[1])
                            means.append(layer_mean)
                            variances.append(layer_var)
                if not means:
                    raise ValueError('No ops to be monitored found with `{}` in their name.'.format(monitor))
                signal_mean = tf.reduce_mean(means, axis=0, name=BaseModel.SIGNAL_MEAN_NAME)
                signal_var = tf.reduce_mean(variances, axis=0, name=BaseModel.SIGNAL_VAR_NAME)
                self._extra_outputs += [signal_mean, signal_var]

            for tower in self._towers:
                tower.find_io_tensors()

            if restore_from is None:
                logging.debug('\tCreating train ops')
                self._create_train_ops(dependencies, optimizer)

            logging.debug('\tCreating Saver')
            self._saver = tf.train.Saver(max_to_keep=100000000)

            if restore_from is None:
                logging.debug('\tInitializing the variables')
                self._initialize_variables(**kwargs)

            logging.debug('\tSearching for the train ops in the created graph')
            try:
                for i in range(len(self._towers)):
                    self._train_ops.append(
                        self._graph.get_operation_by_name(BaseModel.TRAIN_OP_NAME+'_{}'.format(i + 1)))
            except (KeyError, ValueError, TypeError) as ex:
                raise ValueError('Cannot find train op {} in graph. '
                                 'The op must be named `train_op_{}`.'.format(i+1, i+1)) from ex

            train_vars = self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            logging.debug('Trainable variables: %s', [var.name for var in train_vars])
            logging.info('Number of parameters: %s', sum([np.prod(var.get_shape().as_list()) for var in train_vars
                                                          if var.shape.is_fully_defined()]))

    @property
    def input_names(self) -> List[str]:  # pylint: disable=invalid-sequence-index
        """List of TF input tensor (placeholder) names."""
        return self._towers[0].input_names

    @property
    def output_names(self) -> List[str]:  # pylint: disable=invalid-sequence-index
        """List of TF output tensor names."""
        return self._towers[0].output_names

    @property
    def is_training(self) -> tf.Tensor:
        """
        Training flag tensor.

        This is useful for determining whether to use certain ops such as dropout.
        """
        return self._is_training

    @property
    def graph(self) -> tf.Graph:
        """TF graph object."""
        return self._graph

    @property
    def session(self) -> tf.Session:
        """TF session object."""
        return self._session

    def run(self, batch: el.Batch, train: bool=False, stream: el.datasets.StreamWrapper=None) -> Mapping[str, object]:
        """
        Run the model with the given ``batch``. Update the trainable variables only if ``train`` is true.

        Fetch and return all the model outputs as a dict.

        :param batch: batch dict ``{source_name: values}``
        :param train: flag whether parameters update (``train_op``) should be included in fetches
        :param stream: stream wrapper (useful for precise buffer management)
        :raise ValueError: if an output is wrongly typed or its batch size differs from the input batch size
        :return: outputs dict
        """
        # setup the feed dict
        batch_size = len(batch[next(iter(batch))])
        tower_batch_size = math.ceil(batch_size / len(self._towers))
        nonempty_towers = batch_size // tower_batch_size + (0 if batch_size % tower_batch_size == 0 else 1)

        feed_dict = {self._is_training: train}
        fetches = [self._train_ops[nonempty_towers-1]] if train else []
        fetches += self._extra_outputs

        for i, tower in enumerate(self._towers):
            if i*tower_batch_size < batch_size:
                for placeholder_name in self.input_names:
                    tower_batch = batch[placeholder_name][i*tower_batch_size:(i+1)*tower_batch_size]
                    feed_dict[tower[placeholder_name]] = tower_batch
                for output_name in self.output_names:
                    fetches.append(tower[output_name])

        run_fn = self._profiler.run if self._profile else self._session.run

        # run the computational graph for one batch and allow buffering in the meanwhile
        if stream is not None:
            with stream.allow_buffering:
                outputs = run_fn(fetches, feed_dict)
        else:
            outputs = run_fn(fetches, feed_dict)

        if train:
            outputs = outputs[1:]

        extra_outputs = outputs[:len(self._extra_outputs)]
        outputs = outputs[len(self._extra_outputs):]

        for i, output in enumerate(outputs):
            if not isinstance(output, (list, tuple, np.ndarray)):
                output_name = self.output_names[i % len(self.output_names)]
                raise ValueError('Model output `{}` is not one of list, tuple or numpy array. Found `{}` instead. '
                                 'Model outputs should be batched, i.e. the first dimension should refer to '
                                 'different examples.'.format(output_name, type(output)))

        # stack partial tower outputs
        num_outputs = len(self.output_names)
        stacked_outputs = [np.concatenate(outputs[i::num_outputs]) for i in range(num_outputs)]

        extra_outputs_names = list(map(lambda x: x.name.split(':')[0], self._extra_outputs))
        return dict(zip(self.output_names+extra_outputs_names, stacked_outputs+extra_outputs))

    def save(self, name_suffix: str = '') -> str:
        """
        Save current tensorflow graph to a checkpoint named with the given name suffix.

        The checkpoint will be locaced in self.log_dir directory.
        :param name_suffix: saved checkpoint name suffix
        :return: path to the saved checkpoint
        """
        if name_suffix != '':
            name_suffix = '_'+name_suffix
        graph_path = path.join(self._log_dir, 'model{}.graph'.format(name_suffix))
        checkpoint_path = path.join(self._log_dir, 'model{}.ckpt'.format(name_suffix))
        frozen_graph_path = path.join(self._log_dir, 'model{}.pb'.format(name_suffix))

        tf.train.write_graph(self._session.graph_def, '', graph_path, as_text=False)
        self._saver.save(self._session, checkpoint_path)

        if self._freeze_graph:
            with tf.Graph().as_default():
                freeze_graph(input_graph=graph_path,
                             input_checkpoint=checkpoint_path,
                             output_node_names=self.output_names,
                             output_graph=frozen_graph_path)

        if self._quantize:
            logging.info('Quantizing and saving the model')
            # Okay, this is a bit tricky, we need to create a new eval graph, restore the variables
            # create a TFLiteConverter from session and then we can quantize the thing
            session_config = None
            if self._session_config:
                session_config = tf.ConfigProto(**self._session_config)
            with tf.Graph().as_default() as eval_graph, tf.Session(config=session_config).as_default() as eval_sess:
                # lets clear the quantize inputs and outputs in order to prevent stacking
                # of inputs and outputs from multiple _create_model calls
                self._quantize_inputs = []
                self._quantize_outputs = []
                old_is_training = self._is_training  # some models may use self._is_training so we need to re-create it
                with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):  # lets create the model as usual
                    self._is_training = tf.constant(False, name=BaseModel.TRAINING_FLAG_NAME)
                    with self._towers[0]:  # we enforce either n_gpus=0 or n_gpus=1 so this is safe
                        self._create_model(**self._kwargs)
                self._is_training = old_is_training
                tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)

            # copy the current variables to the eval graph
            logging.info('Copying variables to the eval graph, this may take a while')
            for target_var in eval_graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                source_tensor = self.graph.get_tensor_by_name(target_var.name)
                eval_sess.run(target_var.assign(self.session.run(source_tensor)))

            logging.info(f'Converting model with inputs {self._quantize_inputs} and outputs f{self._quantize_outputs}')
            converter = tf.lite.TFLiteConverter.from_session(eval_sess, self._quantize_inputs, self._quantize_outputs)
            converter.inference_type = tf.uint8
            # TODO: this will work only for inputs scaled to [0, positive_num] ... the user was warned though ...
            converter.quantized_input_stats = {input_array: (0.0, 1.0) for input_array in converter.get_input_arrays()}
            quantized = converter.convert()
            quantized_model_path = path.join(self._log_dir, f'{self._quantize_model_name}.tflite')
            with open(quantized_model_path, "wb") as file:
                file.write(quantized)
            logging.info(f'Saving the quantized model to `{quantized_model_path}`')

        return checkpoint_path

    def _register_quantize_input(self, input: tf.Tensor) -> None:
        """Register the given input for quantization."""
        self._quantize_inputs.append(input)

    def _register_quantize_output(self, output: tf.Tensor) -> None:
        """Register the given output for quantization."""
        self._quantize_outputs.append(output)

    def _restore_checkpoint(self, checkpoint_path: str) -> None:
        """
        Restore model from the given ``checkpoint_path``.

        :param checkpoint_path: full path to the checkpoint, e.g. ``my_dir/model_3.ckpt``.
        """
        logging.debug('Loading meta graph')
        saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
        logging.debug('Restoring model')
        saver.restore(self._session, checkpoint_path)

    def _restore_model(self, restore_from: str) -> None:
        """
        Restore TF model from the given ``restore_from`` path and ``restore_model_name``.

        The model name can be derived if the ``restore_from`` is a directory containing exactly one checkpoint or if
        its base name specifies a checkpoint.

        :param restore_from: path to directory from which the model is restored, optionally including model filename
        """

        logging.info('Restoring model from `{}`'.format(restore_from))
        restore_model_name = None
        if not path.isdir(restore_from):
            restore_model_name = path.basename(restore_from)
            restore_from = path.dirname(restore_from)
        assert path.isdir(restore_from), '`BaseModel` expect `restore_from` to be an existing directory.'
        meta_files = glob('{}/*.ckpt.meta'.format(restore_from))

        if len(meta_files) == 0:
            raise ValueError('No `{}/*.ckpt.meta` files found.'.format(restore_from))
        elif len(meta_files) == 1:
            logging.info('Restoring model from checkpoint metafile`{}`'.format(meta_files[0]))
            self._restore_checkpoint(meta_files[0][:-5])
        else:
            logging.info('Multiple checkpoint metafiles found.')

            if restore_model_name is None:
                raise ValueError('There are multiple checkpoint metafiles found in the directory {}. '
                                 'Please specify the full checkpoint path.'.format(restore_from))

            logging.info('Restoring model from checkpoint `{}` located in directory `{}`'.format(restore_model_name,
                                                                                                 restore_from))
            self._restore_checkpoint(path.join(restore_from, restore_model_name))

    def _create_session(self, session_config: Optional[dict]) -> tf.Session:
        """
        Create and return TF Session for this model.

        By default the session is configured with ``tf.ConfigProto`` created with
        the given ``session_config`` as ``**kwargs``. Nested dictionaries such as
        ``gpu_options`` or ``graph_options`` are handled automatically.

        :param session_config: session configuration dict as specified in the config yaml
        :return: TensorFlow session
        """
        if session_config:
            session_config = tf.ConfigProto(**session_config)
        return tf.Session(graph=self._graph, config=session_config)

    def _create_train_ops(self, dependencies: List[List[tf.Operation]], optimizer_config: Optional[dict]) -> None:
        """
        Create the train ops for training. In order to handle incomplete batches, there must be one train op for
        each number of empty towers. E.g. for 2 GPU training, one must define 2 train ops for 1 and 2 towers
        respectively. The train ops must be named ``train_op_1``, ``train_op_2`` etc.
        wherein the suffixed number stands for the number of towers.

        By default the train ops are constructed in the following way:
            - optimizer is created from the ``model.optimizer`` configuration dict
            - REGULARIZATION_LOSSES collection is summed to ``regularization_loss``
            - gradients minimizing the respective tower losses and ``regularization_loss`` are computed
            - for each number of non-empty towers
                - gradients of the respective towers are averaged and applied

        To implement a custom behavior, override this method and create your own op named as :py:attr:`TRAIN_OP_NAME`.

        .. code-block:: yaml
            :caption: example optimizer config

            model:
                optimizer:
                    class: RMSPropOptimizer
                    learning_rate: 0.001

        :param dependencies: a list of dependent operations (e.g. batch normalization updates) for each number of towers
        :param optimizer_config: optimizer configuration dict
        """
        if optimizer_config is None:
            raise ValueError('Optimizer config was not specified although it is required for creating the train op. '
                             'Please specify the configuration in `model.optimizer`.')
        grads_and_vars = []
        if self._quantize:
            logging.info(f'Applying quantization aware training updates with {self._quantize_delay} delay')
            tf.contrib.quantize.create_training_graph(input_graph=self.graph, quant_delay=self._quantize_delay)

        optimizer = create_optimizer(optimizer_config)
        regularization_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regularization_loss = tf.reduce_sum(tf.stack(regularization_losses))
        if regularization_losses:
            logging.info('\tAdding regularization losses')
            logging.debug('\tRegularization losses: %s', [var.name for var in regularization_losses])
        for tower in self._towers:
            with tower:
                grads_and_vars.append(optimizer.compute_gradients(tf.reduce_mean(tower.loss) + regularization_loss))

        # gradient clipping
        if self._clip_gradient is not None:
            for tower_grads_vars in grads_and_vars:
                for i, (grad, var) in enumerate(tower_grads_vars):
                    if grad is not None:
                        tower_grads_vars[i] = (tf.clip_by_value(grad, -self._clip_gradient, self._clip_gradient), var)

        for i in range(len(self._towers)):
            with tf.control_dependencies(dependencies[i]):
                optimizer.apply_gradients(average_gradients(grads_and_vars[:(i + 1)]),
                                          name=BaseModel.TRAIN_OP_NAME + '_{}'.format(i + 1))

    def _create_model(self, **kwargs) -> None:
        """
        Create your TensorFlow model.

        Every model has to define:

        - loss tensor named according to given ``loss_name``
        - input placeholders and output tensors named according to the specified input and output names

        .. warning::
            To support multi-GPU training, all the variables must be created with ``tf.get_variable``
            and appropriate variable scopes.

        :param kwargs: model configuration as specified in ``model`` section of the configuration file
        """
        raise NotImplementedError('`_create_model` method must be implemented in order to construct a new model.')

    def _initialize_variables(self, init_from: Optional[str] = None, **kwargs) -> None:
        """
        Initialize variables of your TensorFlow model.

        By default variables are initialized randomly.

        .. tip::

            Override this method to load variables from some check-point and fine-tune the model.

        :param kwargs: model configuration as specified in ``model`` section of the configuration file
        """
        self._session.run(tf.global_variables_initializer())
        self._session.run(tf.local_variables_initializer())

        if init_from:
            logging.info('Initializing variables from %s', init_from)
            session_config = None
            if self._session_config:
                session_config = tf.ConfigProto(**self._session_config)
            with tf.Graph().as_default() as graph, tf.Session(config=session_config).as_default() as sess:
                saver = tf.train.import_meta_graph(init_from + '.meta')
                saver.restore(sess, init_from)
                for target_var in self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                    try:
                        source_tensor = graph.get_tensor_by_name(target_var.name)
                        self.session.run(target_var.assign(sess.run(source_tensor)))
                    except KeyError:
                        logging.warning(f'Could not initialize {target_var.name} as it is missing in the source graph')
