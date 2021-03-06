Tutorial
########

Let’s imagine that one of your colleagues prepared a **emloop** compatible dataset for the common task
of recognizing images of hand-written digits, and you are responsible for the implementation of a
baseline neural network for said dataset. That’s where **emloop-tensorflow** comes in: it’s simple, it’s
fast and it pretty much does all the work for you. Here is how you set it up:

First emloop-tensorflow model
-----------------------------
Create a simple convolutional network by writing only a couple of lines.

.. code-block:: python
  :caption: convnet.py

   import tensorflow as tf
   import tensorflow.contrib.keras as K
   import emloop_tensorflow as eltf


   class SimpleConvNet(eltf.BaseModel):

       def _create_model(self):
           images = tf.placeholder(tf.float32, shape=[None, 28, 28], name='images')
           labels = tf.placeholder(tf.int64, shape=[None], name='labels')

           with tf.variable_scope('conv1'):
               net = tf.expand_dims(images, -1)
               net = K.layers.Conv2D(20, 5)(net)
               net = K.layers.MaxPool2D()(net)
           with tf.variable_scope('conv2'):
               net = K.layers.Conv2D(50, 3)(net)
               net = K.layers.MaxPool2D()(net)
           with tf.variable_scope('dense3'):
               net = K.layers.Flatten()(net)
               net = K.layers.Dropout(0.4).apply(net, training=self.is_training)
               net = K.layers.Dense(100)(net)
           with tf.variable_scope('dense4'):
               logits = K.layers.Dense(10, activation=None)(net)

           loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
           tf.identity(loss, name='loss')
           predictions = tf.argmax(logits, 1, name='predictions')
           tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32, name='accuracy'))

.. tip::
   It does not matter how the graph is created, as long the input/output tensors
   are named properly. Feel free to use your
   favorite framework such as `Keras <https://keras.io/>`_,
   `Slim <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>`_ or vanilla TensorFlow.

Next, write a configuration file putting everything together.

.. code-block:: yaml
   :caption: config.yaml

   dataset:
     class: datasets.MNISTDataset

   model:
     name: ConvNetExample
     class: convnet.SimpleConvNet
     optimizer:
       class: AdamOptimizer
       learning_rate: 0.001
     inputs: [images, labels]
     outputs: [accuracy, predictions, loss]

   main_loop:
     extra_streams: [test]

   hooks:
   - ComputeStats:
       variables:
         loss: [mean, std]
         accuracy: [mean]
   - LogVariables
   - CatchSigint
   - StopAfter:
       minutes: 5

Finally run the training with:

.. code-block:: bash

   emloop train <path to config.yaml>

.. tip::
   Full example may be found in our
   `emloop examples repository @GitHub <https://github.com/iterait/emloop-examples/tree/master/mnist_convnet>`_.

Basic configuration
-------------------
Most of the heavy lifting was done by the **emloop** and **emloop-tensorflow** – just as it should be!
Only the model itself and a few unavoidable configuration options had to be specified. In this
section, we will go through the basic configuration options in greater detail.

Inputs & Outputs
~~~~~~~~~~~~~~~~
To connect the model to the data stream, its *inputs* must be defined in the config.
Similarly, the variables, which are to be fetched, are configured by the *outputs*.
Both *inputs* and *outputs* are nothing more than lists of variable names.
The respective tensors are expected to be found in the created TF graph.

.. code-block:: yaml
   :caption: configuring inputs and outputs
   :emphasize-lines: 4, 5

     optimizer:
       class: AdamOptimizer
       learning_rate: 0.001
     inputs: [images, labels]
     outputs: [accuracy, predictions, loss]
   hooks:

Optimizer
~~~~~~~~~
By default, **emloop-tensorflow** creates a TF optimizer specified in the configuration and attempts to
minimize the model ``loss``.
Therefore, we need to both, specify the optimizer and include a tensor named ``loss`` in the graph.
Arbitrary `TF Optimizer <https://www.tensorflow.org/api_guides/python/train>`_ may be referenced by its name.

.. code-block:: yaml
   :caption: config.yaml
   :emphasize-lines: 2, 3, 4

      class: convnet.SimpleConvNet
      optimizer:
        class: AdamOptimizer
        learning_rate: 0.001
      inputs: [images, labels]

Model parameters
~~~~~~~~~~~~~~~~
Note that the model (hyper-)parameters such as the number of layers were all hard-coded in our example.
Contrary to that, those parameters happen to frequently change as we search for the best performing configuration.

In **emloop**, model parameters may be defined and configured quite easily.
For example, to introduce new ``dense_size`` parameter controlling the number of neurons in the fully connected layer,
one would update the code as follows:

.. code-block:: python
   :caption: convnet.py
   :emphasize-lines: 1, 5

       def _create_model(self, dense_size:int =100):
           ...
           with tf.variable_scope('dense3'):
               net = K.layers.Flatten()(net)
               net = K.layers.Dense(dense_size)(net)

.. code-block:: yaml
   :caption: passing the model parameters
   :emphasize-lines: 4

   model:
     name: ConvNetExample
     class: convnet.SimpleConvNet
     dense_size: 50
     optimizer:

In fact, **any** parameter found in the configuration under the ``model`` section is directly forwarded to the
``_create_model`` function.
This way, the whole model can be easily parametrized.

.. tip::
   Try to experiment with the ``dense_size`` parameter. How small the fully connected layer can be before the performance
   degrades?

Next steps
----------
See our `emloop examples repository @GitHub <https://github.com/iterait/emloop-examples>`_.
for additional examples or read the :py:class:`emloop_tensorflow.BaseModel` for a full list of customization options.

This project contains additional utility functions and **emloop** hooks documented in the
:doc:`emloop_tensorflow/index`.
Make sure you don`t miss the :py:class:`emloop_tensorflow.hooks.WriteTensorboard` hook providing seamless integration
with `TensorBoard <https://www.tensorflow.org/get_started/summaries_and_tensorboard>`_.
