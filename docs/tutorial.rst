Tutorial
########

Imagine you colleague prepared a **cxflow** compatible dataset for the well 
known task of recognizing images of hand-written letters. Now, your task is to 
implement a baseline neural network for that. It can not be done easier than
with cxflow-tensorflow.

First cxflow-tensorflow model
-----------------------------
A simple convolutional network takes only a couple of lines.

.. code-block:: python
  :caption: convnet.py

   import tensorflow as tf
   import tensorflow.contrib.keras as K
   import cxflow_tensorflow as cxtf


   class SimpleConvNet(cxtf.BaseModel):

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
   It does not matter how the graph is created as long the input/output tensors 
   are named properly. Feel free to use your
   favorite framework such as `Keras <https://keras.io/>`_,
   `Slim <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>`_ or vanilla TensorFlow.

The next step is to write a configuration file putting everything together:

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

   cxflow train <path to config.yaml>

.. tip::
   Full example may be found in our
   `GitHub examples repository <https://github.com/Cognexa/cxflow-examples/tree/master/convnet>`_.

Basic configuration
-------------------
As intended, most of the heavy lifting was done by the **cxflow** and **cxflow-tensorflow**.
Only the model itself and a few unavoidable configuration options had to be specified.
In this section, we go through the basic configuration options in greater detail.

Inputs & Outputs
~~~~~~~~~~~~~~~~
To connect the model to the data stream, its *inputs* must be defined in the config.
Similarly, the variables to be fetched are configured by the *outputs*.
Both *inputs* and *outputs* are nothing more than lists of variable names.
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
By default, **cxflow-tensorflow** creates a TF optimizer specified in the configuration and attempts to
minimize the model ``loss``.
Hence, we need to both specify the optimizer and include a tensor named ``loss`` in the graph.
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

In **cxflow**, model parameters may be defined and configured quite easily.
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

In fact, **any** parameter found in the configuration under the ``model`` 
section is directly forwarded
to the ``_create_model`` function. This way, the whole model can be easily 
parametrized.

.. tip::
   Try to experiment with the ``dense_size`` parameter. How small the fully connected layer can be before the performance
   degrades?

Next steps
----------
See our `GitHub examples repository <https://github.com/Cognexa/cxflow-examples>`_.
for additional examples or read the :py:class:`cxflow_tensorflow.BaseModel` reference for the full list of
customization options.

This project contains additional util functions and **cxflow** hooks documented in the :doc:`cxflow_tensorflow/index`.
Be sure you do not miss the :py:class:`cxflow_tensorflow.hooks.WriteTensorboard` hook providing seamless integration with
`TensorBoard <https://www.tensorflow.org/get_started/summaries_and_tensorboard>`_.