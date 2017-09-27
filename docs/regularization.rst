Model regularization
####################

Model over-fitting is the most common cause of sub-optimal model performance.
For this reason, various model regularization techniques have emerged in the past years.
All of them, naturally, can be used in **cxflow-tensorflow**.

In this brief tutorial we show how to incorporate the most common ones to **cxflow-tensorflow** models.

Dropout
-------

Dropout is well known technique preventing over-fitting introduced by
`Srivastava et al. <https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf>`_ in 2014.
Simply said, dropout layer randomly selects certain portion of units and sets them to zero.
In fact, this is done only in the training phase.
For this reason, we need to distinguish training and evaluation phases.
In **cxflow-tensorflow**, this information is fed to the model via special scalar tensor defined in
:py:class:`cxflow_tensorflow.BaseModel` and accessible through ``self.is_training`` property.

E.g. we can add dropout layer during the model creation as follows:

.. code-block:: python
    :caption: tensorflow layers

    def _create_model(self):
        # ...
        output = tf.dense(input, 512)
        output = tf.dropout(input, 0.5, training=self.is_training)

or if we prefer Keras:

.. code-block:: python
    :caption: Keras

    def _create_model(self):
        # ...
        output = K.layers.Dense(512)(input)
        output = K.layers.Dropout(0.5)(output, training=self.is_training)

Weight decay
------------

Weight decay, also known as L2 regularization is yet another common way to regularize deep learning models.
To apply weight decay in **cxflow-tensorflow**, add the regularization cost the ``REGULARIZATION_LOSSES``
 TensorFlow graph collection.

E.g.:

.. code-block:: python
    :caption: adding explicit regularization loss

    def _create_model(self):
        # ...
        self.graph.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, my_loss)

Luckily, most of the common layer APIs will do that for you. E.g.

.. code-block:: python
    :caption: implicit regularization

    def _create_model(self):
        # ...
        output = K.layers.Conv2D(64, (3, 3), kernel_regularizer=K.regularizers.l2(0.0001))(net)

.. tip::
    To see the list of utilized regularization loss names, run **cxflow** with ``--verbose`` argument.
