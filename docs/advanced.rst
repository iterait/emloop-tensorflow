Profiling networks
------------------
Profiling execution of tensorflow graph can be enabled with following setting:

.. code-block:: yaml
    :caption config.yaml

    model:
        profile: True
        keep_profiles: 10

This saves profiles of last 10 runs to the log directory (output directory).
Profiles are in JSON format and can be viewed using Google Chrome.
To view them go to address `chrome://tracing/` and load the json file.

Gradient clipping
-----------------
For gradient clipping use following setting:

.. code-block:: yaml
    :caption config.yaml

    model:
        clip_gradient: 5.0

This clips the absolute value of gradient to 5.0.
Note that the clipping is done to raw gradients before they are multiplied by learning rate or processed in other ways.
