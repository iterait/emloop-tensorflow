# cxflow-tensorflow
[![CircleCI](https://circleci.com/gh/Cognexa/cxflow-tensorflow/tree/master.svg?style=shield)](https://circleci.com/gh/Cognexa/cxflow-tensorflow/tree/master)
[![Development Status](https://img.shields.io/badge/status-CX%20Regular-brightgreen.svg?style=flat)]()
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)]()
[![Master Developer](https://img.shields.io/badge/master-Petr%20Bělohlávek-lightgrey.svg?style=flat)]()

TensorFlow extension for [cxflow framework](https://github.com/cognexa/cxflow) allowing to train nets defined in TensorFlow.

## Installation

1. Install TensorFlow according to the [official instructions](https://www.tensorflow.org/install/)

2. Install cxflow-tensorflow with pip
```
pip install cxflow-tf
```

## Usage
After installation, the following classes are available:

### Nets

- `cxflow_tf.BaseTFNet`
- `cxflow_tf.BaseTFNetRestore`

### Hooks

- `cxflow_tf.TensorBoardHook`
- `cxflow_tf.LRDecayHook`

### Additional utils

Additional helper and util functions in `cxflow_tf.utils` module.

## Testing
Unit tests might be run by `$ python setup.py test`.

## License
MIT License
