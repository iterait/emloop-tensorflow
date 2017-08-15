# cxflow-tensorflow
[![CircleCI](https://circleci.com/gh/Cognexa/cxflow-tensorflow/tree/master.svg?style=shield)](https://circleci.com/gh/Cognexa/cxflow-tensorflow/tree/master)
[![Development Status](https://img.shields.io/badge/status-CX%20Regular-brightgreen.svg?style=flat)]()
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)]()
[![Master Developer](https://img.shields.io/badge/master-Petr%20Bělohlávek-lightgrey.svg?style=flat)]()

TensorFlow extension for [cxflow framework](https://github.com/cognexa/cxflow) allowing to train models defined in TensorFlow.

## Installation

1. Install TensorFlow according to the [official instructions](https://www.tensorflow.org/install/)

2. Install cxflow-tensorflow with pip
```
pip install cxflow-tensorflow
```

## Usage
After installation, the following classes are available:

### Models

- `cxflow_tensorflow.BaseTFModel`
- `cxflow_tensorflow.BaseTFModelRestore`

### Hooks

- `cxflow_tensorflow.TensorBoardHook`
- `cxflow_tensorflow.LRDecayHook`

### Metric utils
- `cxflow_tensorflow.bin_dice` computing Dice score for binary classification
- `cxflow_tensorflow.bin_stats` computing f1, precision and recall scores for binary classification

### Additional utils

Additional helper and util functions in `cxflow_tensorflow.utils` module.

## Testing
Unit tests might be run by `$ python setup.py test`.

## License
MIT License
