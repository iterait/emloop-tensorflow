# cxflow-tensorflow

TensorFlow backend for cxflow framework.

## Development Status

- [![Build Status](https://gitlab.com/Cognexa/cxflow-tensorflow/badges/master/build.svg)](https://gitlab.com/Cognexa/cxflow-tensorflow/builds/)
- [![Development Status](https://img.shields.io/badge/status-CX%20PoC-yellow.svg?style=flat)]()
- [![Master Developer](https://img.shields.io/badge/master-Petr%20Bělohlávek-lightgrey.svg?style=flat)]()

## Requirements
The main dependency is cxflow (see [Cognexa/cxflow](https://gitlab.com/Cognexa/cxflow)
for installation guide).
Other requirements are listed in `requirements.txt`.

## Installation
Installation to a [virtualenv](https://docs.python.org/3/library/venv.html) is suggested, however, completely optional. 

### Standard Installation
1. Install **cxflow-tensorflow** `$ pip install git+https://gitlab.com/Cognexa/cxflow-tensorflow.git`

### Development Installation
1. Clone the **cxflow-tensorflow** repository `$ git clone git@gitlab.com:Cognexa/cxflow-tensorflow.git`
2. Enter the directory `$ cd cxflow-tensorflow`
3. Install **cxflow-tensorflow**: `$ pip install -e .`

## Usage
When `cxflow-tensorflow` installed, the following classes are available:

### Nets

- `cxflow_tf.BaseTFNet`
- `cxflow_tf.BaseTFNetRestore`

### Hooks

- `cxflow_tf.TensorBoardHook`

## Testing
Unit tests might be run by `$ python setup.py test`.

## License
MIT License
