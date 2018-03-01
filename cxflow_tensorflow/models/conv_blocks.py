from typing import Tuple, Callable, Optional, Union

import tensorflow as tf

from ..ops import repeat
from .blocks import BaseBlock

__all__ = ['ConvBaseBlock', 'ConvBlock', 'ResBlock', 'IncBlock', 'PoolBaseBlock', 'MaxPoolBlock', 'AveragePoolBlock',
           'UnPoolBlock']


class ConvBaseBlock(BaseBlock):
    """Base block for convolution-like blocks."""

    def __init__(self, conv_fn: Callable=tf.identity, bn_fn: Callable=tf.identity, ln_fn: Callable=tf.identity,
                 extra_dim: Tuple[int]=(), **kwargs):
        """
        Try to parse and create new :py:class:`ConvBaseBlock`.

        :param conv_fn: convolution function (with ``num_outputs``, ``kernel_size``, ``stride`` and ``scope`` kwargs)
        :param bn_fn: batch normalization function
        :param ln_fn: layer normalization function
        :param extra_dim: extra dimension (for 5-dim tensors)
        :param kwargs:
        """
        self._conv_fn = conv_fn
        self._bn_fn = bn_fn
        self._ln_fn = ln_fn
        self._extra_dim = extra_dim
        super().__init__(**kwargs)


class ConvBlock(ConvBaseBlock):
    """
    2D/3D convolutional layer.

    **code**: ``(num_filters)c[(time_kernel_size)-](kernel_size)[s(stride)]``

    **examples**: ``64c3``, ``64c3s2``, ``64c3-5s2`` (convolution with kernel (3x5x5) assuming BTHWC data)
    """

    def __init__(self, **kwargs):
        """Try to parse and create new :py:class:`ConvBlock`."""
        super().__init__(regexp='([1-9][0-9]*)c(([1-9][0-9]*)-)?([1-9][0-9]*)(s([1-9][0-9]*))?',
                         defaults=(None, None, 1, None,  None, 1),
                         **kwargs)

    def _handle_parsed_args(self, channels: str, _, time_kernel: Union[str, int], kernel: str,
                            __, stride: Union[str, int]) -> None:
        """
        Handle parsed arguments.

        :param channels: number of output channels
        :param time_kernel: time kernel size (default 1)
        :param kernel: spatial kernel size
        :param stride: spatial stride (default 1)
        """
        self._channels, self._kernel, self._time_kernel, self._stride = \
            int(channels), int(kernel), int(time_kernel), int(stride)

    def apply(self, x: tf.Tensor) -> tf.Tensor:
        time_kernel = self._extra_dim
        if self._time_kernel > 1:
            if len(x.shape) != 5:
                raise ValueError('Conv block with time kernel size {} can be applied only to 5-dim tensors '
                                 '({} were given).'.format(self._time_kernel, len(x.shape)))
            time_kernel = (self._time_kernel,)
        x = self._conv_fn(x, num_outputs=self._channels, kernel_size=time_kernel+(self._kernel, self._kernel),
                          stride=self._extra_dim+(self._stride, self._stride), scope='inner')
        x = self._bn_fn(x)
        x = self._ln_fn(x)
        return x

    def inverse_code(self) -> str:
        if self._stride > 1:
            raise ValueError('Inverse code for conv block is not defined for stride `{}`'.format(self._stride))
        return self._code


class IncBlock(ConvBaseBlock):
    """Inception-v3 block."""

    def __init__(self, pool_fn: Callable=tf.identity, **kwargs):
        """
        Try to parse and create new inception-v3 block.

        :param pool_fn: pooling function
        """
        self._mp_fn = pool_fn
        super().__init__(regexp='([1-9][0-9]*)inc', **kwargs)

    def _handle_parsed_args(self, channels: str) -> None:
        self._channels = int(channels)

    def apply(self, x: tf.Tensor) -> tf.Tensor:
        b1x1_channels = self._channels // 4
        b5x5_channels = b1x1_channels // 2
        b5x5r_channels = (b5x5_channels * 2) // 3
        bpool_channels = b1x1_channels // 2
        b3x3_channels = self._channels - (b5x5_channels + bpool_channels + b1x1_channels)
        b3x3r_channels = (b3x3_channels * 2) // 3

        with tf.variable_scope('b11'):
            inception_1x1 = self._conv_fn(x, num_outputs=b1x1_channels, kernel_size=self._extra_dim + (1, 1),
                                          scope='b11i')

        with tf.variable_scope('b33'):
            inception_3x3 = self._conv_fn(x, num_outputs=b3x3r_channels, kernel_size=self._extra_dim + (1, 1),
                                          scope='_b33ri')
            inception_3x3 = self._conv_fn(inception_3x3, num_outputs=b3x3_channels,
                                          kernel_size=self._extra_dim + (3, 3), scope='_b33i')

        with tf.variable_scope('b55'):
            inception_5x5 = self._conv_fn(x, num_outputs=b5x5r_channels, kernel_size=self._extra_dim + (1, 1),
                                          scope='_b55ri')
            inception_5x5 = self._conv_fn(inception_5x5, num_outputs=b5x5_channels,
                                          kernel_size=self._extra_dim + (5, 5), scope='_b55i')

        with tf.variable_scope('bp'):
            inception_pool = self._mp_fn(x, kernel_size=self._extra_dim + (3, 3), stride=1, padding='SAME')
            inception_pool = self._conv_fn(inception_pool, num_outputs=bpool_channels,
                                           kernel_size=self._extra_dim + (1, 1), scope='bpi')

        x = tf.concat([inception_1x1, inception_3x3, inception_5x5, inception_pool], axis=(len(x.shape) - 1))
        x = self._bn_fn(x)
        x = self._ln_fn(x)

        return x


class ResBlock(ConvBaseBlock):
    """Original residual block."""

    def __init__(self, **kwargs):
        """Try to parse and create new :py:class:`ResBlock`."""
        super().__init__(regexp='([1-9][0-9]*)res(s([1-9][0-9]*))?', defaults=(None, None, 1), **kwargs)

    def _handle_parsed_args(self, channels: str, _, stride: str) -> None:
        self._channels, self._stride = int(channels), int(stride)

    def apply(self, x: tf.Tensor) -> tf.Tensor:
        channels_in = x.get_shape().as_list()[-1]

        if self._stride == 1 and channels_in == self._channels:
            skip_connection = x
        else:
            skip_connection = self._conv_fn(x, num_outputs=self._channels, kernel_size=self._extra_dim + (1, 1),
                                            stride=self._extra_dim+(self._stride, self._stride), scope='skip')
        with tf.variable_scope('c1'):
            x = self._conv_fn(x, num_outputs=self._channels, kernel_size=self._extra_dim+(3, 3),
                              stride=self._extra_dim+(self._stride, self._stride), scope='c1')
            x = self._bn_fn(x)
            x = self._ln_fn(x)
        with tf.variable_scope('c2'):
            x = self._conv_fn(x, num_outputs=self._channels, kernel_size=self._extra_dim+(3, 3), stride=1, scope='c2')
            x = self._bn_fn(x)
            x = self._ln_fn(x)
        x += skip_connection
        return x

    def inverse_code(self) -> str:
        if self._stride > 1:
            raise ValueError('Inverse code for res block is not defined for stride `{}`'.format(self._stride))
        return self._code


class PoolBaseBlock(BaseBlock):
    """Base block for pooling blocks."""

    def __init__(self, prefix: str='', pool_fn: Callable=tf.identity, extra_dim: Tuple[int]=(), **kwargs):
        """
        Try to parse and create new :py:class:`PoolBaseBlock`.

        :param prefix: prefix for the regular expression
        :param pool_fn: pooling function (with ``kernel_size`` and ``stride`` kwargs)
        :param extra_dim: extra dimension (for 5-dim tensors)
        """
        self._pool_fn = pool_fn
        self._extra_dim = extra_dim
        super().__init__(regexp=prefix+'p([1-9][0-9]*)(s([1-9][0-9]*))?', **kwargs)

    def _handle_parsed_args(self, kernel: str, _, stride: Optional[str]) -> None:
        if stride is None:
            stride = kernel
        self._kernel, self._stride = int(kernel), int(stride)

    @property
    def kernel_size(self) -> int:
        return self._kernel

    def inverse_code(self) -> str:
        if self._stride != self._kernel:
            raise ValueError('Inverse code for pool block is not defined for stride `{}`'.format(self._stride))
        return UnPoolBlock.CODE_PREFIX + self._code[1:]

    def apply(self, x: tf.Tensor) -> tf.Tensor:
        return self._pool_fn(x, kernel_size=self._extra_dim+(self._kernel, self._kernel),
                             stride=self._extra_dim+(self._stride, self._stride))


class MaxPoolBlock(PoolBaseBlock):
    """Max pooling block."""

    def __init__(self, mp_fn: Callable=tf.identity, **kwargs):
        if 'pool_fn' in kwargs:
            kwargs.pop('pool_fn')
        super().__init__(prefix='m', pool_fn=mp_fn, **kwargs)


class AveragePoolBlock(PoolBaseBlock):
    """Average pooling block."""
    def __init__(self, ap_fn: Callable=tf.identity, **kwargs):
        if 'pool_fn' in kwargs:
            kwargs.pop('pool_fn')
        super().__init__(prefix='a', pool_fn=ap_fn, **kwargs)


class UnPoolBlock(BaseBlock):
    """Un pooling block."""

    CODE_PREFIX = 'u'
    """Un pooling code prefix character."""

    def __init__(self, **kwargs):
        super().__init__(regexp=UnPoolBlock.CODE_PREFIX+'p([1-9][0-9]*)', **kwargs)

    def _handle_parsed_args(self, kernel: str) -> None:
        self._kernel = int(kernel)

    def apply(self, x: tf.Tensor) -> tf.Tensor:
        x = repeat(x, self._kernel, len(x.shape) - 3)
        x = repeat(x, self._kernel, len(x.shape) - 2)
        return x
