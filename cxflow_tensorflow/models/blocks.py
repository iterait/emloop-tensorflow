from typing import Iterable, Mapping

import re
from typing import Optional, Tuple, Any, Type

import tensorflow as tf

__all__ = ['Block', 'BaseBlock', 'UnrecognizedCodeError', 'get_block_instance']


class Block:
    """
    Base **cxflow-tensorflow** block concept.

    A block shall be configurable from a single string (code) and provide :py:meth:`apply` method
    which takes and returns a single tensor.
    """

    def __init__(self, code: str, **kwargs):
        """
        Create new :py:class:`Block` from the given ``code``.

        :param code: short human-readable code for block parametrization
        """
        self._code = code

    def inverse_code(self) -> str:
        """Get code for the inverse block."""
        return self._code

    def apply(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply the block to the given tensor.

        :param x: Input tensor
        :return: Output tensor
        """
        raise NotImplementedError('Block must implement the ``apply`` method.')


class BaseBlock(Block):
    """
    Base **cxflow-tensorflow** block.
    """

    def __init__(self, code: str, regexp: str, defaults: Optional[Tuple[Optional[Any]]]=None, **kwargs):
        """
        Try to parse and create new :py:class:`BaseBlock` using the following procedure:

        1. try to match the given ``regexp`` to the given ``code`` (raise :py:class:`UnrecognizedCodeError` if it fails)
        2. pass the matched groups to the :py:meth:`_handle_parsed_args` method (must be overridden)

        :param code: block configuration code
        :param regexp: regular expression to be matched to the code
        :param defaults: (in-order) default arguments for each of the ``regexp`` groups (optional)
        :raise UnrecognizedCodeError: if the given ``regexp`` can not be matched to the given ``code``
        """
        super().__init__(code, **kwargs)
        searcher = re.fullmatch(regexp, code)
        if searcher is None:
            raise UnrecognizedCodeError('Can\' interpret `{}` as layer config'.format(code))
        defaults = defaults or (None,)*len(searcher.groups())
        self._handle_parsed_args(*(group or default for group, default in zip(searcher.groups(), defaults)))

    def _handle_parsed_args(self, *args) -> None:
        """
        Handle (most likely save) the arguments matched with the ``regexp`` in the ``code``.

        :param args: parsed arguments
        """
        raise NotImplementedError('Block must override `_handle_parsed_args` method')


class UnrecognizedCodeError(ValueError):
    pass


def get_block_instance(code: str, blocks: Iterable[Type[Block]],
                       block_kwargs: Optional[Mapping]=None) -> Tuple[Block, Type[Block]]:
    """
    Try to create a block instance from the given code and an iterable of block candidates.

    :param code: block code
    :param blocks: iterable of block candidates
    :param block_kwargs: additional block kwargs
    :raise UnrecognizedCodeError: if none of the block candidates can parse the given code
    :return: a tuple of (block instance, block type)
    """
    block_kwargs = block_kwargs or {}
    for block in blocks:
        try:
            block_instance = block(code=code, **block_kwargs)
            return block_instance, block
        except UnrecognizedCodeError:
            pass
    else:
        raise UnrecognizedCodeError('Block code `{}` was not recognized by any of the following blocks `{}`'
                                    .format(code, [block.__name__ for block in blocks]))
