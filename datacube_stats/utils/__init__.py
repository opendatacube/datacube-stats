"""
Useful utilities used in Stats
"""
import itertools


def tile_iter(tile, chunk_size):
    """
    Return the sequence of chunks to split a tile into computable regions.

    :param Tile tile: a tile of `.shape` size containing `.dim` dimensions
    :param chunk_size: dict of dimension sizes
    :return: Sequence of chunks to iterate across the entire tile
    """
    defaults = tuple(size if dim in chunk_size else None
                     for dim, size in zip(tile.dims, tile.shape))
    steps = _tuplify(tile.dims, chunk_size, defaults)
    return _block_iter(steps, tile.shape)


def _tuplify(keys, values, defaults):
    assert not set(values.keys()) - set(keys), 'bad keys'
    return tuple(values.get(key, default) for key, default in zip(keys, defaults))


def _block_iter(steps, shape):
    return itertools.product(*(_slicify(step, size) for step, size in zip(steps, shape)))


def _slicify(step, size):
    if step is None:
        return [slice(None)]
    else:
        return (slice(i, min(i + step, size)) for i in range(0, size, step))
