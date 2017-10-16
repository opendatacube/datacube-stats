from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers
from hypothesis import given
import numpy as np
from datacube_stats.utils import wofs_fuser


def is_dry(data):
    d = data & ~4  # unset land/sea flag
    return d == 0


def is_wet(data):
    d = data & ~4  # unset land/sea flag
    return d == 128


def is_clear(data):
    return is_wet(data) | is_dry(data)

wofs_arrays = arrays(np.uint8, (1,), elements=integers(0, 255))


@given(wofs_arrays, wofs_arrays)
def test(dest, src):
    orig_dest = np.copy(dest)

    wofs_fuser(dest, src)  # Result stored in `dest`

    if src == orig_dest:
        assert src == orig_dest == dest

    if not is_clear(src) and not is_clear(orig_dest):
        assert not is_clear(dest)

    if is_wet(src) and is_wet(orig_dest):
        assert is_wet(src)

    if is_dry(src) and is_dry(orig_dest):
        assert is_dry(dest)
