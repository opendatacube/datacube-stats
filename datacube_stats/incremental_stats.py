import xarray as xr
from .utils import bunch


def assemble_updater(proc, init, finalise=None):
    """
      proc -- reducing function S_{t-1},X_t -> S_t
      init -- creates initial state from first observed element X_0 -> S_{-1}
      finalise -- Convert final state to result, defaults to identity transform

      s = init(xs[0])
      for x in xs:
         s = proc(s,x)
      result = finalise(s)

    Becomes:

      for x in xs:
         update(x)
      result = update()

    """
    _state = bunch(val=None, empty=True)

    def _proc(*x):

        # Extract result case
        if len(x) == 0:
            if finalise is None:
                return _state.val
            return finalise(_state.val)

        # Very first call case
        if _state.empty:
            _state.val = init(*x)
            _state.empty = False

        _state.val = proc(_state.val, *x)

    return _proc


def mk_incremental_min():
    def init(ds):
        return ds.min(dim='time')

    def proc(min_so_far, ds):
        return xr.ufuncs.fmin(min_so_far, ds.min(dim='time'))

    return assemble_updater(proc, init)


def mk_incremental_max():
    def init(ds):
        return ds.max(dim='time')

    def proc(max_so_far, ds):
        return xr.ufuncs.fmax(max_so_far, ds.max(dim='time'))

    return assemble_updater(proc, init)


def mk_incremental_sum(dtype='float32'):
    def init(ds):
        return xr.zeros_like(ds.isel(time=0), dtype=dtype)

    def proc(count, ds):
        count += ds.sum(dim='time').astype(dtype)
        return count

    return assemble_updater(proc, init)


def mk_incremental_counter(dtype='int16'):
    def init(ds):
        return xr.zeros_like(ds.isel(time=0), dtype=dtype)

    def proc(count, ds):
        count += ds.count(dim='time').astype(dtype)
        return count

    return assemble_updater(proc, init)


def mk_incremental_mean(dtype='float32'):
    op_sum = mk_incremental_sum(dtype)
    op_count = mk_incremental_counter(dtype)

    def finalise():
        s = op_sum()
        n = op_count()
        return s/n.where(n > 0)

    def proc(ds=None):
        if ds is None:
            return finalise()

        op_sum(ds)
        op_count(ds)

    return proc
