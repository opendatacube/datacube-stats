import xarray as xr
from .utils import bunch, nodata_like, da_is_float, da_nodata


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


def compose_proc(input_transform, proc, output_transform=None):
    """
    Apply input transform before feeding data into an accumulator, optionally
    apply output transform on the result also.

      out_proc(x) == proc( input_transform(x) )
      out_proc()  == output_transform(proc())

    """

    def identity(x):
        return x

    if output_transform is None:
        output_transform = identity

    def _proc(*x):
        if len(x) == 0:
            return output_transform(proc())
        return proc(input_transform(*x))
    return _proc


def broadcast_proc(*procs, combine=None):
    """Combine two or more updaters into one. Input data is passed on to all of
    them, output is combined using combine(...) method or just assembled into a
    tuple if combine argument was omitted.

    combine -- should accept as many arguments as there are procs

    out_proc(x) == proc1(x), proc2(x),...
    out_proc()  == combine(proc1(), proc2(), ...)

    """

    def tuplify(*x):
        return tuple(x)

    if combine is None:
        combine = tuplify

    def _proc(*x):
        if len(x) == 0:
            return combine(*(proc() for proc in procs))

        for proc in procs:
            proc(*x)
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
        return s / n.where(n > 0)

    def proc(ds=None):
        if ds is None:
            return finalise()

        op_sum(ds)
        op_count(ds)

    return proc


def mk_incremental_latest():
    """
    Every new valid pixel overwrites previous pixels. Note this is in order of
    processing, doesn't consider timestamp. So if you use reverse=True option this
    will create "oldest" valid pixel on the output.
    """

    def valid_mask(da):
        if da_is_float(da):
            return xr.ufuncs.isfinite(da)
        return da != da_nodata(da)

    def init(ds):
        return nodata_like(ds.isel(time=0))

    def proc(s, ds):
        for name, da in ds.isel(time=0).data_vars.items():
            s_da = s.get(name)
            m = valid_mask(da)
            s_da.values[m] = da.values[m]

        return s

    return assemble_updater(proc, init)


def mk_incremental_or():
    """
    Logical OR
    """

    def init(ds):
        return xr.zeros_like(ds.isel(time=0))

    def proc(s, ds):
        return xr.ufuncs.logical_or(s, ds.isel(time=0))  # TODO: re-use memory of `s`

    return assemble_updater(proc, init)


def mk_incremental_and():
    """
    Logical AND, assumes boolean data
    """

    def init(ds):
        return xr.ones_like(ds.isel(time=0))

    def proc(s, ds):
        return xr.ufuncs.logical_and(s, ds.isel(time=0))

    return assemble_updater(proc, init)
