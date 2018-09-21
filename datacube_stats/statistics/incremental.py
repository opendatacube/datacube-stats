import numpy as np
import xarray

from datacube.storage.masking import create_mask_value
from datacube_stats.incremental_stats import (mk_incremental_sum, mk_incremental_or,
                                              compose_proc, broadcast_proc)
from datacube_stats.utils import mk_masker, first_var

from .core import Statistic, Measurement


class MaskMultiCounter(Statistic):
    # pylint: disable=redefined-builtin
    def __init__(self, vars, nodata_flags=None):
        """
        vars:
           - name: <output_variable_name: String>
             simple: <optional Bool, default: False>
             flags:
               field_name1: expected_value1
               field_name2: expected_value2

        # optional, define input nodata as a mask
        # when all inputs match this, then output will be set to nodata
        # this allows to distinguish 0 from nodata

        nodata_flags:
           contiguous: False

        If variable is marked simple, then there is no distinction between 0 and nodata.
        """
        self._vars = [v.copy() for v in vars]
        self._nodata_flags = nodata_flags
        self._valid_pq_mask = None

    def measurements(self, input_measurements):
        nodata = -1
        bit_defs = input_measurements[0].flags_definition

        if self._nodata_flags is not None:
            self._valid_pq_mask = mk_masker(*create_mask_value(bit_defs, **self._nodata_flags), invert=True)

        for v in self._vars:
            flags = v['flags']
            v['_mask'] = create_mask_value(bit_defs, **flags)
            v['mask'] = mk_masker(*v['_mask'])

        return [Measurement(name=v['name'], dtype='int16', units='1', nodata=nodata)
                for v in self._vars]

    def is_iterative(self):
        return True

    def make_iterative_proc(self):
        def _to_mask(ds):
            da = first_var(ds)
            return xarray.Dataset({v['name']: v['mask'](da) for v in self._vars},
                                  attrs=ds.attrs)

        # PQ -> BoolMasks: DataSet<Bool> -> Sum: DataSet<int16>
        proc = compose_proc(_to_mask,
                            proc=mk_incremental_sum(dtype='int16'))

        if self._valid_pq_mask is None:
            return proc

        def invalid_data_mask(da):
            mm = da.values
            if mm.all():  # All pixels had at least one valid observation
                return None
            return np.logical_not(mm, out=mm)

        # PQ -> ValidMask:DataArray<Bool> -> OR:DataArray<Bool> |> InvertMask:ndarray<Bool>|None
        valid_proc = compose_proc(lambda ds: self._valid_pq_mask(first_var(ds)),
                                  proc=mk_incremental_or(),
                                  output_transform=invalid_data_mask)

        _vars = {v['name']: v for v in self._vars}

        def apply_mask(ds, mask, nodata=-1):
            if mask is None:
                return ds

            for name, da in ds.data_vars.items():
                simple = _vars[name].get('simple', False)
                if not simple:
                    da.values[mask] = nodata

            return ds

        # Counts      ----\
        #                  +----------- Counts[ InvalidMask ] = nodata
        # InvalidMask ----/

        return broadcast_proc(proc, valid_proc, combine=apply_mask)

    def compute(self, data):
        proc = self.make_iterative_proc()

        for i in range(data.time.shape[0]):
            proc(data.isel(time=slice(i, i + 1)))

        return proc()

    def __repr__(self):
        return 'MaskMultiCounter<{}>'.format(','.join([v['name'] for v in self._vars]))
