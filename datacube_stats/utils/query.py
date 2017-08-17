def common_subset(sets, key_by=None):
    """ From a list of lists compute set common to all lists.

        NOTE: list can also be a lazy i.e. and iterator
    """
    def mk_get(key_by):
        if key_by is None:
            return lambda x: x if type(x) is set else set(x)
        else:
            return lambda xs: set(key_by(x) for x in xs)

    get = mk_get(key_by)

    cc = None
    for s in sets:
        if cc is None:
            cc = get(s)
        else:
            missing = cc - get(s)
            cc = cc - missing
    return cc


def common_obs_per_cell(*tile_obs):

    def ds_time(ds): return ds.center_time

    tt_common = common_subset([o['datasets'] for o in tile_obs], ds_time)

    def pick_with(o, pred):
        o_ = o.copy()
        o_['datasets'] = [ds for ds in o['datasets'] if pred(ds)]
        return o_

    def pick_common(o): return pick_with(o, lambda ds: ds_time(ds) in tt_common)

    def pick_unmatched(o): return pick_with(o, lambda ds: ds_time(ds) not in tt_common)

    return ([pick_common(o) for o in tile_obs],
            [pick_unmatched(o) for o in tile_obs])


def multi_product_list_cells(products, gw, cell_index=None, product_query={}, **query):
    """
    products list of product names
    gw - GridWorkflow
    product_query -- dict product_name => product specific query
    """
    from functools import reduce
    from datacube.api.query import query_group_by
    from datacube.api import GridWorkflow

    empty_cell = dict(datasets=[], geobox=None)
    co_common = [dict() for _ in products]
    co_unmatched = [dict() for _ in products]

    group_by = query_group_by(**query)

    obs = [gw.cell_observations(product=product,
                                cell_index=cell_index,
                                **product_query.get(product, {}),
                                **query)
           for product in products]

    # set of all cell indexes found across all products
    all_cell_idx = set(reduce(list.__add__,
                              [list(o.keys()) for o in obs]))

    def cell_is_empty(c): return len(c['datasets']) == 0

    for cidx in all_cell_idx:
        common, unmatched = common_obs_per_cell(*[o.get(cidx, empty_cell) for o in obs])

        for i in range(len(products)):
            if cidx in obs[i]:
                if not cell_is_empty(common[i]):
                    co_common[i][cidx] = common[i]

                if not cell_is_empty(unmatched[i]):
                    co_unmatched[i][cidx] = unmatched[i]

    co_common = [GridWorkflow.group_into_cells(c, group_by=group_by) for c in co_common]
    co_unmatched = [GridWorkflow.group_into_cells(c, group_by=group_by) for c in co_unmatched]

    return co_common, co_unmatched
