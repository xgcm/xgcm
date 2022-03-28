import itertools


def iterate_axis_combinations(items):
    """A function to find the right combination of metrics


    Parameters
    ----------
    items : list
        List of axes needed for calculations to be performed on data array

    Yields
    -------
    items_set : frozenset
        Frozen set of axes for initial double checking with metric dimensions
    """
    items_set = frozenset(items)
    yield (items_set,)
    N = len(items)
    for nleft in range(N - 1, 0, -1):
        nright = N - nleft
        for sub_loop, sub_items in itertools.product(
            range(min(nright, nleft), 0, -1),
            itertools.combinations(items_set, nleft),
        ):
            these = frozenset(sub_items)
            those = items_set - these
            others = [frozenset(i) for i in itertools.combinations(those, sub_loop)]
            yield (these,) + tuple(others)
