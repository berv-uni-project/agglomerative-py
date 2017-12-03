def linkage_tree(X, n_clusters=None, linkage='single', return_distance=False):
    return True


def _single_linkage(*args, **kwargs):
    kwargs['linkage'] = 'single'
    return linkage_tree(*args, **kwargs)


def _complete_linkage(*args, **kwargs):
    kwargs['linkage'] = 'complete'
    return linkage_tree(*args, **kwargs)


def _average_linkage(*args, **kwargs):
    kwargs['linkage'] = 'average'
    return linkage_tree(*args, **kwargs)


def _average_group_linkage(*args, **kwargs):
    kwargs['linkage'] = 'average_group'
    return linkage_tree(*args, **kwargs)


_TREE_BUILDERS = dict(
    single=_single_linkage,
    complete=_complete_linkage,
    average=_average_linkage,
    average_group=_average_group_linkage)


class Agglomerative:
    _tree = ''
    _labels = []

    def __init__(self, n_clusters=2, linkage='single'):
        self.linkage = linkage
        self.n_clusters = n_clusters

    def fit(self, X):
        if self.n_clusters <= 0:
            raise ValueError("n_clusters should be an integer greater than 0."
                             " %s was provided." % str(self.n_clusters))

        if self.linkage not in _TREE_BUILDERS:
            raise ValueError("Unknown linkage type %s."
                             "Valid options are %s" % (self.linkage,
                                                       _TREE_BUILDERS.keys()))
        tree_builder = _TREE_BUILDERS[self.linkage]
