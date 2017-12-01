import math
import pandas as pd
from itertools import islice

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


_TREE_BUILDERS = dict(single=_single_linkage,
    complete=_complete_linkage,
    average=_average_linkage,
    average_group=_average_group_linkage)


class Agglomerative:
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
        # Initialize data to X
        self.data = X
        # Calculate distance matrix
        self.distance_matrix = []
        self.init_distance_matrix()



        tree_builder = _TREE_BUILDERS[self.linkage]

    def init_distance_matrix(self):
        """ Buat distance matrix antar instans data pada dataset.

        Bentuk distance matrix : [[dist_01, ... , dist_0n], [dist_12, ... , dist_1n],...,[dist_{n-k}{n-k+1},...dist_{n_k}{n}],...[dist_{n-1}{n}]
        dimana 0 < k <= n """
        raw_data = self.data.as_matrix()
        for index, eval_instance in enumerate(raw_data):
            distance_array = []
            for pair_instance in raw_data[index+1:]:
                distance = self.calculate_distance(eval_instance, pair_instance)
                distance_array.append(distance)
            if(distance_array):
                self.distance_matrix.append(distance_array)

    def calculate_distance(self, instance1, instance2):
        """ Hitung jarak antara instance1 dengan instance2

        Instance1 & instance2 diasumsikan ada di satu dataset yang sama.
        Jarak dihitung dengan metode euclidean : sqrt(sum((atr_instance1 - atr_instance2)^2)) """
        distance = 0
        for index, val in enumerate(instance1):
            attr_distance = (val -  instance2[index])**2
            distance += attr_distance
        return math.sqrt(distance) 
        
if __name__ ==  "__main__" : 
    #import pandas as pd
    test_data = {'A' : [1, 3, 2], 'B' : [2, 4, -1]}
    test_dataframe = pd.DataFrame(test_data)
    test_agglomerative = Agglomerative()
    test_agglomerative.fit(test_dataframe)