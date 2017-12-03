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
    tree_ = ''
    labels_ = []
    average_points = []
    clusters_members = []

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
        # Treat every instance as single cluster
        self.init_cluster()
        # Calculate distance matrix
        self.distance_matrix = []
        self.init_distance_matrix()

        while len(self.cluster) > self.n_clusters :
            # Find index of "minimum" value in distance matrix
            idxmin = self.distance_matrix_idxmin()
            # Cluster two instance based on "minimum" value
            # Kluster disimpan dalam bentuk array. Contoh bentuk cluster: [[1], [[2], [3]], [[4], [[5], [6]], [[7], [8]]], [9]]
            # key digunakan untuk distance matrix, misalnya
            #
            #                                     [1]  |  [[2],[3]]  |   [4,[[5],[6]],[[7],[8]]]  |   [9]
            # 0  [1]                          |            1.2                      1.4               3.4
            # 1  [[2], [3]]                   |                                     7.6               2.16
            # 2  [[4], [[5],[6]], [[7],[8]]]  |                                                       1.1
            #    [9]
            # print(idxmin)
            self.update_cluster(idxmin[0], idxmin[1])
            # Update distance matrix
            self.distance_matrix_update(idxmin[0], idxmin[1])
            print(len(self.cluster))



        tree_builder = _TREE_BUILDERS[self.linkage]
        self.generate_label()

    def init_cluster(self):
        """Inisialisasi kluster.

        Kluster diinisialisasi dengan index semua instans data"""
        raw_data = self.data.as_matrix()
        self.cluster = [[x] for x, val in enumerate(raw_data)]

    def init_distance_matrix(self):
        """ Buat distance matrix antar instans data pada dataset.

        Bentuk distance matrix : [[dist_01, ... , dist_0n], [dist_12, ... , dist_1n],...,[dist_{n-k}{n-k+1},...dist_{n_k}{n}],...[dist_{n-1}{n}]
        dimana 0 < k <= n (Matriks segitiga atas, dengan diagonal dihilangkan)"""
        raw_data = self.data.as_matrix()
        for index, eval_instance in enumerate(raw_data):
            if (self.linkage == "average"):
                self.clusters_members.insert(index, self.cluster[index])
            if (self.linkage == "average_group"):
                self.average_points.append(eval_instance)

            distance_array = []
            for pair_instance in raw_data[index + 1:]:
                distance = self.calculate_distance(eval_instance, pair_instance)
                distance_array.append(distance)
            if (distance_array):
                self.distance_matrix.append(distance_array)

    def calculate_distance(self, instance1, instance2):
        """ Hitung jarak antara instance1 dengan instance2

        Instance1 & instance2 diasumsikan ada di satu dataset yang sama.
        Jarak dihitung dengan metode euclidean : sqrt(sum((atr_instance1 - atr_instance2)^2)) """
        distance = 0
        for index, val in enumerate(instance1):
            attr_distance = (val - instance2[index]) ** 2
            distance += attr_distance

        return math.sqrt(distance)

    def distance_matrix_idxmin(self):
        """ Cari index [i, j] dimana self.distance_matrix[i][j] minimum.

        Untuk saat ini masih memakai metrik nilai minimum (single_linkage)
        """
        min_val = self.distance_matrix[0][0]
        min_idx = [0, 0]
        for i, val_i in enumerate(self.distance_matrix):
            for j, val_j in enumerate(val_i):
                if min_val > val_j:
                    min_val = val_j
                    min_idx = [i, j]
        min_idx[1] = min_idx[0] + j + 1
        # print(min_idx)
        return min_idx

    def get_all_cluster_member(self, cluster):
        if len(cluster) == 1:
            return cluster
        else:
            member = []
            for subcluster in cluster:
                member = member+self.get_all_cluster_member(subcluster)
            return member;

    def calculate_distance_average(self, cluster1, cluster2):
        distance_sum = 0;
        raw_data = self.data.as_matrix()
        for member1 in cluster1:
            data1 = raw_data[member1]
            for member2 in cluster2:
                data2 = raw_data[member2]
                distance_sum  += self.calculate_distance(data1, data2)

        return distance_sum/(len(cluster1)*len(cluster2))

    def get_average_point(self, cluster):
        attr = []
        raw_data = self.data.as_matrix()
        for i in range(0, len(raw_data[0])):
            attr.insert(i,0)
            for member in cluster:
                attr[i] += raw_data[member][i]
            attr[i] /= len(cluster)

        return attr

    def update_distance_average_group(self, newClusterIdx, delIdx):
        #initialize cluster member
        members = self.get_all_cluster_member(self.cluster[newClusterIdx])
        del self.distance_matrix[:]
        self.average_points[newClusterIdx] = self.get_average_point(members)
        for i in range(0, len(self.cluster)-1):
            distance_array = []
            for j in range(i+1, len(self.cluster)):
                distance = self.calculate_distance(self.average_points[i], self.average_points[j])
                distance_array.append(distance)
            self.distance_matrix.append(distance_array)



    def update_distance_average(self, newClusterIdx,delIdx) :
        del self.clusters_members[delIdx]

        self.clusters_members[newClusterIdx] = self.get_all_cluster_member(self.cluster[newClusterIdx])
        del self.distance_matrix[:]
        for i in range(0, len(self.cluster)-1):
            distance_array = []
            for j in range(i+1, len(self.cluster)):
                distance = self.calculate_distance_average(self.clusters_members[i], self.clusters_members[j])
                distance_array.append(distance)
            self.distance_matrix.append(distance_array)


    def distance_matrix_update(self, instance1, instance2):
        """ Update self.distance_matrix

        Pastikan instance1 < instance2
        Instance1 dan instance2 adalah dua instans/kluster yang baru saja diklusterkan.

        Untuk saat ini metrik yang baru diimplementasikan adalah nilai minimum (single_linkage).
        """
        self.distance_matrix.append([])
        coordinate_to_delete = []
        del self.clusters_members[instance2]
        if (self.linkage == "average"):
            self.clusters_members[instance1] = self.get_all_cluster_member(self.cluster[instance1])
        if(self.linkage == "average_group"):
            members = self.get_all_cluster_member(self.cluster[instance1])
            self.average_points[instance1] = self.get_average_point(members)
        for index, val in enumerate(self.distance_matrix):
            if index != instance1 and index != instance2:
                coordinate = self.transform_matrix_coordinate(index, instance1)
                coordinate_compare = self.transform_matrix_coordinate(index, instance2)
                cell_x = coordinate[0]
                cell_y = coordinate[1]
                cell_x_compare = coordinate_compare[0]
                cell_y_compare = coordinate_compare[1]
                # Perhitungan Single Linkage-nya ada di sini
                if (self.linkage == "complete"):
                    val_update = max(self.distance_matrix[cell_x][cell_y],
                                    self.distance_matrix[cell_x_compare][cell_y_compare])
                elif(self.linkage == "average"):
                    # print(cell_x)
                    # print(cell_y)
                    val_update = self.calculate_distance_average(self.clusters_members[cell_x], self.clusters_members[cell_y])
                elif(self.linkage == "average_group"):
                    val_update = distance = self.calculate_distance(self.average_points[cell_x], self.average_points[cell_y])
                else :
                    val_update = min(self.distance_matrix[cell_x][cell_y],
                                    self.distance_matrix[cell_x_compare][cell_y_compare])
                self.distance_matrix[cell_x][cell_y] = val_update
                self.distance_matrix[cell_x_compare][cell_y_compare] = 0
                coordinate_to_delete.append(coordinate_compare)
        coord_to_del = self.transform_matrix_coordinate(instance1, instance2)
        coordinate_to_delete.append(coord_to_del)
        # Delete all 0-valued cells
        for index, val in enumerate(coordinate_to_delete):
            del self.distance_matrix[val[0]][val[1]]
            for j, next_vals in enumerate(coordinate_to_delete[index + 1:]):
                if next_vals[0] == val[0]:
                    coordinate_to_delete[index + 1 + j][1] -= 1
        # Delete all empty cluster
        cluster_length = len(self.distance_matrix)
        cell_row_idx = 0
        while cell_row_idx < cluster_length:
            if not self.distance_matrix[cell_row_idx]:
                del self.distance_matrix[cell_row_idx]
                cell_row_idx -= 1
                cluster_length -= 1
            cell_row_idx += 1

    def transform_matrix_coordinate(self, cell_x, cell_y):
        coordinate = [min(cell_x, cell_y), max(cell_x, cell_y)]
        coordinate[1] = coordinate[1] - (coordinate[0] + 1)
        # print(coordinate)
        return coordinate

    def update_cluster(self, index_instance1, index_instance2):
        self.cluster[index_instance1] = [self.cluster[index_instance1], self.cluster[index_instance2]]
        del self.cluster[index_instance2]

    def find_value_in_cluster(self, subcluster, value):
        if len(subcluster) == 1:
            if subcluster[0] == value:
                return True
            else:
                return False
        else:
            for cluster_val in subcluster:
                if self.find_value_in_cluster(cluster_val, value):
                    return True
            return False

    def set_label(self, component_tree=None, label=None):
        if not isinstance(component_tree, list):
            self.labels_.insert(component_tree, label)
        else:
            for component in component_tree:
                self.set_label(component, label)

    def generate_label(self):
        nol = self.cluster[:1]
        satu = self.cluster[1:]
        self.set_label(nol, 0)
        self.set_label(satu, 1)


if __name__ == "__main__":
    # import pandas as pd
    test_data = {'A': [1, 2, -2, -3, 4, 5, 6, 7], 'B': [1, 0, 1, 1, 4, 3, 6, 6]}
    test_dataframe = pd.DataFrame(test_data)
    test_agglomerative = Agglomerative()
    test_agglomerative.fit(test_dataframe)
