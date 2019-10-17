import numpy as np

class Tree:
    def __init__(self, value):
        self.value = value
        self.j = None
        self.p = None
        self.left = None
        self.right = None
        self.entropy = None
        self.gini = None
        self.var = None


def variance(x):
    return np.sum(np.square(x - np.mean(x)))


class CART:
    def __init__(self):
        self.tree = None

    def build_regression_tree(self, X, y, min_leaf_sampes=1):
        m, n = X.shape
        feature_empty = [False] * n
        self.tree = self.build_regression_tree_iter(X, y, feature_empty, min_leaf_sampes=min_leaf_sampes)
        return self.tree

    def build_regression_tree_iter(self, X, y, feature_empty, min_leaf_sampes=1):
        m, n = X.shape
        if len(y) == 0:
            return None
        tree = Tree(np.mean(y))
        tree.var = variance(y)
        if len(y) <= min_leaf_sampes:
            return tree
        min_variances = tree.var
        min_j, min_s = 0, 0
        min_sort_idx = np.arange(len(y))
        for j in range(n):
            if feature_empty[j]:
                continue
            idx = np.argsort(X[:, j], axis=0)
            x1 = X[:, j][idx]
            y1 = y[idx]
            p = x1[0]
            is_empty = True
            for s in range(m):
                if x1[s] > p:
                    is_empty = False
                    current_var = variance(y1[:s]) + variance(y1[s:])
                    if current_var < min_variances:
                        min_variances = current_var
                        min_j, min_s = j, s
                        min_sort_idx = idx
            feature_empty[j] = is_empty
        if all(feature_empty):
            return tree
        tree.j = min_j
        tree.p = X[min_j][min_s]
        X, y = X[min_sort_idx], y[min_sort_idx]
        tree.left = self.build_regression_tree_iter(X[:s], y[:s],
                                                    feature_empty, min_leaf_sampes=min_leaf_sampes)
        tree.right = self.build_regression_tree_iter(X[s:], y[s:],
                                                     feature_empty, min_leaf_sampes=min_leaf_sampes)
        return tree
