# might need to review this file as i used some bit of ai to make the implementation

import numpy as np

from naive_bayes import check_categories, class_counts, count_tables, joint_scores, probabilities, winners


def learn_tree(X, y, sizes):
    """
    we learn a tree here... gien the conditional mutual information between nodes/pairs

    constructs max-weight spanning tree- feature 0 as the root.
    """

    X = check_categories(X, sizes)
    y = np.asarray(y, dtype=int)

    n_features, n_classes = len(sizes), len(class_counts(y))
    weights = np.zeros((n_features, n_features))

    # edge weight: conditional mutual information (i made ai translate the paper's formula here so i cannot really explain the np parts)
    for i in range(n_features):
        for j in range(i):
            table = np.zeros((n_classes, sizes[i], sizes[j]))
            np.add.at(table, (y, X[:, i], X[:, j]), 1)

            for c in range(n_classes):
                count = table[c].sum()

                if count == 0: continue

                joint = table[c] / count
                independent = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
                positive = joint > 0

                information = np.sum(joint[positive] * np.log(joint[positive] / independent[positive]))
                weights[i, j] += count / len(y) * information

            weights[j, i] = weights[i, j]

    # prim's algorithm
    parents = [-1] * n_features
    visited = {0}

    while len(visited) < n_features:
        edges = [(-weights[i, j], i, j) for i in sorted(visited) for j in range(n_features) if j not in visited]
        _, parent, child = min(edges)

        parents[child] = parent
        visited.add(child)

    return parents


class TAN:
    def __init__(self, strength=1):
        self.strength = strength # again, only s=1 used in our case, but nice to have

    def fit(self, X, y, sizes):
        """
        learn feature tree then estimate the class prior and the cond. prob tables
        """
        self.sizes = sizes
        self.parents = learn_tree(X, y, sizes)

        self.prior = probabilities(class_counts(y), self.strength)
        self.tables = [probabilities(t, self.strength) for t in count_tables(X, y, sizes, self.parents)]

        return self

    def scores(self, X):
        return joint_scores(X, self.prior, self.tables, self.parents, self.sizes)

    def predict(self, X):
        return winners(self.scores(X))
