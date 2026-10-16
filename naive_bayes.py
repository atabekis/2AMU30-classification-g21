import numpy as np


def class_counts(y):
    """
    count observations in each class.
    """

    y = np.asarray(y, dtype=int)

    if y.ndim != 1 or len(y) == 0 or np.any(y < 0): raise ValueError('use a nonempty vector of class codes 0, 1, ...')  # should not happen with main but my values can be stupid, jik

    return np.bincount(y).astype(float)


def check_categories(X, sizes):
    """
    ensure each feat. column contains category codes in the feat. domain
    basically a func to raise errors if we give non-allowed inputs
    """

    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(sizes): raise ValueError('X must have one column for each feature')

    if not np.isfinite(X).all() or np.any(X != X.astype(int)): raise ValueError('use integer category codes.')

    X = X.astype(int)
    if np.any(X < 0) or np.any(X >= np.asarray(sizes)): raise ValueError('a category is outside its feature domain')

    return X


def count_tables(X, y, sizes, parents):
    """
    count feat. values for each class and parent value
    """

    X = check_categories(X, sizes)
    y = np.asarray(y, dtype=int)
    
    n_classes = len(class_counts(y))
    tables = []

    for j, parent in enumerate(parents):
        parent_values = np.zeros(len(X), dtype=int) if parent == -1 else X[:, parent]
        table = np.zeros((n_classes, 1 if parent == -1 else sizes[parent], sizes[j]))

        np.add.at(table, (y, parent_values, X[:, j]), 1)
        tables.append(table)

    return tables


def probabilities(counts, strength):
    """
    est. probs. from counts, with a symmetric dirichlelt (or whatever it was) prior on the last axis

    note: a given empty table without smoothing gets the uniform dist.
    """

    # if strength < 0: raise ValueError('Prior strength cannot be negative.')
    # r = counts.shape[1]

    r = counts.shape[-1]
    denominator = counts.sum(axis=-1, keepdims=True) + strength

    return np.divide(counts + strength / r, denominator, out=np.full_like(counts, 1 / r), where=denominator > 0)


def joint_scores(X, prior, tables, parents, sizes):
    """
    prod of the class prior by the entry of each feat.
    """

    X = check_categories(X, sizes)
    scores = np.tile(prior, (len(X), 1))

    for j, parent in enumerate(parents):
        parent_values = np.zeros(len(X), dtype=int) if parent == -1 else X[:, parent]
        scores *= tables[j][:, parent_values, X[:, j]].T

    return scores


def winners(scores):
    """
    get all classes with the maximum score
    """
    return np.isclose(scores, scores.max(axis=1, keepdims=True), rtol=1e-12, atol=1e-15) # the tols are needed, trust me


class NaiveBayes: 
    def __init__(self, strength=0):
        self.strength = strength # WE CONTROL THIS FOR 0 AND 1

    def fit(self, X, y, sizes):
        """
        est the class prior and the cond probs of each feature
        """

        self.sizes = sizes
        self.parents = [-1] * len(sizes)

        self.prior = probabilities(class_counts(y), self.strength)
        self.tables = [probabilities(t, self.strength) for t in count_tables(X, y, sizes, self.parents)]

        return self

    def scores(self, X):
        """
        calc. the joint score of each observation and class
        """
        return joint_scores(X, self.prior, self.tables, self.parents, self.sizes)


    def predict(self, X):
        """
        pred the classes with the max joint score
        """
        return winners(self.scores(X))
