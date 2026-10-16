import numpy as np

from datasets import hard_categories
from naive_bayes import class_counts, probabilities, winners


class FuzzyZhengTang:
    def __init__(self, strength=0):
        self.strength = strength

    def fit(self, grades, y):
        """
        fit/estimate class prior and the graded cond. prob. of each feature
        """
        y = np.asarray(y, dtype=int)
        counts = class_counts(y)

        self.prior = probabilities(counts, self.strength)
        self.tables = []

        for g in grades:
            mass = np.array([g[y == c].sum(axis=0) for c in range(len(counts))])
            self.tables.append(probabilities(mass, self.strength))

        return self

    def scores(self, grades):
        """
        get joint score of each observation and class
        """

        X = hard_categories(grades)
        result = np.tile(self.prior, (len(X), 1))

        for j, table in enumerate(self.tables):
            result *= table[:, X[:, j]].T

        return result

    def predict(self, grades):
        """
        get/find classes with the max joint score
        """
        return winners(self.scores(grades))  # winners is from NB
