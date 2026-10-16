import numpy as np

from naive_bayes import check_categories, class_counts, count_tables, winners


class NaivePossibilistic:  # this paper will be the end of me
    def fit(self, X, y, sizes):
        """
        esetimate possibility of each feature.
        """
        self.sizes = sizes
        self.n_classes = len(class_counts(y))
        self.tables = []

        counted_tables = count_tables(X, y, sizes, [-1] * len(sizes)) 
        for counts in counted_tables:
            counts = counts[:, 0, :]
            matching = counts.sum(axis=0, keepdims=True)
            means = np.divide(counts, matching, out=np.zeros_like(counts), where=matching > 0)

            maximum = means.max(axis=1, keepdims=True)
            self.tables.append(np.divide(means, maximum, out=np.zeros_like(means), where=maximum > 0))

        return self

    def scores(self, X):
        """
        prod of the pos. of each feature value ofc for each class
        """
        X = check_categories(X, self.sizes)
        result = np.ones((len(X), self.n_classes))

        for j, table in enumerate(self.tables):
            result *= table[:, X[:, j]].T  # uhh

        return result

    def predict(self, X):
        return winners(self.scores(X))
