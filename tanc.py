from naive_bayes import class_counts, count_tables, joint_scores
from naive_credal import idm_bounds, undominated
from tan import learn_tree


class TANC:
    def __init__(self, strength=1):
        self.strength = strength

    def fit(self, X, y, sizes):
        """
        learn feature tree (same as tan) then estimate the L and U prob tables -- idm func from NC
        """
        self.sizes = sizes
        self.parents = learn_tree(X, y, sizes)
        
        self.prior_lower, self.prior_upper = idm_bounds(class_counts(y), self.strength)

        bounds = [idm_bounds(t, self.strength) for t in count_tables(X, y, sizes, self.parents)]
        self.lower_tables, self.upper_tables = zip(*bounds)

        return self

    def joint_bounds(self, X):
        """
        calc the L and U joint score of each observation and class
        """
        lower = joint_scores(X, self.prior_lower, self.lower_tables, self.parents, self.sizes)
        upper = joint_scores(X, self.prior_upper, self.upper_tables, self.parents, self.sizes)

        return lower, upper

    def predict(self, X):
        return undominated(*self.joint_bounds(X))
