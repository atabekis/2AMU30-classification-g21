import numpy as np

from naive_bayes import class_counts, count_tables, joint_scores


def idm_bounds(counts, strength):
    """
    Imprecise Dirichlet model (IDM) given axis and s.
    """

    denominator = counts.sum(axis=-1, keepdims=True) + strength
    return counts / denominator, (counts + strength) / denominator


def undominated(lower, upper):
    """
    find the classes that no other class dominates (as the name suggests lmao)
    how: if class a dominates class b when the lower join score of a is larger than the upper joint score of b
    """
    
    lo, hi = lower[:, :, None], upper[:, None, :]
    dominates = (lo > hi) & ~np.isclose(lo, hi, rtol=1e-12, atol=1e-15)

    return ~dominates.any(axis=1)


class NaiveCredal:
    def __init__(self, strength=1):
        self.strength = strength  # we keep this at 1 at all times but good to include

    def fit(self, X, y, sizes):

        self.sizes = sizes
        self.parents = [-1] * len(sizes)

        self.prior_lower, self.prior_upper = idm_bounds(class_counts(y), self.strength) # get the triangle

        bounds = [idm_bounds(t, self.strength) for t in count_tables(X, y, sizes, self.parents)]
        self.lower_tables, self.upper_tables = zip(*bounds)

        return self

    def joint_bounds(self, X):
        """
        get/calc the L U joint score of each observation and class
        """

        lower = joint_scores(X, self.prior_lower, self.lower_tables, self.parents, self.sizes)
        upper = joint_scores(X, self.prior_upper, self.upper_tables, self.parents, self.sizes)

        return lower, upper

    def predict(self, X):
        """
        find the undominated class(es)
        """
        return undominated(*self.joint_bounds(X))
