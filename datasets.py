import numpy as np


# cond. probabilities of 00, 01, 10, 11
# we generate 2 binary features A, B, where here row 0 is A
INDEPENDENT = np.array([[.49, .21, .21, .09], [.09, .21, .21, .49]])
DEPENDENT = np.array([[.68, .02, .02, .28], [.02, .28, .28, .42]])


def sample(table, n, rng):
    """
    sample observations with two equal probable classes and two numeric features in [0, 1).
    """
    y = rng.integers(0, 2, n)
    cell = np.empty(n, dtype=int)

    for c in range(2):
        selected = y == c
        cell[selected] = rng.choice(4, selected.sum(), p=table[c])

    # uniform position in the half interval of each bit adds no class information (we dont want leak lol)
    bits = np.column_stack((cell // 2, cell % 2))
    X = (bits + rng.random((n, 2))) / 2

    posterior = table[:, cell].T / table[:, cell].sum(axis=0)[:, None] # equal class priors and the equal cell areas cancel in bayes rule

    return X, y, posterior


def memberships(X, centers):
    """
    membership grades of linear shoulder and triangle terms -- the grades of each value sum to one
    """
    X = np.asarray(X, dtype=float)
    return [np.column_stack([np.interp(X[:, j], c, row) for row in np.eye(len(c))]) for j, c in enumerate(centers)]


def hard_categories(grades):
    """
    select term with maximum membership for each feature -- if tie, select higher term
    """
    return np.column_stack([g.shape[1] - 1 - g[:, ::-1].argmax(axis=1) for g in grades])  #sorry very ugly code
