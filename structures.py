import numpy as np


# Sphere Function
def sphere(x):
    return sum(x**2)


class Problem:
    costfunc = sphere
    nvar = 5
    varmin = [-10, -10, -10, -5, -2]
    varmax = [ 10,  10,  10,  5,  2]


class Params:
    maxit = 100
    npop = 50
    beta = 1.    # probability pressure for roulette wheel parent selection
    pc = 1.      # ratio of children from parents (if 1. there is same number of children as parents)
    gamma = 0.1  # (during crossover) possible overflow of alpha around 0 and 1 (allows better exploration)
    mu = 0.1    # mutation ratio
    sigma = 0.1  # mutation step size


class Individual:

    def __init__(self, position, costfunc, cost=None):
        self.position = self._ensure_bounds(position)
        self._costfunc = costfunc

        if cost:
            self.cost = cost
        else:
            self.cost = self._costfunc(position)

    @staticmethod
    def _ensure_bounds(position):
        position = np.maximum(position, Problem.varmin)
        position = np.minimum(position, Problem.varmax)
        return position

    def update(self, position):
        self.position = self._ensure_bounds(position)
        self.cost = self._costfunc(position)
