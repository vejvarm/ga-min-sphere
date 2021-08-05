from copy import deepcopy
from collections import namedtuple
import numpy as np

from structures import Individual


def crossover(p1, p2, gamma=0.1):
    """ Uniform crossover with overflow

    :param p1: (object) parent 1
    :param p2: (object) parent 2
    :param gamma: (float) possible overflow of alpha around 0 and 1 (allows better exploration)
    :return (c1, c2): tuple of 2 children objects
    """
    nvar = p1.position.shape
    alpha = np.random.uniform(-gamma, 1+gamma, size=nvar)

    c1 = deepcopy(p1)
    c2 = deepcopy(p2)

    c1.update(alpha*p1.position + (1-alpha)*p2.position)
    c2.update(alpha*p2.position + (1-alpha)*p1.position)

    return c1, c2


def mutate(x, mu, sigma):
    """ mutation with rate mu and step size sigma

    :param x: object to mutate
    :param mu: mutation rate
    :param sigma: mutation step size (distribution standard deviation)
    :return: mutated object
    """
    y = deepcopy(x)

    mask = np.random.rand(*y.position.shape) <= mu  # binary mask for mutation selection

    y.update(x.position + mask*np.random.normal(0, sigma, x.position.shape))

    return y


def calc_cummulative_probs(pop, beta=1.):
    costs = np.array([p.cost for p in pop])
    avg_costs = costs.mean()

    if avg_costs == 0:
        avg_costs = 1.

    probs = np.exp(-beta * costs / avg_costs)
    cum_probs = np.cumsum(probs)/sum(probs)

    return cum_probs


def roulette_wheel(cum_probs):
    rng = np.random.rand()*cum_probs[-1]
    selection = np.argwhere(rng <= cum_probs)[0, 0]
    return selection


def run(problem, params):

    # Problem
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    # Params
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    pc = params.pc
    nc = int(pc*npop/2)*2  # must be an even number (ensured)
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma

    # Best solution ever found
    bestsol = Individual(np.random.uniform(varmin, varmax, nvar), costfunc, cost=np.inf)

    # Initialization
    pop = []
    for _ in range(npop):
        pop.append(Individual(np.random.uniform(varmin, varmax, nvar), costfunc))
        if pop[-1].cost < bestsol.cost:
            bestsol = deepcopy(pop[-1])

    # Best cost of iterations
    bestcost = np.empty(maxit)

    # Main loop
    for i in range(maxit):

        # Create population of children
        popc = []
        cum_probs = calc_cummulative_probs(pop, beta)  # roulette wheel selection probabilities for current loop
        for _ in range(nc//2):
            # Parent selection
            p1_idx = roulette_wheel(cum_probs)
            p2_idx = roulette_wheel(np.concatenate([cum_probs[:p1_idx], cum_probs[p1_idx+1:]]))
            p1 = pop[p1_idx]
            p2 = pop[p2_idx]
            # p1, p2 = np.random.choice(pop, size=2, replace=False)  # random selection without replacement

            # Crossover
            c1, c2 = crossover(p1, p2, gamma)

            # Mutate and evaluate
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)

            # Update bestsol if costs of children are better
            if c1.cost < bestsol.cost:
                bestsol = deepcopy(c1)
            if c2.cost < bestsol.cost:
                bestsol = deepcopy(c2)

            # Add children to popc
            popc.append(c1)
            popc.append(c2)

        # Merge offsprings and parents
        pop += popc

        # Sort and select top npop
        pop.sort(key=lambda x: x.cost)
        pop = pop[:npop]

        # Store best cost
        bestcost[i] = bestsol.cost

        # Print out results of iteration
        print(f"Iteration {i}: Best Cost = {bestcost[i]}")

    # Output
    out = namedtuple("out", ["pop", "bestsol", "bestcost"])
    out.pop = pop
    out.bestsol = bestsol
    out.bestcost = bestcost

    return out
