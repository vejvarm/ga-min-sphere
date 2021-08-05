from matplotlib import pyplot as plt

import ga
from structures import Problem, Params

if __name__ == '__main__':
    # Run GA
    out = ga.run(Problem, Params)

    # Results
    plt.semilogy(out.bestcost)
    plt.xlim(0, Params.maxit)
    plt.xlabel("Iterations")
    plt.ylabel("Best cost")
    plt.title("Best costs over GA iterations")

    plt.show()