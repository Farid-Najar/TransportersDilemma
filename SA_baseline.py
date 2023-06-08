from typing import Dict, List
import numpy as np
from assignment import AssignmentGame

#import itertools as it
import networkx as nx
from threading import Thread
from numpy import random as rd
import copy as cp
from numpy import exp

def recuit(game : AssignmentGame, T_init, T_limit, lamb = .99, var = False, id = 0, log = True) :
    """
    This function finds a solution for the steiner problem
        using annealing algorithm
    :param game: the assignment game
    :param T_init: the initial temperature
    :param T_limit: the lowest temperature allowed
    :return: the solution found and the evolution of the best evaluations
    """
    best = np.ones(game.num_packages, dtype=int)
    T = T_init
    eval_best, info = eval_annealing(best, game)
    m = 0
    list_best_costs = [eval_best]
    flag100 = True
    while(T>T_limit):
        sol = rand_neighbor(best)
        eval_sol, info = eval_annealing(sol, game)
        if m%20 == 0 and log:
            print(20*'-')
            print(m)
            print('- searcher ', id)
            print('temperature : ', T)
            print('excess_emission : ', info['excess_emission'])
            print('omitted : ', info['omitted'])
            print('cost : ', eval_sol)
            print('best cost : ', eval_best)
        if eval_sol < eval_best :
            prob = 1
        else :
            prob = exp((eval_best - eval_sol)/T)
        rand = rd.random()
        if rand <= prob :
            best = sol
            eval_best = eval_sol
        list_best_costs.append(eval_best)
        T *= lamb
        m += 1

        
        if(var and flag100 and T<=100):
            flag100 = False
            lamb = .999
        #print(T)

    print(f'm = {m}')
    print(eval_best)
    return best, list_best_costs


def recuit_multiple(games : List[AssignmentGame], T_init, T_limit = 2, nb_researchers = 2, lamb = .99, log = True):
    """
    This function finds a solution for the steiner problem
        using annealing algorithm with multiple researchers
    :param game: the assignment game
    :param nb_researchers: the number of researchers for the best solution
    :param T_init: the initial temperature
    :param T_limit: the lowest temperature allowed
    :return: the solution found which is a set of edges
    """
    
    def process(res : Dict, id):
        best, list_best_costs = recuit(games[id], T_init = T_init, T_limit = T_limit, lamb = lamb, id = id, log=log)
        res['sol'] = best
        res['list_best_costs'] = list_best_costs
    
    res = {
        i : dict()
        for i in range(nb_researchers)
    }
    
    threads = []
    for i in range(nb_researchers):
        threads.append(Thread(target = process, args = (res[i], i)))
        threads[i].start()

    for i in range(nb_researchers):
        threads[i].join()

    return res


def eval_annealing(sol, game : AssignmentGame, malus = 500):
    """
    This evaluates the solution of the algorithm.
    :param sol: the solution which is list of booleans
    :param graph: the graph for each we search a solution
    :param terms: the list of terminal nodes
    :param malus: the coefficient that we use to penalize bad solutions
    :return: the evaluation of the solution that is an integer
    """
    r, _, info = game.step(sol)
    # with open('log.txt', 'w+') as f:
    #     f.write(str(info))
    
    return -r, info

def rand_neighbor(solution : np.ndarray, nb_changes = 1) :
    """
    Generates new random solution.
    :param solution: the solution for which we search a neighbor
    :param nb_changes: maximum number of the changes alowed
    :return: returns a random neighbor for the solution
    """
    new_solution = solution.astype(bool)
    i = rd.choice(len(new_solution), nb_changes, replace=False)
    new_solution[i] = ~new_solution[i]
    return new_solution.astype(int)


if __name__ == '__main__' :
    NB = 7
    games = []
    for _ in range(NB):
        game = AssignmentGame(Q=50)
        K = 50
        game.reset(num_packages = K)
        games.append(game)
    
    res = recuit_multiple(games, 2000, 1, nb_researchers=NB, log = False)
    import pickle
    with open("res_multiple_SA.pkl","wb") as f:
        pickle.dump(res, f)
    
    bests = np.zeros(len(res))
    import matplotlib.pyplot as plt
    for i in res.keys():
        costs = res[i]['list_best_costs']
        bests[i] = costs[-1]
        plt.semilogy(costs, label=f'Searcher {i}')
    
    sol = res[np.argmax(bests)]['sol']
    print('solution : ', sol)
    plt.title('Best solution costs in multiple-SA')
    plt.legend()
    plt.show()
