import numpy as np
from assignment import AssignmentGame
import pickle
from time import time
import multiprocess as mp
from copy import deepcopy

def mcts(
    game : AssignmentGame,
    time_budget = 1,
    full_OR = False,
    OR_every = 1,
    ):
    # vehicles_by_priority = np.argsort(game.emissions_KM)
    
    action = np.ones(game.num_packages, dtype=bool)
    rewards = np.zeros(game.num_packages)
    excess_emission = np.zeros(game.num_packages)
    infos = []#[dict() for _ in range(len(action))]
    # omitted = np.zeros(game.num_packages)
    
    best = action.copy()
    solution = best.copy()
    
    # l = [game.hub] + nodes
    # x, y = np.ix_(l, l)
    
    # infos = []
    
    # A = game.distance_matrix[x, y]@np.diag(1/q)
    # indices = np.flip(np.argsort(np.mean(A[1:, 1:] + np.max(A[1:, 1:])*np.eye(len(A[1:, 1:])), axis=1)))
    r_best, _, info = game.step(action.astype(int), time_budget, call_OR=True)
    
    emission = info['excess_emission']
    # o = info['omitted']
    
    indices = list(range(game.num_packages))
    
    for t in range(game.num_packages):
        excess_emission[t] = emission
        # omitted[t] = o
        rewards[t] = r_best
        
        if emission <= 0:
            break
        
        r_best = float('-inf')
        
        for i in indices:
            a = action.copy()
            a[i] = not a[i]
            r, _, info = game.step(a.astype(int), time_budget, call_OR=(full_OR and t%OR_every == 0))
            
            if r > r_best:
                emission = info['excess_emission']
                # o = info['omitted']
                r_best = r
                if r > np.max(rewards[:t+1]):
                    solution = a.copy()
                best = a.copy()
                infos.append(info)
                ii = i
                
        # print(len(indices))
        indices.remove(ii)
        action = best.copy()
        a = action.copy()
                
        
    
    res = {
        'solution' : solution,
        'rewards' : rewards,
        'excess_emission' : excess_emission,
        'infos' : infos,
    }
    
    return res