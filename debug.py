import pickle
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import sys
# direc = os.path.dirname(__file__)
# pri&
# caution: path[0] is reserved for script path (or '' in REPL)
# print(str(path)+'/ppo')
sys.path.insert(1, '/Users/faridounet/PhD/TransportersDilemma')
from a_star import A_Star
from SA_baseline import recuit
from greedy_baseline import baseline
from assignment import RemoveActionEnv, AssignmentEnv, GameEnv
import pickle
from shortcut import multi_types
import numpy as np
import matplotlib.pyplot as plt
import collections

from assignment import AssignmentGame
from shortcut import multi_types
from GameLearning import LRI, GameLearning, EXP3
from SA_baseline import recuit
from a_star import A_Star
from greedy_baseline import baseline, greedy

np.random.seed(42)

def DP(env : RemoveActionEnv, excess, log = False):
    
    rtes = np.array([
        [
            env._env.initial_routes[m, i] 
            for i in range(0, len(env._env.initial_routes[m]), 2)
        ]
        for m in range(len(env._env.initial_routes))
    ], dtype=int)


    # print(env._env.distance_matrix)
    # print(CM)
    coeff = env._env._game.emissions_KM
    # CM = np.array([
    #     env._env.distance_matrix*coeff[i]
    #     for i in range(len(coeff))
    # ]).copy()
    a = multi_types(env._env.distance_matrix, rtes, coeff, excess)


    _, r, *_, info = env.step(a)
    
    info['routes'] = env._env.initial_routes
    
    if log:
        print(env.destinations)
        print(env._env.distance_matrix)
    
    return a, r, info

def SA(env, log = False):
    
    if log:
        print(env.destinations)
        print(env._env.distance_matrix)
    T_init = 5_000
    T_limit = 1
    lamb = 0.9999
    T = 100_000

    action_SA, *_ = recuit(deepcopy(env._env), T_init, T_limit, lamb, H=T)
            # res = recuit_multiple(game, T_init = T_init, T_limit = T_limit, lamb = lamb, log=log, H=T)
    a = np.where(action_SA == 0)[0]

    # CM = np.array([
    #     env._env.distance_matrix*coeff[i]
    #     for i in range(len(coeff))
    # ]).copy()

    # env = RemoveActionEnv(game = g, saved_routes = routes, saved_dests=dests, 
    #                       obs_mode='elimination_gain', 
    #                       action_mode = 'destinations',
    #                         change_instance = True, rewards_mode='normalized_terminal', instance_id = 89)

    _, r, *_, info = env.step(a)
    
    info['routes'] = env._env.initial_routes
    if log:
        print(env.destinations)
        print(env._env.distance_matrix)
    
    return a, r, info

g = AssignmentGame(
            grid_size=12,
            max_capacity=1,
            Q = 7,
            K=2,
            emissions_KM = [.1, .3],
            costs_KM = [1, 1],
            seed=42
        )
env = RemoveActionEnv(game = deepcopy(g))
# print(env.reset())

for i in range(20):
    *_, info = env.reset()
    R = env._env.initial_routes
    D = env._env.distance_matrix
    aDP, rDP, infoDP = DP(deepcopy(env), info['excess_emission'])
    aSA, rSA, infoSA = SA(deepcopy(env))
    
    if rSA>rDP:
        print(50*'+')
        print(i)
        print('instance info :')
        print(info)
        print(10*'-')
        print('la solution DP : ', aDP +1)
        print('reward DP : ', rDP)
        print('info DP : ', infoDP)

        print('la solution SA : ', aSA +1)
        print('reward SA : ', rSA)
        print('info SA : ', infoSA)
        
        print(50*'+')
        
    _, r, *_, info = env.step([])
    print(r)
    print(info)
    
    # Premiere observation, la DP tient compte des emissions totalement. Or les emissions comptent uniquement jusqu'au respect du quota.
    # Par ex, enlever un paquet dans un camion hybrid qui parcours bcp de distance est plus avantageux par rapport au diesel
    
    