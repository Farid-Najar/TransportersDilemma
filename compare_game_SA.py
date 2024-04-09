import sys

# direc = os.path.dirname(__file__)
# pri&
# caution: path[0] is reserved for script path (or '' in REPL)
# print(str(path)+'/ppo')
sys.path.insert(1, '/Users/faridounet/PhD/TransportersDilemma')

from shortcut import multi_types
from GameLearning import LRI, GameLearning, EXP3
from SA_baseline import recuit
from a_star import A_Star
from greedy_baseline import baseline, greedy
import multiprocess as mp
import numpy as np
import matplotlib.pyplot as plt
import pickle

from copy import deepcopy
from time import time

from assignment import RemoveActionEnv

def EHEG(env, obs, *args, **kwargs):
    """Eliminates the package with the highest elimination gain

    Parameters
    ----------
    env :
    """
    
    # eenv = deepcopy(env)
    
    # this_env = RemoveActionEnv(game = eenv._env._game, saved_routes = eenv._env.saved_routes, saved_dests=eenv._env.saved_dests, 
    #           obs_mode='elimination_gain', 
    #         action_mode = 'destinations',
    #           change_instance = False, rewards_mode='normalized_terminal', instance_id = i)
    # obs, _ = env.reset()
    while True:
      obs, r, d, *_ = env.step(np.argmax(obs))
      if d:
        break
      
    return r

def compare(
    n_simulation = 1,
    # strategy = LRI,
    T = 100_000,
    # Q = 30,
    K = 50,
    T_init = 5_000,
    T_limit = 1,
    lamb = 0.9999,
    log = False,
    packages = None,
    ):

    def process_DP(env, id, q, excess):
        t0 = time()
        res = dict()
        rtes = np.array([
            [
                env._env.initial_routes[m, i] 
                for i in range(0, len(env._env.initial_routes[m]), 2)
            ]
            for m in range(len(env._env.initial_routes))
        ], dtype=int)
        # print(CM)
        coeff = env._env._game.emissions_KM
        # CM = np.array([
        #     env._env.distance_matrix*coeff[i]
        #     for i in range(len(coeff))
        # ]).copy()
        a = multi_types(env._env.distance_matrix, rtes, coeff, excess)
        # a = dests[i_id][np.where(a_GTS == 0)].astype(int)
        _, r_opt, *_ = env.step(a)
        # res = GameLearning(game, T=T, strategy=strategy, log = log)
        res['time'] = time() - t0
        res['r'] = r_opt
        q.put((id, res))
        print(f'DP {id} done')
        # res_dict = d
        
    def process_multiple_SA(env, id, q):
        t0 = time()
        res = dict()
        action_SA, *_ = recuit(env._env, T_init, T_limit, lamb, H=T)
        # res = recuit_multiple(game, T_init = T_init, T_limit = T_limit, lamb = lamb, log=log, H=T)
        _, r_SA, *_ = env.step(np.where(action_SA == 0)[0])
        res['time'] = time() - t0
        res['r'] = r_SA
        q.put((id, res))
        print(f'SA {id} done')
        
    def process_A(env, id, q):
        t0 = time()
        res = dict()
        res_A = A_Star(env._env, max_time=20)
        # res = baseline(game)
        action_A = res_A['solution'].astype(int)
        _, r_A, *_ = env.step(np.where(action_A == 0)[0])
        res['time'] = time() - t0
        res['r'] = r_A
        q.put((id, res))
        print(f'baseline {id} done')
        
    def process_greedy(env, id, q, obs):
        t0 = time()
        res = dict()
        r_EG = EHEG(env, obs)
        # res = greedy(game)
        res['time'] = time() - t0
        res['r'] = r_EG
        q.put((id, res))
        print(f'greedy {id} done')
        
    q_SA = mp.Manager().Queue()
    q_DP = mp.Manager().Queue()
    q_baseline = mp.Manager().Queue()
    q_greedy = mp.Manager().Queue()
    
    res_DP = dict()
    res_SA = dict()
    res_baseline = dict()
    res_greedy = dict()
    
    ps = []
    
    with open(f'TransportersDilemma/RL/game_K{K}.pkl', 'rb') as f:
        g = pickle.load(f)
    routes = np.load(f'TransportersDilemma/RL/routes_K{K}.npy')
    dests = np.load(f'TransportersDilemma/RL/destinations_K{K}.npy')

    env = RemoveActionEnv(game = g, saved_routes = routes, saved_dests=dests, 
                obs_mode='elimination_gain', 
                action_mode = 'destinations',
                  change_instance = True, rewards_mode='normalized_terminal', instance_id = 0)

    for i in range(n_simulation):
        # game = AssignmentGame(Q=Q, K = K)
        obs, info= env.reset()
        # threads.append(Thread(target = process, args = (game, res[i])))
        ps.append(mp.Process(target = process_DP, args = (deepcopy(env), i, q_DP, info['excess_emission'], )))
        ps.append(mp.Process(target = process_multiple_SA, args = (deepcopy(env), i, q_SA,)))
        ps.append(mp.Process(target = process_A, args = (deepcopy(env), i, q_baseline,)))
        ps.append(mp.Process(target = process_greedy, args = (deepcopy(env), i, q_greedy,obs,)))
        ps[4*i].start()
        ps[4*i+1].start()
        ps[4*i+2].start()
        ps[4*i+3].start()
        
    for i in range(n_simulation):
        ps[4*i].join()
        ps[4*i+1].join()
        ps[4*i+2].join()
        ps[4*i+3].join()
        
    print('all done !')
    while not q_DP.empty():
        i, d = q_DP.get()
        res_DP[i] = d
        
    while not q_SA.empty():
        i, d = q_SA.get()
        res_SA[i] = d
        
    while not q_baseline.empty():
        i, d = q_baseline.get()
        res_baseline[i] = d
    
    while not q_greedy.empty():
        i, d = q_greedy.get()
        res_greedy[i] = d
    
    res = {
        'res_DP' : res_DP,
        'res_SA' : res_SA,
        'res_baseline' : res_baseline,
        'res_greedy' : res_greedy,
    }
    
    # with open(f"res_compare_baseline_greedy_SA_{strategy.__name__}_Q{Q}_K{K}_n{n_simulation}_T{T}.pkl","wb") as f:
    with open(f"res_compare_EG_A*_SA_K{K}_n{n_simulation}_T{T}.pkl","wb") as f:
        pickle.dump(res, f)
    
    r_DP = np.array([
        res_DP[i]['r']
        for i in res_DP.keys()
    ])
    
    r_baseline = np.array([
        res_baseline[i]['r']
        for i in res_baseline.keys()
    ])
    
    r_greedy = np.array([
        res_greedy[i]['r']
        for i in res_greedy.keys()
    ])
    
    r_SA = np.array([
        res_SA[i]['r']
        for i in res_SA.keys()
    ])
    # np.amin([[
    #     res_SA[i][j]['list_best_costs']
    #     for j in res_SA[i].keys() if j != 'time'
    # ] for i in res_SA.keys()],
    # axis=1)
    
    # print(costs_game.shape)
    # print(costs_SA.shape)
    
    rs = [
        r_DP, # np.amin(costs_game, axis=1),
        r_SA,#np.amin(r_SA, axis=1),
        r_baseline,#np.amin(r_baseline, axis=1),
        r_greedy,#np.amin(r_greedy, axis=1),
    ]
    
    labels = [
        'DP',# strategy.__name__,
        'SA',
        'A*',
        'EHEG'
    ]
    
    plt.boxplot(
        rs,
        labels=labels,
        patch_artist=True)
    plt.title(f'Costs for different methods : K={K}, n={n_simulation}')
    # r_min = np.amin(rewards_game, axis=0)
    # r_max = np.amax(rewards_game, axis=0)
    # r_mean = np.mean(rewards_game, axis=0)
    # std = np.std(rewards_game, axis=0)
    # r_median = np.median(rewards_game, axis=0)

    # # fig, ax = plt.subplots(2, 1)
    # plt.plot(r_min, linestyle=':', label='min rewards', color='black')
    # plt.plot(r_mean, label='mean rewards')
    # plt.plot(r_median, label='median rewards', linestyle='--', color='black')
    # plt.plot(r_max, label='max rewards', linestyle='-.', color='black')
    # plt.fill_between(range(len(r_mean)), r_mean - 2*std, r_mean + 2*std, alpha=0.3, label="mean $\pm 2\sigma$")
    # plt.fill_between(range(len(r_mean)), r_mean - std, r_mean + std, alpha=0.7, label="mean $\pm \sigma$")
    # plt.title(f'Rewards in {strategy.__name__}')
    # plt.xlabel("Time $t$")
    # plt.legend()
    plt.savefig('rs')
    plt.show()
    
    times = {
        k : [res[k][i]['time'] for i in res[k].keys()]
        for k in res.keys()
    }
    
    plt.boxplot(
        times.values(),
        labels=labels,
        patch_artist=True)
    plt.title(f'Times for different methods : K={K}, n={n_simulation}')
    plt.savefig('ts')
    plt.show()
    
    # sol = res['solution']
    # print('solution : ', sol)
    


if __name__ == '__main__' :
    compare(n_simulation=500, K=100)