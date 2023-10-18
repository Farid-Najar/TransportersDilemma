from GameLearning import LRI, GameLearning, EXP3
from SA_baseline import recuit_multiple
from greedy_baseline import baseline, greedy
import multiprocess as mp
import numpy as np
import matplotlib.pyplot as plt
import pickle

from copy import deepcopy
from time import time

from assignment import AssignmentGame

def compare(
    n_simulation = 1,
    strategy = LRI,
    T = 500,
    Q = 30,
    K = 50,
    T_init = 2_000,
    T_limit = 1,
    lamb = .99,
    log = False,
    packages = None,
    ):

    def process_game(game, id, q):
        t0 = time()
        res = GameLearning(game, T=T, strategy=strategy, log = log)
        res['time'] = time() - t0
        q.put((id, res))
        print(f'game {id} done')
        # res_dict = d
        
    def process_multiple_SA(game, id, q):
        t0 = time()
        res = recuit_multiple(game, T_init = T_init, T_limit = T_limit, lamb = lamb, log=log, H=T)
        res['time'] = time() - t0
        q.put((id, res))
        print(f'SA {id} done')
        
    def process_baseline(game, id, q):
        t0 = time()
        res = baseline(game)
        res['time'] = time() - t0
        q.put((id, res))
        print(f'baseline {id} done')
        
    def process_greedy(game, id, q):
        t0 = time()
        res = greedy(game)
        res['time'] = time() - t0
        q.put((id, res))
        print(f'greedy {id} done')
        
    q_SA = mp.Manager().Queue()
    q_game = mp.Manager().Queue()
    q_baseline = mp.Manager().Queue()
    q_greedy = mp.Manager().Queue()
    
    res_game = dict()
    res_SA = dict()
    res_baseline = dict()
    res_greedy = dict()
    
    ps = []

    for i in range(n_simulation):
        game = AssignmentGame(Q=Q, K = K)
        game.reset()
        # threads.append(Thread(target = process, args = (game, res[i])))
        ps.append(mp.Process(target = process_game, args = (deepcopy(game), i, q_game,)))
        ps.append(mp.Process(target = process_multiple_SA, args = (deepcopy(game), i, q_SA,)))
        ps.append(mp.Process(target = process_baseline, args = (deepcopy(game), i, q_baseline,)))
        ps.append(mp.Process(target = process_greedy, args = (deepcopy(game), i, q_greedy,)))
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
    while not q_game.empty():
        i, d = q_game.get()
        res_game[i] = d
        
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
        'res_game' : res_game,
        'res_SA' : res_SA,
        'res_baseline' : res_baseline,
        'res_greedy' : res_greedy,
    }
    
    with open(f"res_compare_baseline_greedy_SA_{strategy.__name__}_Q{Q}_K{K}_n{n_simulation}_T{T}.pkl","wb") as f:
        pickle.dump(res, f)
    
    costs_game = -np.array([
        res_game[i]['rewards']
        for i in res_game.keys()
    ])
    
    costs_baseline = -np.array([
        res_baseline[i]['rewards']
        for i in res_baseline.keys()
    ])
    
    costs_greedy = -np.array([
        res_greedy[i]['rewards']
        for i in res_greedy.keys()
    ])
    
    costs_SA = np.amin([[
        res_SA[i][j]['list_best_costs']
        for j in res_SA[i].keys() if j != 'time'
    ] for i in res_SA.keys()],
    axis=1)
    
    # print(costs_game.shape)
    # print(costs_SA.shape)
    
    costs = [
        np.amin(costs_game, axis=1),
        np.amin(costs_SA, axis=1),
        np.amin(costs_baseline, axis=1),
        np.amin(costs_greedy, axis=1),
    ]
    
    labels = [
        strategy.__name__,
        'SA',
        'Greedy Tree Search',
        'IRWDP'
    ]
    
    plt.boxplot(
        costs,
        labels=labels,
        patch_artist=True)
    plt.title(f'Costs for different methods : Q={Q}, K={K}, n={n_simulation}, T={T}')
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
    plt.show()
    
    times = {
        k : [res[k][i]['time'] for i in res[k].keys()]
        for k in res.keys()
    }
    
    plt.boxplot(
        times.values(),
        labels=labels,
        patch_artist=True)
    plt.title(f'Times for different methods : Q={Q}, K={K}, n={n_simulation}, T={T}')
    plt.show()
    
    # sol = res['solution']
    # print('solution : ', sol)
    


if __name__ == '__main__' :
    compare(n_simulation=100, T=500)