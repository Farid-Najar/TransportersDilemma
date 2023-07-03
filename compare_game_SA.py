from GameLearning import LRI, GameLearning, EXP3
from SA_baseline import recuit_multiple
import multiprocess as mp
import numpy as np
import matplotlib.pyplot as plt
import pickle

from time import time

from TransportersDilemma.assignment import AssignmentGame

def compare(
    n_simulation = 1,
    strategy = LRI,
    T = 500,
    Q = 30,
    K = 50,
    T_init = 2_000,
    T_limit = 1,
    lamb = .99,
    log = True,
    packages = None,
    ):

    def process_game(game, res_dict, id, q):
        res_dict = GameLearning(game, T=T, strategy=strategy, log = log)
        q.put((id, res_dict))
        # res_dict = d
        
    def process_multiple_SA(game, res, id, q):
        res = recuit_multiple(game, T_init = T_init, T_limit = T_limit, lamb = lamb, id = id, log=log, H=T)
        q.put((id, res))
        
    q_SA = mp.Queue()
    q_game = mp.Queue()
    res_game = dict()
    res_SA = dict()
    ps = []

    for i in range(n_simulation):
        game = AssignmentGame(Q=Q)
        game.reset(num_packages = K)
        res_game[i] = dict()
        res_SA[i] = dict()
        # threads.append(Thread(target = process, args = (game, res[i])))
        ps.append(mp.Process(target = process_game, args = (game, res_game[i], q_game, i,)))
        ps.append(mp.Process(target = process_multiple_SA, args = (game, res_SA[i], q_SA, i,)))
        ps[2*i].start()
        ps[2*i+1].start()
        
    for i in range(n_simulation):
        ps[2*i].join()
        ps[2*i+1].join()
        
    while not q_game.empty():
        i, d = q_game.get()
        res_game[i] = d
        
    while not q_SA.empty():
        i, d = q_SA.get()
        res_SA[i] = d
    
    res = {
        'res_game' : res_game,
        'res_SA' : res_SA
    }
    
    with open(f"res_compare_SA_{strategy.__name__}_Q{Q}_K{K}_n{n_simulation}_T{T}.pkl","wb") as f:
        pickle.dump(res, f)
    
    rewards = np.array([
        res[i]['rewards']
        for i in res.keys()
    ])
    r_min = np.amin(rewards, axis=0)
    r_max = np.amax(rewards, axis=0)
    r_mean = np.mean(rewards, axis=0)
    std = np.std(rewards, axis=0)
    r_median = np.median(rewards, axis=0)

    # fig, ax = plt.subplots(2, 1)
    plt.plot(r_min, linestyle=':', label='min rewards', color='black')
    plt.plot(r_mean, label='mean rewards')
    plt.plot(r_median, label='median rewards', linestyle='--', color='black')
    plt.plot(r_max, label='max rewards', linestyle='-.', color='black')
    plt.fill_between(range(len(r_mean)), r_mean - 2*std, r_mean + 2*std, alpha=0.3, label="mean $\pm 2\sigma$")
    plt.fill_between(range(len(r_mean)), r_mean - std, r_mean + std, alpha=0.7, label="mean $\pm \sigma$")
    plt.title(f'Rewards in {strategy.__name__}')
    plt.xlabel("Time $t$")
    plt.legend()
    plt.show()
    
    # sol = res['solution']
    # print('solution : ', sol)
    


if __name__ == '__main__' :
    compare(n_simulation=50, T=50, log=False)