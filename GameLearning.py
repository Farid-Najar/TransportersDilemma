from typing import Dict, List
import numpy as np
from assignment import AssignmentGame
from tqdm import tqdm
from threading import Thread
import matplotlib.pyplot as plt
import multiprocess as mp

#import itertools as it
from numpy import random as rd

def UCB(pi, a, r, mu, N, t, confidence_level = 0.7, *args, **kwargs):
    pi = np.zeros(mu.shape)
    mu[a] = (r + (N[a]-1)*mu[a])/N[a]
    A = np.argmax(mu + confidence_level*np.sqrt(np.log(t)/N))
    pi[A] = 1
    return pi

def LRI(pi, a, r, b = 3e-3, *args, **kwargs):
    pi_a = pi[a]
    pi = pi - b*r*pi
    pi[a] = pi_a + b*r*(1-pi_a)
    ps = np.exp(pi)
    pi = ps/np.sum(ps)
    
    return pi

def EXP3(pi, a, r, mu, N, t, gamma = 0.1, *args, **kwargs):
    global w
    r = 1 + r/1e9
    assert r>=0 and r<=1
    K = len(pi)
    x = r/pi[a]
    w[a] *= np.exp(gamma*x/(K*np.sqrt(t)))
    pi = (1-gamma)*w/np.sum(w) + gamma/K
    # print(w)
    # if eta is None:
    #     eta = 
    return pi

class Player:
    def __init__(self, num_actions, strategy):
        self.pi = np.ones(num_actions)/num_actions
        self.mu = np.zeros(num_actions)
        self.N  = np.zeros(num_actions)
        self.strategy = strategy
        
        if self.strategy == EXP3:
            global w
            w = np.ones(num_actions)
        
    def act(self):
        return int(rd.choice(len(self.pi), p = self.pi))
    
    def update(self, action, reward):
        self.N[action] += 1
        self.pi = self.strategy(
            pi = self.pi, a = action, r = reward, mu = self.mu, N = self.N, t = np.sum(self.N)
        )
        

def GameLearning(game : AssignmentGame, strategy = UCB, T = 1_000, log = True, res = None):
            
    players = [Player(game.num_actions, strategy) for _ in range(game.num_packages)]
    # actions = [players[i].act() for i in range(len(players))]
    # best = rd.randint(game.num_actions, size=game.num_packages, dtype=int)
    best = np.ones(game.num_packages, dtype=int)
    r, _, info = game.step(best)
    
    for i in range(len(players)):
        players[i].update(best[i], r)
    
    if res is None:
        res = dict()
    res['actions_hist'] = [best]
    res['rewards'] = np.zeros(T+1)
    res['rewards'][0] = r
    
    best_reward = r
    
    for t in tqdm(range(T)):
        
        try:
            actions = [players[i].act() for i in range(len(players))]
        except Exception as e:
            print('Problem occured :')
            print(e)
            print(w)
            break
        r, _, info = game.step(actions)
        for i in range(len(players)):
            players[i].update(actions[i], r)
            
        res['rewards'][t+1] = r
        
        if r > best_reward:
            best_reward = r
            best = actions.copy()
        if t%20 == 0 and log:
            print(20*'-')
            print(t)
            print('excess_emission : ', info['excess_emission'])
            print('omitted : ', info['omitted'])
            print('reward : ', r)
            print('best reward : ', best_reward)
            
    res['solution'] = best
    return res
            
            
def make_different_sims(n_simulation = 1, strategy = LRI, T = 500, Q = 30, K=50, log = True):


    def process(game, res_dict, q, i):
        res = GameLearning(game, T=T, strategy=strategy, log = log, res=res_dict)
        q.put((i, res))
        # res_dict = d
        
    q = mp.Queue()
    res = dict()
    ps = []
    for i in range(n_simulation):
        game = AssignmentGame(Q=Q)
        game.reset(num_packages = K)
        res[i] = dict()
        # threads.append(Thread(target = process, args = (game, res[i])))
        ps.append(mp.Process(target = process, args = (game, res[i], q, i,)))
        ps[i].start()
        
    for i in range(n_simulation):
        ps[i].join()
        
    while not q.empty():
        i, d = q.get()
        res[i] = d
    
    import pickle
    with open(f"res_GameLearning_{strategy.__name__}_Q{Q}_K{K}_n{n_simulation}.pkl","wb") as f:
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
    make_different_sims(n_simulation=50, T=50, log=False)
