from copy import deepcopy
from time import time
from typing import Dict, List
import numpy as np
from assignment import AssignmentEnv, GameEnv
from tqdm import tqdm
from threading import Thread
import matplotlib.pyplot as plt
import multiprocess as mp
import pickle

#import itertools as it
from numpy import random as rd

def UCB(pi, a, r, mu, N, t, confidence_level = 0.7, *args, **kwargs):
    pi = np.zeros(mu.shape)
    mu[a] = (r + (N[a]-1)*mu[a])/N[a]
    A = np.argmax(mu + confidence_level*np.sqrt(np.log(t)/N))
    pi[A] = 1
    return pi

def EGreedy(pi, a, r, mu, N, t, epsilon = 0.1, *args, **kwargs):
    pi = np.zeros(mu.shape)
    if not N.all():
        A = np.argmin(N)
        pi[A] = 1
        return pi
    if np.random.rand()<epsilon:
        pi += 1/len(mu)
        return pi
    
    A = np.argmax(mu)
    pi[A] = 1
    return pi

def LRI(pi, a, r, m, M, b = 3e-3, *args, **kwargs):
    pi_a = pi[a]
    if M==m:
        r = 1.
    else:
        r = (M-r)/(M-m)
    pi = pi - b*r*pi
    pi[a] = pi_a + b*r*(1-pi_a)
    ps = np.exp(pi)
    pi = ps/np.sum(ps)
    
    return pi

def EXP3(w, pi, a, r, mu, N, t, gamma = 0.1, *args, **kwargs):
    # global w
    r = 1 + r/2
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
        self.w = np.ones(num_actions)
        
        # if self.strategy == EXP3:
        #     global w
        
    def act(self):
        return int(rd.choice(len(self.pi), p = self.pi))
    
    def update(self, action, reward):
        if np.sum(self.N) == 0:
            self.min = reward
            self.max = reward
        else:
            self.min = min(self.min, reward)
            self.max = max(self.max, reward)
            
        self.N[action] += 1
        self.pi = self.strategy(
            pi = self.pi, a = action, r = reward, mu = self.mu, N = self.N, t = np.sum(self.N), m = self.min, M=self.max,
            w = self.w
        )
        

def GameLearning(env : AssignmentEnv, strategy = LRI, T = 1_000, log = True):
            
    players = [Player(env.num_actions, strategy) for _ in range(env.K)]
    # actions = [players[i].act() for i in range(len(players))]
    # best = rd.randint(game.num_actions, size=game.num_packages, dtype=int)
    best = np.random.randint(env.num_actions, size=env.K)#np.zeros(env.K, dtype=np.int64)
    _, loss, done, _, info = env.step(best)
    
    for i in range(len(players)):
        players[i].update(best[i], -loss[i])
    
    res = dict()
    # res['actions_hist'] = [best]
    res['rewards'] = np.zeros(T+1)
    nrmlz = env.K*env.omission_cost
    res['rewards'][0] = float(done)*(nrmlz + info['r'])/nrmlz
    
    res['infos'] = []
    
    best_reward = float(done)*(nrmlz + info['r'])/nrmlz
    
    for t in tqdm(range(T)):
        
        try:
            actions = np.array([players[i].act() for i in range(len(players))], dtype=np.int64)
        except Exception as e:
            print('Problem occured :')
            print(e)
            # print(w)
            break
        _, loss, done, _, info = env.step(actions)
        for i in range(len(players)):
            players[i].update(actions[i], -loss[i])
            
        res['rewards'][t+1] = (nrmlz + info['r'])/nrmlz
        
        if float(done)*(nrmlz + info['r'])/nrmlz> best_reward:
            best_reward = float(done)*(nrmlz + info['r'])/nrmlz
            best = actions.copy()
            res['infos'].append(info)
        if t%20 == 0 and log:
            print(20*'-')
            print(t)
            print('excess_emission : ', info['excess_emission'])
            print('omitted : ', info['omitted'])
            print('reward : ', info['r'])
            print('best reward : ', best_reward)
            
    res['solution'] = best
    return res
            
            
def make_different_sims(n_simulation = 1, strategy = LRI, T = 500, Q = 30, K=50, log = True, tsp = False, comment = ''):


    def process(env, q, i):
        t0 = time()
        res = GameLearning(env, T=T, strategy=strategy, log = log)
        res['time'] = time() - t0
        q.put((i, res))
        # res_dict = d
        
    q = mp.Manager().Queue()
    res = dict()
    ps = []
    with open(f'TransportersDilemma/RL/game_K{K}.pkl', 'rb') as f:
        g = pickle.load(f)
    routes = np.load(f'TransportersDilemma/RL/routes_K{K}.npy')
    dests = np.load(f'TransportersDilemma/RL/destinations_K{K}.npy')

    # with open(f'game_K{K}.pkl', 'rb') as f:
    #     g = pickle.load(f)
    # routes = np.load(f'routes_K{K}.npy')
    # dests = np.load(f'destinations_K{K}.npy')
    if tsp:
        env = GameEnv(AssignmentEnv(g, routes, dests, 'game'))
    else:
        env = AssignmentEnv(g, routes, dests, 'game')
    
    for i in range(n_simulation):
        env.reset()
        # threads.append(Thread(target = process, args = (game, res[i])))
        ps.append(mp.Process(target = process, args = (deepcopy(env), q, i,)))
        ps[i].start()
        
    for i in range(n_simulation):
        ps[i].join()
        
    while not q.empty():
        i, d = q.get()
        res[i] = d
    
    with open(f"res_GameLearning_{strategy.__name__}_K{K}_n{n_simulation}{comment}.pkl","wb") as f:
        pickle.dump(res, f)
    
    rewards = np.array([
        res[i]['rewards']
        for i in res.keys()
    ])
    r_min = np.amin(rewards, axis=0)
    r_max = np.amax(rewards, axis=0)
    r_mean = np.mean(rewards, axis=0)
    std = np.std(rewards, axis=0) / np.sqrt(len(rewards))
    r_median = np.median(rewards, axis=0)

    # fig, ax = plt.subplots(2, 1)
    # plt.plot(r_min, linestyle=':', label='min rewards', color='black')
    plt.plot(r_mean, label='mean rewards')
    # plt.plot(r_median, label='median rewards', linestyle='--', color='black')
    # plt.plot(r_max, label='max rewards', linestyle='-.', color='black')
    plt.fill_between(range(len(r_mean)), r_mean - 2*std, r_mean + 2*std, alpha=0.3, label="mean $\pm 2\sigma$")
    plt.fill_between(range(len(r_mean)), r_mean - std, r_mean + std, alpha=0.7, label="mean $\pm \sigma$")
    plt.title(f'Rewards in {strategy.__name__}')
    plt.xlabel("Time $t$")
    plt.legend()
    plt.show()
    
    # sol = res['solution']
    # print('solution : ', sol)
    
if __name__ == '__main__' :
    K = 250
    make_different_sims(K = K, strategy = EXP3, n_simulation=52, T=50_000, log=False, tsp=False, comment = 'randomStart')
    # game = AssignmentEnv(obs_mode='game')
    # game.reset()
    # with open(f'TransportersDilemma/RL/game_K{K}.pkl', 'rb') as f:
    #     g = pickle.load(f)
    # routes = np.load(f'TransportersDilemma/RL/routes_K{K}.npy')
    # dests = np.load(f'TransportersDilemma/RL/destinations_K{K}.npy')
    # env = AssignmentEnv(g, routes, dests, 'game')
    # env.reset()
    # GameLearning(env, strategy=EXP3)
