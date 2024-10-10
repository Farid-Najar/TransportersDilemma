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
    if not(r>=0 and r<=1):
        r = np.clip(r, 0, 1)
        print(f'r  : {r}')
    # assert r>=0 and r<=1, f'r  : {r}'
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
    nrmlz = np.sum(env.quantities)*env.omission_cost
    res['rewards'][0] = float(done)*(nrmlz + info['r'])/nrmlz
    
    res['infos'] = []
    
    best_reward = res['rewards'][0]
    
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
            
        res['rewards'][t+1] = float(done)*(nrmlz + info['r'])/nrmlz
        
        if res['rewards'][t+1]> best_reward:
            best_reward = res['rewards'][t+1]
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
            
            
def make_different_sims(n_simulation = 1, strategy = LRI, T = 500, Q = 30, K=50, log = True, tsp = False, 
                        comment = '', n_threads=5, real_data = False):

    real = "real_" if real_data else ""

    def process(env, q, i):
        t0 = time()
        res = GameLearning(env, T=T, strategy=strategy, log = log)
        res['time'] = time() - t0
        q.put((i, res))
        # res_dict = d
        
    q = mp.Manager().Queue()
    res = dict()
    ps = []
    if K == 20:
        with open(f'TransportersDilemma/RL/{real}game_K{K}_retain1.0.pkl', 'rb') as f:
            g = pickle.load(f)
        routes = np.load(f'TransportersDilemma/RL/{real}routes_K{K}_retain1.0.npy')
        dests = np.load(f'TransportersDilemma/RL/{real}destinations_K{K}_retain1.0.npy')
        qs = np.load(f'TransportersDilemma/RL/{real}quantities_K{K}_retain1.0.npy')
        
        if tsp:
            env = GameEnv(AssignmentEnv(game=g, saved_routes=routes, saved_dests=dests, saved_q=qs, obs_mode='game'))
        else:
            env = AssignmentEnv(game=g, saved_routes=routes, saved_dests=dests, saved_q=qs, obs_mode='game')

    else:
        with open(f'TransportersDilemma/RL/{real}game_K{K}.pkl', 'rb') as f:
            g = pickle.load(f)
        routes = np.load(f'TransportersDilemma/RL/{real}routes_K{K}.npy')
        dests = np.load(f'TransportersDilemma/RL/{real}destinations_K{K}.npy')

        if tsp:
            env = GameEnv(AssignmentEnv(game=g, saved_routes=routes, saved_dests=dests, obs_mode='game'))
        else:
            env = AssignmentEnv(game=g, saved_routes=routes, saved_dests=dests, obs_mode='game')
    
    for i in range(n_simulation//n_threads):
        # threads.append(Thread(target = process, args = (game, res[i])))
        for j in range(n_threads):
            _, info = env.reset()
            ps.append(mp.Process(target = process, args = (deepcopy(env), q, i*n_threads+j,)))
            ps[i*n_threads+j].start()
        for j in range(n_threads):
            ps[i*n_threads+j].join()
        print(f'{i*n_threads+j} done')
    # for p in ps:
    #     p.join()
        
    while not q.empty():
        i, d = q.get()
        res[i] = d
    
    with open(f"{real}res_GameLearning_{strategy.__name__}_K{K}_n{n_simulation}{comment}.pkl","wb") as f:
        pickle.dump(res, f)
    
    # rewards = np.array([
    #     res[i]['rewards']
    #     for i in res.keys()
    # ])
    # r_min = np.amin(rewards, axis=0)
    # r_max = np.amax(rewards, axis=0)
    # r_mean = np.mean(rewards, axis=0)
    # std = np.std(rewards, axis=0) / np.sqrt(len(rewards))
    # r_median = np.median(rewards, axis=0)

    # # fig, ax = plt.subplots(2, 1)
    # # plt.plot(r_min, linestyle=':', label='min rewards', color='black')
    # plt.plot(r_mean, label='mean rewards')
    # # plt.plot(r_median, label='median rewards', linestyle='--', color='black')
    # # plt.plot(r_max, label='max rewards', linestyle='-.', color='black')
    # plt.fill_between(range(len(r_mean)), r_mean - 2*std, r_mean + 2*std, alpha=0.3, label="mean $\pm 2\sigma$")
    # plt.fill_between(range(len(r_mean)), r_mean - std, r_mean + std, alpha=0.7, label="mean $\pm \sigma$")
    # plt.title(f'Rewards in {strategy.__name__}')
    # plt.xlabel("Time $t$")
    # plt.legend()
    # plt.show()
    
    # sol = res['solution']
    # print('solution : ', sol)
    
if __name__ == '__main__' :
    # K = 50
    # make_different_sims(K = K, strategy = LRI, n_simulation=100, T=10_000, log=False, tsp=True, comment = '_tsp')
    # make_different_sims(K = K, strategy = LRI, n_simulation=100, T=10_000, log=False, tsp=False, comment = '_vrp')
    # make_different_sims(K = K, strategy = EXP3, n_simulation=100, T=10_000, log=False, tsp=True, comment = '_tsp')
    # make_different_sims(K = K, strategy = EXP3, n_simulation=100, T=10_000, log=False, tsp=False, comment = '_vrp')
    
    # K = 100
    # make_different_sims(K = K, strategy = LRI, n_simulation=100, T=15_000, log=False, tsp=True, comment = '_tsp')
    # make_different_sims(K = K, strategy = LRI, n_simulation=100, T=15_000, log=False, tsp=False, comment = '_vrp')
    # make_different_sims(K = K, strategy = EXP3, n_simulation=100, T=15_000, log=False, tsp=True, comment = '_tsp')
    # make_different_sims(K = K, strategy = EXP3, n_simulation=100, T=15_000, log=False, tsp=False, comment = '_vrp')
    
    K = 20
    # make_different_sims(K = K, strategy = LRI, n_simulation=100, T=10_000, log=False, tsp=True, comment = '_tsp')
    # # make_different_sims(K = K, strategy = LRI, n_simulation=100, T=10_000, log=False, tsp=False, comment = '_vrp')
    make_different_sims(K = K, strategy = EXP3, n_simulation=100, T=10_000, log=False, tsp=True, comment = '_tsp')
    # K = 30
    # make_different_sims(K = K, strategy = EXP3, n_simulation=100, T=10_000, log=False, tsp=True, comment = '_tsp')
    # make_different_sims(K = K, strategy = EXP3, n_simulation=100, T=10_000, log=False, tsp=False, comment = '_vrp')
    # game = AssignmentEnv(obs_mode='game')
    # game.reset()
    # with open(f'TransportersDilemma/RL/game_K{K}_retain1.0.pkl', 'rb') as f:
    #     g = pickle.load(f)
    # routes = np.load(f'TransportersDilemma/RL/routes_K{K}_retain1.0.npy')
    # dests = np.load(f'TransportersDilemma/RL/destinations_K{K}_retain1.0.npy')
    # qs = np.load(f'TransportersDilemma/RL/quantities_K{K}_retain1.0.npy')
    # env = GameEnv(AssignmentEnv(game=deepcopy(g), saved_routes=routes, saved_dests=dests, saved_q=qs, obs_mode='game', instance_id=37, change_instance=False))
    
    # # with open(f'TransportersDilemma/RL/game_K{K}.pkl', 'rb') as f:
    # #     g = pickle.load(f)
    # # routes = np.load(f'TransportersDilemma/RL/routes_K{K}.npy')
    # # dests = np.load(f'TransportersDilemma/RL/destinations_K{K}.npy')
    # # env = GameEnv(AssignmentEnv(game=deepcopy(g), saved_routes=routes, saved_dests=dests, obs_mode='game', instance_id=1, change_instance=False))
    # _, info = env.reset()
    # # env.render()
    # print(info)
    # action = np.array([0, 1, 0, 0 ,0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3])
    # # action = np.zeros(K, dtype=int)
    # # print(np.sum(env._env.quantities[np.where(action==0)[0]]))
    # *_, info = env.step(action)
    # nrmlz = env.K*env.omission_cost
    # print(info)
    # print((nrmlz + info['r'])/nrmlz)
    # env.render()
