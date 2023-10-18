from assignment import AssignmentGame
import numpy as np
from copy import deepcopy
import multiprocess as mp
from time import time
import pickle

def alphabeta(
    game : AssignmentGame,
    time_budget = 1,
    depth = 2,
    *args,
    **kwargs,
    ):
    
    def aux(
        g : AssignmentGame,
        a : np.ndarray,
        d = 5,
        alpha = -np.infty,
        beta = np.infty,
        ):
    
        r, done, _ = g.step(a, call_OR=False)
        # print(a)
        if d == 0 or done:
            return r
        
        value = -np.infty
        indices = np.where(a == 1)[0]
        for i in indices:
            aa = a.copy()
            aa[i] = 0
            value = max(value, aux(g, aa, d-1, alpha, beta))
            if value > beta:
                break
            beta = min(beta, value)
        
        return value
    
    action = np.ones(game.num_packages)
    rewards = np.zeros(game.num_packages)
    excess_emission = np.zeros(game.num_packages)
    infos = []#[dict() for _ in range(len(action))]
    # omitted = np.zeros(game.num_packages)
    
    # l = [game.hub] + nodes
    # x, y = np.ix_(l, l)
    
    # infos = []
    
    # A = game.distance_matrix[x, y]@np.diag(1/q)
    # indices = np.flip(np.argsort(np.mean(A[1:, 1:] + np.max(A[1:, 1:])*np.eye(len(A[1:, 1:])), axis=1)))
    
    # o = info['omitted']
    
    r_best, _, info = game.step(action, time_budget, call_OR=True)
    
    indices = list(range(game.num_packages))
    
    for t in range(game.num_packages):
        excess = info['excess_emission']
        excess_emission[t] = excess
        # omitted[t] = o
        rewards[t] = r_best
        
        if excess <= 0:
            break
        
        r_best = -np.infty
        values = np.full(game.num_packages, -np.infty)
        # dones = np.full(game.num_packages, False)
        
        for i in indices:
            a = action.copy()
            a[i] = 0
            values[i] = aux(game, a, depth)
            # action = np.ones(game.num_packages, dtype=bool)
            
            # if r > r_best:
            #     # o = info['omitted']
            #     r_best = r
            #     if r > np.max(rewards[:t+1]):
            #         solution = a.copy()
            #     best = a.copy()
            #     infos.append(info)
            #     ii = i
                
        # print(len(indices))
        ii = np.argmax(values)
        indices.remove(ii)
        action[ii] = 0
        r_best, _, info = game.step(action, call_OR=False)
        
        # infos.append(info)
    # for m in vehicles_by_priority:
    #     frm = game.hub
    
    print(r_best)
    
    
    res = {
        'solution' : action,
        'rewards' : rewards,
        'excess_emission' : excess_emission,
        'infos' : infos,
    }
    
    return res
    
    
    



def simulate(
    n_simulation = 100,
    Q = 30,
    K = 50,
    T = 500,
    full_OR = False,
    OR_every = 1,
    depth = 2,
):
    def process_alphabeta(game, id, q):
        t0 = time()
        res = alphabeta(game, depth=depth)
        res['time'] = time() - t0
        q.put((id, res))
        print(f'baseline {id} done')
        
    q = mp.Manager().Queue()
    res = dict()
    
    ps = []

    for i in range(n_simulation):
        game = AssignmentGame(Q=Q, K = K)
        game.reset()
        # threads.append(Thread(target = process, args = (game, res[i])))
        ps.append(mp.Process(target = process_alphabeta, args = (deepcopy(game), i, q,)))
        ps[i].start()
        
    for i in range(n_simulation):
        ps[i].join()
        
    print('all done !')
    while not q.empty():
        i, d = q.get()
        res[i] = d
        
    with open(f"res_AlphaBeta{depth}_Q{Q}_K{K}_n{n_simulation}_T{T}.pkl","wb") as f:
        pickle.dump(res, f)
        

if __name__ == '__main__':
    # game = AssignmentGame(Q=30)
    # game.reset(num_packages = 50)
    # res = alphabeta(game)
    # print(res)
    simulate(n_simulation = 100, depth=3)
    simulate(n_simulation = 100, depth=4)