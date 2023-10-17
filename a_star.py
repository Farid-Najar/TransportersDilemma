import py_compile

try:
    py_compile.compile("TransportersDilemma/assignment.py")
except Exception as e:
    py_compile.compile("assignment.py")

from assignment import AssignmentGame
import numpy as np
from copy import deepcopy
import multiprocess as mp
from time import time
import pickle

def A_Star(
    game : AssignmentGame,
    time_budget = 1,
    max_time = 60,
    # depth = 2,
    ):
    
    t0 = time()
    key = lambda l : ''.join([str(x) for x in l])
    key_to_np = lambda k : np.array(list(k), dtype=int)
    action = np.ones(game.num_packages, dtype=int)
    r_best, d, info = game.step(action, time_budget, call_OR=True)
    
    rewards = np.zeros(game.num_packages)
    excess_emission = np.zeros(game.num_packages)
    infos = []#[dict() for _ in range(len(action))]
    
    open_nodes = {key(action)}
    
    done = dict()
    f = dict()
    # pivots = dict()
    
    done[key(action)] = d
    f[key(action)] = -r_best
    # pivots[key(action)] = 0
    t = 0
    # r, done, _ = game.step(action, call_OR=False)
    
    while len(open_nodes) > 0:
        excess_emission[t] = info['excess_emission']
        rewards[t] = r_best
        infos.append(info)
        
        current = min(f, key=f.get)
        
        if done[current]:
            break
        
        action = key_to_np(current)
        if time() - t0 > max_time:
            break
        # pivot = pivots[current]
        # indices = np.where(action[pivot:] == 1)[0]
        indices = np.where(action == 1)[0]
        open_nodes.discard(current)
        
        for i in indices:
            a = action.copy()
            a[i] = 0
            neighbor = key(a)
            # g[neighbor] = g[current] + 1
            r, d, inf = game.step(a, call_OR=False)
            f[neighbor] = - r
            done[neighbor] = d
            # pivots[neighbor] = i
            
            open_nodes.add(neighbor)
            
            if r>r_best:
                r_best = r
                info = inf
            
            
            
            

    
    # omitted = np.zeros(game.num_packages)
    
    # l = [game.hub] + nodes
    # x, y = np.ix_(l, l)
    
    # infos = []
    
    # A = game.distance_matrix[x, y]@np.diag(1/q)
    # indices = np.flip(np.argsort(np.mean(A[1:, 1:] + np.max(A[1:, 1:])*np.eye(len(A[1:, 1:])), axis=1)))
    
    # o = info['omitted']
    
    
    # print(r_best)
    
    
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
    time_budget = 1,
    save = True,
):
    # from multiprocessing import Lock
    # lock = Lock()
    # global tt
    
    def process_A_Star(game, id, q):
        global tt
        
        t0 = time()
        res = A_Star(game, time_budget=time_budget)
        res['time'] = time() - t0
        q.put((id, res))
        print(f'baseline {id} done')
        # with lock:
        #     tt += 1
        #     print(f'{tt}/{n_simulation}')
        
    q = mp.Manager().Queue()
    
    res = dict()
    
    ps = []

    for i in range(n_simulation):
        game = AssignmentGame(
            Q=Q,
            grid_size=max(12, int(np.sqrt(K))+2),
            max_capacity=K//4+1
        )
        game.reset(num_packages = K)
        # threads.append(Thread(target = process, args = (game, res[i])))
        ps.append(mp.Process(target = process_A_Star, args = (deepcopy(game), i, q,)))
        ps[i].start()
        
    for i in range(n_simulation):
        ps[i].join()
        
    print('all done !')
    while not q.empty():
        i, d = q.get()
        res[i] = d
        
    if save:
        with open(f"res_A*_v1_Q{Q}_K{K}_n{n_simulation}.pkl","wb") as f:
            pickle.dump(res, f)
        

if __name__ == '__main__':
    # game = AssignmentGame(
    #     Q=50,
    #     grid_size=15,
    #     max_capacity=25
    # )
    # game.reset(num_packages = 100)
    # res = A_Star(game)
    # print(res)
    simulate(
        n_simulation = 10,
        # Q=30,
        # K=100,
        # max_time = 60,
        # time_budget=5,
        # save = False
    )