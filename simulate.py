import py_compile

py_compile.compile("TransportersDilemma/a_star.py")

from assignment import AssignmentGame
import numpy as np
from copy import deepcopy
import multiprocess as mp
from time import time
from a_star import A_Star
import pickle


def simulate(
    algorithm,
    n_simulation = 100,
    Q = 30,
    K = 50,
    T = 500,
    full_OR = False,
    OR_every = 1,
    depth = 2,
    time_budget = 1,
    max_time = 600,
    add_text = '',
):
    # from multiprocessing import Lock
    # lock = Lock()
    # global tt
    
    def process_A_Star(game, id, q):
        global tt
        
        t0 = time()
        res = algorithm(game, time_budget=time_budget, max_time=max_time)
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
            K = K,
            grid_size=max(12, int(np.sqrt(K))+2),
            max_capacity=K//4+1
        )
        game.reset()
        # threads.append(Thread(target = process, args = (game, res[i])))
        ps.append(mp.Process(target = process_A_Star, args = (deepcopy(game), i, q,)))
        ps[i].start()
        
    for i in range(n_simulation):
        ps[i].join()
        
    print('all done !')
    while not q.empty():
        i, d = q.get()
        res[i] = d
        
    with open(f"res_({algorithm.__name__})_{add_text}_Q{Q}_K{K}_n{n_simulation}.pkl","wb") as f:
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
        A_Star,
        n_simulation = 100,
        Q=3,
        K=10,
        # max_time = 60,
        # time_budget=5
        # add_text='woCompile'
    )