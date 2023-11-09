import py_compile

# try:
#     py_compile.compile("TransportersDilemma/assignment.py")
# except Exception as e:
#     py_compile.compile("assignment.py")

from assignment import AssignmentGame
import numpy as np
from copy import deepcopy
import multiprocess as mp
from time import time
import pickle

from assignment import AssignmentGame, test_assignment_env, test_assignment_game, AssignmentEnv
import pstats
import cProfile
from timeit import timeit

# def f():
#     game = AssignmentGame(
#             Q=300,
#             grid_size=45,
#             max_capacity=125
#         )
#     game.reset(num_packages = 500)
#     res = A_Star(game, 10)


def compare_limit_times(game : AssignmentGame, times = range(1, 11)):
    game.reset()
    costs = np.zeros(len(times))
    t = 0
    for time in times:
        r, *_ = game.step(
            np.ones(game.num_packages, int),
            time_budget=time
        )
        costs[t] = -r
        t += 1
    return costs
    
    
    


def simulate(
    n_simulation = 100,
    # Q = 30,
    # K = 50,
    # T = 500,
    # full_OR = False,
    # OR_every = 1,
    # depth = 2,
    # time_budget = 1,
    save = True,
):
    # from multiprocessing import Lock
    # lock = Lock()
    # global tt
    game50 = AssignmentGame(
        Q=0,
        K = 50,
        grid_size=25,
        max_capacity=15
    )

    game75 = AssignmentGame(
            Q=0,
            K = 75,
            grid_size=25,
            max_capacity=20
        )

    game100 = AssignmentGame(
            Q=0,
            K = 100,
            grid_size=25,
            max_capacity=25
        )
    
    def process(game : AssignmentGame, id, q):
        if game.num_packages == 50:
            times = range(1, 11)
        elif game.num_packages == 75:
            times = range(1, 16)
        else:
            times = range(1, 21, 2)
            
        # t0 = time()
        costs = compare_limit_times(game, times)
        # res['time'] = time() - t0
        q.put((id, costs))
        print(f'{game.num_packages} {id} done')
        # with lock:
        #     tt += 1
        #     print(f'{tt}/{n_simulation}')
        
    q50 = mp.Manager().Queue()
    q75 = mp.Manager().Queue()
    q100 = mp.Manager().Queue()
    
    res50 = dict()
    res75 = dict()
    res100 = dict()
    
    res50['times'] = range(1, 11)
    res75['times'] = range(1, 16)
    res100['times'] = range(1, 21, 2)
    
    ps = []
    game50.reset()
    game75.reset()
    game100.reset()
    for i in range(n_simulation):
        
        # threads.append(Thread(target = process, args = (game, res[i])))
        ps.append(mp.Process(target = process, args = (deepcopy(game50), i, q50,)))
        ps.append(mp.Process(target = process, args = (deepcopy(game75), i, q75,)))
        ps.append(mp.Process(target = process, args = (deepcopy(game100), i, q100,)))
        ps[3*i].start()
        ps[3*i+1].start()
        ps[3*i+2].start()
        
    for i in range(n_simulation):
        ps[3*i].join()
        ps[3*i+1].join()
        ps[3*i+2].join()
        
    print('all done !')
    while not q50.empty():
        i, d = q50.get()
        res50[i] = d
        
    while not q75.empty():
        i, d = q75.get()
        res75[i] = d
        
    while not q100.empty():
        i, d = q100.get()
        res100[i] = d
    
    
    res = {
        'res50' : res50,
        'res75' : res75,
        'res100' : res100,
    }
    
    with open(f"res_compare_ORtimes_50_75_100_n{n_simulation}.pkl","wb") as f:
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
        n_simulation = 50,
        # Q=30,
        # K=100,
        # max_time = 60,
        # time_budget=5,
        # save = False
    )