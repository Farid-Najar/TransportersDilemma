import numpy as np
from assignment import AssignmentGame
import pickle
from time import time
import multiprocess as mp
from copy import deepcopy

def greedy(game : AssignmentGame, time_budget = 1):
    
    # vehicles_by_priority = np.argsort(game.emissions_KM)
    
    q = np.ones(len(game.packages)+1)
    q[1:] = np.array([
        d.quantity for d in game.packages
    ])
    
    nodes = [
        d.destination for d in game.packages
    ]
    
    action = np.ones(game.num_packages, dtype=bool)
    rewards = np.zeros(game.num_packages)
    excess_emission = np.zeros(game.num_packages)
    
    infos = []
    
    l = [game.hub] + nodes
    x, y = np.ix_(l, l)
    
    A = game.distance_matrix[x, y]@np.diag(1/q)
    A += np.max(A)*np.eye(len(A))
    
    indices = np.flip(np.argsort(np.min(A[1:, 1:], axis=0)))
    
    for t in range(len(action)):
        i = indices[t]
        # A = np.delete(A, (i+1), axis=0)
        # A = np.delete(A, (i+1), axis=1)
        r, _, info = game.step(action.astype(int), time_budget, call_OR=(t==0))
        # action = np.ones(game.num_packages, dtype=bool)
        action[i] = not action[i]
        excess_emission[t] = infos[t]['excess_emission']
        # print(info)
        rewards[t] = r
        infos.append(info)
        if excess_emission[t] <= 0:
            break
    # for m in vehicles_by_priority:
    #     frm = game.hub

    res = {
        'rewards' : rewards,
        'excess_emission' : excess_emission,
        'infos' : infos,
    }
    
    return res

def eliminate_max_median(game : AssignmentGame, time_budget = 1):
    
    # vehicles_by_priority = np.argsort(game.emissions_KM)
    
    q = np.ones(len(game.packages)+1)
    q[1:] = np.array([
        d.quantity for d in game.packages
    ])
    
    nodes = [
        d.destination for d in game.packages
    ]
    
    action = np.ones(game.num_packages, dtype=bool)
    rewards = np.zeros(game.num_packages)
    excess_emission = np.zeros(game.num_packages)
    
    l = [game.hub] + nodes
    x, y = np.ix_(l, l)
    
    A = game.distance_matrix[x, y]@np.diag(1/q) + np.max(A)*np.eye(len(A))
    
    for t in range(len(action)):
        i = np.flip(np.argsort(np.median(A[1:, 1:] + np.max(A[1:, 1:])*np.eye(len(A[1:, 1:])), axis=1)))
        # i = indices[t]
        r, _, info = game.step(action.astype(int), time_budget, call_OR=(t==0))
        # action = np.ones(game.num_packages, dtype=bool)
        action[i] = not action[i]
        # print(info)
        excess_emission[t] = info['excess_emission']
        rewards[t] = r
    # for m in vehicles_by_priority:
    #     frm = game.hub
    
    res = {
        'rewards' : rewards,
        'excess_emission' : excess_emission
    }
    
    return res

def baseline(
    game : AssignmentGame,
    time_budget = 1,
    full_OR = False,
    OR_every = 1,
    *args,
    **kwargs,
    ):
    # vehicles_by_priority = np.argsort(game.emissions_KM)
    
    # q = np.ones(len(game.packages)+1)
    # q[1:] = np.array([
    #     d.quantity for d in game.packages
    # ])
    
    # nodes = [
    #     d.destination for d in game.packages
    # ]
    
    action = np.ones(game.num_packages, dtype=bool)
    rewards = np.zeros(game.num_packages)
    excess_emission = np.zeros(game.num_packages)
    infos = []#[dict() for _ in range(len(action))]
    # omitted = np.zeros(game.num_packages)
    
    best = action.copy()
    solution = best.copy()
    
    # l = [game.hub] + nodes
    # x, y = np.ix_(l, l)
    
    # infos = []
    
    # A = game.distance_matrix[x, y]@np.diag(1/q)
    # indices = np.flip(np.argsort(np.mean(A[1:, 1:] + np.max(A[1:, 1:])*np.eye(len(A[1:, 1:])), axis=1)))
    r_best, _, info = game.step(action.astype(int), time_budget, call_OR=True)
    
    emission = info['excess_emission']
    # o = info['omitted']
    
    indices = list(range(game.num_packages))
    
    for t in range(game.num_packages):
        excess_emission[t] = emission
        # omitted[t] = o
        rewards[t] = r_best
        
        if emission <= 0:
            break
        
        r_best = float('-inf')
        
        for i in indices:
            a = action.copy()
            a[i] = not a[i]
            r, _, info = game.step(a.astype(int), time_budget, call_OR=(full_OR and t%OR_every == 0))
            # action = np.ones(game.num_packages, dtype=bool)
            
            if r > r_best:
                emission = info['excess_emission']
                # o = info['omitted']
                r_best = r
                if r > np.max(rewards[:t+1]):
                    solution = a.copy()
                best = a.copy()
                infos.append(info)
                ii = i
                
        # print(len(indices))
        indices.remove(ii)
        action = best.copy()
        a = action.copy()
                
        
        # infos.append(info)
    # for m in vehicles_by_priority:
    #     frm = game.hub
    
    res = {
        'solution' : solution,
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
):
    def process_baseline(game, id, q):
        t0 = time()
        res = baseline(game, full_OR=full_OR, OR_every=OR_every)
        res['time'] = time() - t0
        q.put((id, res))
        print(f'baseline {id} done')
        
    q = mp.Manager().Queue()
    res = dict()
    
    ps = []

    for i in range(n_simulation):
        game = AssignmentGame(Q=30)
        game.reset(num_packages = 50)
        # threads.append(Thread(target = process, args = (game, res[i])))
        ps.append(mp.Process(target = process_baseline, args = (deepcopy(game), i, q,)))
        ps[i].start()
        
    for i in range(n_simulation):
        ps[i].join()
        
    print('all done !')
    while not q.empty():
        i, d = q.get()
        res[i] = d
        
    with open(f"res_fullOR{OR_every}_baseline_Q{Q}_K{K}_n{n_simulation}_T{T}.pkl","wb") as f:
        pickle.dump(res, f)
        
if __name__ == '__main__':
    
    simulate(n_simulation = 100, full_OR=True, OR_every = 10)
    simulate(n_simulation = 100, full_OR=True, OR_every = 5)
    simulate(n_simulation = 100, full_OR=True, OR_every = 1)
    
    
    # q = (np.random.randint(5, size=(5)) + 1)
    # Q = np.diag(1/q)
    # A = np.random.randint(50, size=(5, 5))
    # np.fill_diagonal(A,0)
    # B = A@np.diag(1/q)

    # print(np.all(B == A@Q))
    # import matplotlib.pyplot as plt
    
    # game = AssignmentGame()#Q = 50, max_capacity=150, grid_size=25)
    # game.reset(50)
    
    # res = greedy(game, time_budget=2)
    # # print(best)
    # # rg, eg = greedy(game, time_budget=1)
    
    # fig, ax1 = plt.subplots()

    # color = 'tab:red'
    # ax1.set_ylabel('rewards', color=color)
    # ax1.plot(res['rewards'], color=color)
    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # ax2.set_ylabel('excess emissions', color=color)  # we already handled the x-label with ax1
    # ax2.plot(res['excess_emission'], color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.title('Results median')
    # plt.show()
    
    # plt.plot(om)
    # plt.show()
    
    # fig, ax1 = plt.subplots()
    
    # color = 'tab:red'
    # ax1.set_ylabel('rewards', color=color)
    # ax1.plot(rg, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # ax2.set_ylabel('excess emissions', color=color)  # we already handled the x-label with ax1
    # ax2.plot(eg, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.title('Results greedy')
    # plt.show()