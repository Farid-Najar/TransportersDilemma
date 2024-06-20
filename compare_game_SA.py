import sys

# direc = os.path.dirname(__file__)
# pri&
# caution: path[0] is reserved for script path (or '' in REPL)
# print(str(path)+'/ppo')
sys.path.insert(1, '/Users/faridounet/PhD/TransportersDilemma')

from shortcut import multi_types
from GameLearning import LRI, GameLearning, EXP3
from SA_baseline import recuit, recuit_tsp
from a_star import A_Star
from greedy_baseline import baseline, greedy
import multiprocess as mp
import numpy as np
import matplotlib.pyplot as plt
import pickle

from copy import deepcopy
from time import time

from assignment import AssignmentEnv, GameEnv, RemoveActionEnv

def EHEG(env, obs, *args, **kwargs):
    """Eliminates the package with the highest elimination gain

    Parameters
    ----------
    env :
    """
    
    # eenv = deepcopy(env)
    
    # this_env = RemoveActionEnv(game = eenv._env._game, saved_routes = eenv._env.saved_routes, saved_dests=eenv._env.saved_dests, 
    #           obs_mode='elimination_gain', 
    #         action_mode = 'destinations',
    #           change_instance = False, rewards_mode='normalized_terminal', instance_id = i)
    # obs, _ = env.reset()
    while True:
      obs, r, d, *_ = env.step(np.argmax(obs))
      if d:
        break
      
    return r

def compare(
    n_simulation = 1,
    # strategy = LRI,
    T = 100_000,
    Q = None,
    K = 50,
    T_init = 5_000,
    T_limit = 1,
    lamb = 0.9999,
    log = False,
    packages = None,
    retain = None,
    ):
    
    if retain is None:
        with open(f'TransportersDilemma/RL/game_K{K}.pkl', 'rb') as f:
            g = pickle.load(f)
        routes = np.load(f'TransportersDilemma/RL/routes_K{K}.npy')
        dests = np.load(f'TransportersDilemma/RL/destinations_K{K}.npy')
    else:
        with open(f'TransportersDilemma/RL/game_K{K}_retain{retain}.pkl', 'rb') as f:
            g = pickle.load(f)
        routes = np.load(f'TransportersDilemma/RL/routes_K{K}_retain{retain}.npy')
        dests = np.load(f'TransportersDilemma/RL/destinations_K{K}_retain{retain}.npy')
    if Q is not None:
        g.Q = Q
    
    if K == 20:
        qs = np.load(f'TransportersDilemma/RL/quantities_K{K}_retain{retain}.npy')
    
    np.random.seed(42)

    def process_DP(env, i, q):
        t0 = time()
        res = dict()
        
        _, info = env.reset()
        excess = info['excess_emission']
        rtes = np.array([
            [
                env._env.initial_routes[m, j] 
                for j in range(0, len(env._env.initial_routes[m]), 2)
            ]
            for m in range(len(env._env.initial_routes))
        ], dtype=int)
        # print(CM)
        coeff = env._env._game.emissions_KM
        # CM = np.array([
        #     env._env.distance_matrix*coeff[i]
        #     for i in range(len(coeff))
        # ]).copy()
        a = multi_types(env._env.distance_matrix, rtes, coeff, excess)
        # a = dests[i_id][np.where(a_GTS == 0)].astype(int)

        env.reset()
        
        _, r_opt, *_ = env.step(a)
        # res = GameLearning(game, T=T, strategy=strategy, log = log)
        res['time'] = time() - t0
        res['sol'] = a
        res['r'] = r_opt
        q.put((i, res))
        print(f'DP {i} done')
        return
        # res_dict = d
        
    def process_multiple_SA(env, i, q):
        t0 = time()
        res = dict()

        env.reset()
        action_SA, *_ = recuit(deepcopy(env._env), T_init, T_limit, lamb, H=T)
        # res = recuit_multiple(game, T_init = T_init, T_limit = T_limit, lamb = lamb, log=log, H=T)
        a = np.where(action_SA == 0)[0]
        
        env.reset()
        _, r_SA, *_ = env.step(a)
        res['time'] = time() - t0
        res['sol'] = a
        res['r'] = r_SA
        q.put((i, res))
        print(f'SA {i} done')
        return
        
    def process_A(env, i, q):
        t0 = time()
        res = dict()
        
        env.reset()
        
        res_A = A_Star(deepcopy(env._env), max_time=20)
        # res = baseline(game)
        action_A = res_A['solution'].astype(int)

        obs, info= env.reset()
        a = np.where(action_A == 0)[0]
        _, r_A, *_ = env.step(a)
        res['time'] = time() - t0
        res['sol'] = a
        res['r'] = r_A
        q.put((i, res))
        print(f'baseline {i} done')
        return
        
    def process_greedy(env, i, q):
        t0 = time()
        res = dict()
        
        env.reset()
        # r_EG = EHEG(env, obs)
        rs = baseline(deepcopy(env._env))
        a = rs['solution'].astype(int)
        
        env.reset()
        
        ac = np.where(a == 0)[0]
        _, r, *_ = env.step(ac)
        
        res['time'] = time() - t0
        res['sol'] = ac
        res['r'] = r
        q.put((i, res))
        print(f'greedy {i} done')
        return
        
    q_SA = mp.Manager().Queue()
    q_DP = mp.Manager().Queue()
    # q_A = mp.Manager().Queue()
    q_greedy = mp.Manager().Queue()
    
    res_DP = dict()
    res_SA = dict()
    # res_A = dict()
    res_greedy = dict()
    

    ps = []
    for i in range(n_simulation):
        # game = AssignmentGame(Q=Q, K = K)
        # threads.append(Thread(target = process, args = (game, res[i])))
        
        gg = deepcopy(g)
        if K == 20:
            env = RemoveActionEnv(game = gg, saved_routes = routes, saved_dests=dests, saved_q = qs,
                obs_mode='elimination_gain', 
                action_mode = 'destinations',
                  change_instance = False, rewards_mode='normalized_terminal', instance_id = i)
        else:
            env = RemoveActionEnv(game = gg, saved_routes = routes, saved_dests=dests, 
                obs_mode='elimination_gain', 
                action_mode = 'destinations',
                  change_instance = False, rewards_mode='normalized_terminal', instance_id = i)
        
        ps.append(mp.Process(target = process_DP, args = (deepcopy(env), i, q_DP, )))
        ps.append(mp.Process(target = process_multiple_SA, args = (deepcopy(env), i, q_SA,)))
        # ps.append(mp.Process(target = process_A, args = (deepcopy(env), i, q_A,)))
        ps.append(mp.Process(target = process_greedy, args = (deepcopy(env), i, q_greedy,)))
        # ps[i].start()
        ps[3*i].start()
        ps[3*i+1].start()
        ps[3*i+2].start()
        # ps[4*i+3].start()
        
    # for i in range(n_simulation):
        ps[3*i].join()
        ps[3*i+1].join()
        ps[3*i+2].join()
        # ps[4*i+3].join()
        
    # for i in range(n_simulation):
    #     # ps[i].join()
    #     ps[4*i].join()
    #     ps[4*i+1].join()
    #     ps[4*i+2].join()
    #     ps[4*i+3].join()
    # ps = []
    # for i in range(n_simulation):
    #     # game = AssignmentGame(Q=Q, K = K)
    #     # threads.append(Thread(target = process, args = (game, res[i])))
    #     ps.append(mp.Process(target = process_greedy, args = (i, q_greedy,)))
    #     # ps[i].start()
    #     ps[i].start()
        
    # for p in ps:
    #     # ps[i].join()
    #     p.join()
        
    # ps = []
    # for i in range(n_simulation):
    #     # game = AssignmentGame(Q=Q, K = K)
    #     # threads.append(Thread(target = process, args = (game, res[i])))
    #     ps.append(mp.Process(target = process_DP, args = (i, q_DP, )))
    #     # ps[i].start()
    #     ps[i].start()
        
    # for p in ps:
    #     # ps[i].join()
    #     p.join()
    
    # ps = []
    # for i in range(n_simulation):
    #     # game = AssignmentGame(Q=Q, K = K)
    #     # threads.append(Thread(target = process, args = (game, res[i])))
    #     ps.append(mp.Process(target = process_multiple_SA, args = (i, q_SA,)))
    #     # ps[i].start()
    #     ps[i].start()
        
    # for p in ps:
    #     # ps[i].join()
    #     p.join()
        
    # ps = []
    # for i in range(n_simulation):
    #     # game = AssignmentGame(Q=Q, K = K)
    #     # threads.append(Thread(target = process, args = (game, res[i])))
    #     ps.append(mp.Process(target = process_A, args = (i, q_A,)))
    #     # ps[i].start()
    #     ps[i].start()
        
    # for p in ps:
    #     # ps[i].join()
    #     p.join()
    
        
    print('all done !')
    while not q_DP.empty():
        i, d = q_DP.get()
        res_DP[i] = d
        
    while not q_SA.empty():
        i, d = q_SA.get()
        res_SA[i] = d
        
    # while not q_A.empty():
    #     i, d = q_A.get()
    #     res_A[i] = d
    
    while not q_greedy.empty():
        i, d = q_greedy.get()
        res_greedy[i] = d
    
    res = {
        'res_DP' : res_DP,
        'res_SA' : res_SA,
        # 'res_A' : res_A,
        'res_greedy' : res_greedy,
    }
    
    # with open(f"res_compare_baseline_greedy_SA_{strategy.__name__}_Q{Q}_K{K}_n{n_simulation}_T{T}.pkl","wb") as f:
    with open(f"res_compare_EG_A*_SA_K{K}_n{n_simulation}.pkl","wb") as f:
        pickle.dump(res, f)
    
    # r_DP = np.array([
    #     res_DP[i]['r']
    #     for i in res_DP.keys()
    # ])
    
    # # r_baseline = np.array([
    # #     res_A[i]['r']
    # #     for i in res_A.keys()
    # # ])
    
    # # r_greedy = np.array([
    # #     res_greedy[i]['r']
    # #     for i in res_greedy.keys()
    # # ])
    
    # r_SA = np.array([
    #     res_SA[i]['r']
    #     for i in res_SA.keys()
    # ])
    # # np.amin([[
    # #     res_SA[i][j]['list_best_costs']
    # #     for j in res_SA[i].keys() if j != 'time'
    # # ] for i in res_SA.keys()],
    # # axis=1)
    
    # # print(costs_game.shape)
    # # print(costs_SA.shape)
    
    # for i in range(len(r_SA)):
    #     if r_SA[i]>r_DP[i]:
    #         print('SA :')
    #         print(r_SA[i])
    #         print(res_SA[i]['sol'])
    #         print('DP :')
    #         print(r_DP[i])
    #         print(res_DP[i]['sol'])
            
    #         print(50*'-')
    
    # rs = [
    #     r_DP, # np.amin(costs_game, axis=1),
    #     r_SA,#np.amin(r_SA, axis=1),
    #     r_baseline,#np.amin(r_baseline, axis=1),
    #     r_greedy,#np.amin(r_greedy, axis=1),
    # ]
    
    # labels = [
    #     'DP',# strategy.__name__,
    #     'SA',
    #     'A*',
    #     'GTS'
    # ]
    
    # plt.boxplot(
    #     rs,
    #     labels=labels,
    #     patch_artist=True)
    # plt.title(f'Costs for different methods : K={K}, n={n_simulation}')
    # # r_min = np.amin(rewards_game, axis=0)
    # # r_max = np.amax(rewards_game, axis=0)
    # # r_mean = np.mean(rewards_game, axis=0)
    # # std = np.std(rewards_game, axis=0)
    # # r_median = np.median(rewards_game, axis=0)

    # # # fig, ax = plt.subplots(2, 1)
    # # plt.plot(r_min, linestyle=':', label='min rewards', color='black')
    # # plt.plot(r_mean, label='mean rewards')
    # # plt.plot(r_median, label='median rewards', linestyle='--', color='black')
    # # plt.plot(r_max, label='max rewards', linestyle='-.', color='black')
    # # plt.fill_between(range(len(r_mean)), r_mean - 2*std, r_mean + 2*std, alpha=0.3, label="mean $\pm 2\sigma$")
    # # plt.fill_between(range(len(r_mean)), r_mean - std, r_mean + std, alpha=0.7, label="mean $\pm \sigma$")
    # # plt.title(f'Rewards in {strategy.__name__}')
    # # plt.xlabel("Time $t$")
    # # plt.legend()
    # plt.savefig('rs')
    # plt.show()
    
    # times = {
    #     k : [res[k][i]['time'] for i in res[k].keys()]
    #     for k in res.keys()
    # }
    
    # plt.boxplot(
    #     times.values(),
    #     labels=labels,
    #     patch_artist=True)
    # plt.title(f'Times for different methods : K={K}, n={n_simulation}')
    # plt.savefig('ts')
    # plt.show()
    
    # sol = res['solution']
    # print('solution : ', sol)
    
def run_SA_TSP(
    n_simulation = 1,
    # strategy = LRI,
    T = 100_000,
    Q = None,
    K = 50,
    T_init = 5_000,
    T_limit = 1,
    lamb = 0.9999,
    log = False,
    packages = None,
    retain = None,
    n_threads = 5
    ):
    
    if retain is None:
        with open(f'TransportersDilemma/RL/game_K{K}.pkl', 'rb') as f:
            g = pickle.load(f)
        routes = np.load(f'TransportersDilemma/RL/routes_K{K}.npy')
        dests = np.load(f'TransportersDilemma/RL/destinations_K{K}.npy')
    else:
        with open(f'TransportersDilemma/RL/game_K{K}_retain{retain}.pkl', 'rb') as f:
            g = pickle.load(f)
        routes = np.load(f'TransportersDilemma/RL/routes_K{K}_retain{retain}.npy')
        dests = np.load(f'TransportersDilemma/RL/destinations_K{K}_retain{retain}.npy')
    if Q is not None:
        g.Q = Q
    
    if K == 20:
        qs = np.load(f'TransportersDilemma/RL/quantities_K{K}_retain{retain}.npy')
    
    np.random.seed(42)

    def process(env, i, q):
        t0 = time()
        res = dict()

        env.reset()
        action_SA, *_ = recuit_tsp(deepcopy(env), T_init, T_limit, lamb, H=T)
        # res = recuit_multiple(game, T_init = T_init, T_limit = T_limit, lamb = lamb, log=log, H=T)
        # a = np.where(action_SA == 0)[0]
        
        env.reset()
        *_, d, _, info = env.step(action_SA)
        nrmlz = np.sum(env.quantities)*env.omission_cost
        r_SA = float(d)*(nrmlz + info['r'])/nrmlz
        res['time'] = time() - t0
        res['sol'] = action_SA
        res['a'] = info['a']
        res['r'] = r_SA
        q.put((i, res))
        print(f'SA {i} done')
        return
        
    q_SA = mp.Manager().Queue()
    
    res_SA = dict()
    

    ps = []
    for i in range(n_simulation//n_threads):
        # game = AssignmentGame(Q=Q, K = K)
        # threads.append(Thread(target = process, args = (game, res[i])))
            for j in range(n_threads):
                gg = deepcopy(g)
                if K == 20:
                    env = GameEnv(AssignmentEnv(game = gg, saved_routes = routes, saved_dests=dests, saved_q = qs,
                        obs_mode='elimination_gain', 
                          change_instance = False, instance_id = i*n_threads+j))
                else:
                    env = GameEnv(AssignmentEnv(game = gg, saved_routes = routes, saved_dests=dests, 
                        obs_mode='elimination_gain', 
                          change_instance = False, instance_id = i*n_threads+j))

                ps.append(mp.Process(target = process, args = (deepcopy(env), i*n_threads+j, q_SA,)))
                ps[i*n_threads+j].start()
            
            for j in range(n_threads):  
                ps[i*n_threads+j].join()
            
            print(f'{i*n_threads+j} done')
        
    print('all done !')
    while not q_SA.empty():
        i, d = q_SA.get()
        res_SA[i] = d
    
    res = {
        'res_SA' : res_SA,
    }
    
    # with open(f"res_compare_baseline_greedy_SA_{strategy.__name__}_Q{Q}_K{K}_n{n_simulation}_T{T}.pkl","wb") as f:
    with open(f"res_SA_TSP_K{K}_n{n_simulation}.pkl","wb") as f:
        pickle.dump(res, f)
        
def run_DP(
    n_simulation = 1,
    # strategy = LRI,
    T = 100_000,
    # Q = 30,
    K = 50,
    T_init = 5_000,
    T_limit = 1,
    lamb = 0.9999,
    log = False,
    packages = None,
    ):

    def process_DP(env, i, q, excess):
        t0 = time()
        res = dict()
        rtes = np.array([
            [
                env._env.initial_routes[m, i] 
                for i in range(0, len(env._env.initial_routes[m]), 2)
            ]
            for m in range(len(env._env.initial_routes))
        ], dtype=int)
        # print(CM)
        coeff = env._env._game.emissions_KM
        # CM = np.array([
        #     env._env.distance_matrix*coeff[i]
        #     for i in range(len(coeff))
        # ]).copy()
        a = multi_types(env._env.distance_matrix, rtes, coeff, excess)
        # a = dests[i_id][np.where(a_GTS == 0)].astype(int)
        _, r_opt, *_ = env.step(a)
        # res = GameLearning(game, T=T, strategy=strategy, log = log)
        res['time'] = time() - t0
        res['r'] = r_opt
        q.put((i, res))
        print(f'DP {i} done')
        
    q_DP = mp.Manager().Queue()
    
    res_DP = dict()
    
    ps = []
    
    with open(f'TransportersDilemma/RL/game_K{K}_retain0.8.pkl', 'rb') as f:
        g = pickle.load(f)
    routes = np.load(f'TransportersDilemma/RL/routes_K{K}_retain0.8.npy')
    dests = np.load(f'TransportersDilemma/RL/destinations_K{K}_retain0.8.npy')

    env = RemoveActionEnv(game = g, saved_routes = routes, saved_dests=dests, 
                obs_mode='elimination_gain', 
                action_mode = 'destinations',
                  change_instance = True, rewards_mode='normalized_terminal', instance_id = 0)

    for i in range(n_simulation):
        # game = AssignmentGame(Q=Q, K = K)
        obs, info= env.reset()
        # threads.append(Thread(target = process, args = (game, res[i])))
        ps.append(mp.Process(target = process_DP, args = (deepcopy(env), i, q_DP, info['excess_emission'], )))
        # ps[i].start()
        ps[i].start()
        
    for i in range(n_simulation):
        # ps[i].join()
        ps[i].join()
        
    print('all done !')
    while not q_DP.empty():
        i, d = q_DP.get()
        res_DP[i] = d
    
    
    # with open(f"res_compare_baseline_greedy_SA_{strategy.__name__}_Q{Q}_K{K}_n{n_simulation}_T{T}.pkl","wb") as f:
    with open(f"res_DP_K{K}_n{n_simulation}.pkl","wb") as f:
        pickle.dump(res_DP, f)



def compare_SA_DP(
    n_simulation = 1,
    # strategy = LRI,
    T = 100_000,
    # Q = 30,
    K = 50,
    T_init = 5_000,
    T_limit = 1,
    lamb = 0.9999,
    log = False,
    packages = None,
    ):

    def process_DP(env, id, q, excess):
        t0 = time()
        res = dict()
        rtes = np.array([
            [
                env._env.initial_routes[m, i] 
                for i in range(0, len(env._env.initial_routes[m]), 2)
            ]
            for m in range(len(env._env.initial_routes))
        ], dtype=int)
        # print(CM)
        coeff = env._env._game.emissions_KM
        # CM = np.array([
        #     env._env.distance_matrix*coeff[i]
        #     for i in range(len(coeff))
        # ]).copy()
        a = multi_types(env._env.distance_matrix, rtes, coeff, excess)
        # a = dests[i_id][np.where(a_GTS == 0)].astype(int)
        env.reset()
        _, r_opt, *_ = env.step(a)
        # res = GameLearning(game, T=T, strategy=strategy, log = log)
        res['time'] = time() - t0
        res['sol'] = a
        res['r'] = r_opt
        q.put((id, res))
        print(f'DP {id} done')
        # res_dict = d
        
    def process_multiple_SA(env, id, q):
        t0 = time()
        res = dict()
        action_SA, *_ = recuit(deepcopy(env._env), T_init, T_limit, lamb, H=T)
        # res = recuit_multiple(game, T_init = T_init, T_limit = T_limit, lamb = lamb, log=log, H=T)
        a = np.where(action_SA == 0)[0]
        env.reset()
        _, r_SA, *_ = env.step(a)
        res['time'] = time() - t0
        res['sol'] = a
        res['r'] = r_SA
        q.put((id, res))
        print(f'SA {id} done')
        
        
    q_SA = mp.Manager().Queue()
    q_DP = mp.Manager().Queue()
    
    res_DP = dict()
    res_SA = dict()
    
    ps = []
    
    with open(f'TransportersDilemma/RL/game_K{K}.pkl', 'rb') as f:
        g = pickle.load(f)
    routes = np.load(f'TransportersDilemma/RL/routes_K{K}.npy')
    dests = np.load(f'TransportersDilemma/RL/destinations_K{K}.npy')

    

    for i in range(n_simulation):
        # game = AssignmentGame(Q=Q, K = K)
        env = RemoveActionEnv(game = g, saved_routes = routes, saved_dests=dests, 
                obs_mode='elimination_gain', 
                action_mode = 'destinations',
                  change_instance = False, rewards_mode='normalized_terminal', instance_id = i)
        obs, info= env.reset()
        # threads.append(Thread(target = process, args = (game, res[i])))
        ps.append(mp.Process(target = process_DP, args = (deepcopy(env), i, q_DP, info['excess_emission'], )))
        ps.append(mp.Process(target = process_multiple_SA, args = (deepcopy(env), i, q_SA,)))
        # ps[i].start()
        ps[2*i].start()
        ps[2*i+1].start()
        
    for i in range(n_simulation):
        # ps[i].join()
        ps[2*i].join()
        ps[2*i+1].join()
        
    print('all done !')
    while not q_DP.empty():
        i, d = q_DP.get()
        res_DP[i] = d
        
    while not q_SA.empty():
        i, d = q_SA.get()
        res_SA[i] = d
    
    
    res = {
        'res_DP' : res_DP,
        'res_SA' : res_SA,
    }
    
    # with open(f"res_compare_baseline_greedy_SA_{strategy.__name__}_Q{Q}_K{K}_n{n_simulation}_T{T}.pkl","wb") as f:
    # with open(f"res_compare_EG_A*_SA_K{K}_n{n_simulation}.pkl","wb") as f:
    #     pickle.dump(res, f)
    
    r_DP = np.array([
        res_DP[i]['r']
        for i in res_DP.keys()
    ])
    
    r_SA = np.array([
        res_SA[i]['r']
        for i in res_SA.keys()
    ])
    # np.amin([[
    #     res_SA[i][j]['list_best_costs']
    #     for j in res_SA[i].keys() if j != 'time'
    # ] for i in res_SA.keys()],
    # axis=1)
    
    # print(costs_game.shape)
    # print(costs_SA.shape)
    
    for i in range(len(r_SA)):
        if r_SA[i]>r_DP[i]:
            print(50*'-')
            print(i)
            print('SA :')
            print(r_SA[i])
            print(res_SA[i]['sol'])
            print('DP :')
            print(r_DP[i])
            print(res_DP[i]['sol'])
            
            print(50*'-')
            
            
if __name__ == '__main__' :
    # compare_SA_DP(n_simulation=50, K=50)
    # compare(n_simulation=100, K=16)
    # compare(n_simulation=100, K=20, retain=1.)
    # compare(n_simulation=100, K=30)
    # compare(n_simulation=100, K=50)
    # compare(n_simulation=100, K=50, retain=.8)
    # compare(n_simulation=100, K=100, Q=20)
    # compare(n_simulation=100, K=250)
    # run_DP(n_simulation=50, K=100)
    run_SA_TSP(n_simulation=100, K=20, retain=1.)
    run_SA_TSP(n_simulation=100, K=50)
    run_SA_TSP(n_simulation=100, K=100)