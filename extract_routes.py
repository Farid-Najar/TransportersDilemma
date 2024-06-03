from copy import deepcopy
from assignment import AssignmentEnv, AssignmentGame, Package, RemoveActionEnv
import numpy as np
import pickle
import multiprocess as mp
from SA_baseline import recuit

def create_routes(env : AssignmentEnv, nb_routes = 5_000, retain_rate = 0., time_budget = 1, change_quantity = False):
    
    def process(env: AssignmentEnv, id, q, retained_dests):
        np.random.seed(id)
        
        dests = retained_dests+list(np.random.choice(
            [
            i for i in range(env._game.grid_size**2)
                if i not in retained_dests
            ],
            size=env._game.num_packages - len(retained_dests)+1,
            replace=False
        ))
        assert len(dests) == len(set(dests))
        
        dests.remove(env._game.hub)
        assert len(dests) == env._game.num_packages
        
        quantities = np.ones(len(dests), dtype=int)
        if change_quantity:
            C = env._game.max_capacity * env._game.num_vehicles - env._game.num_packages
            c = (C*np.random.dirichlet(np.ones(len(dests)))).astype(int)
            quantities += c
        packages = [
            Package(
                destination=dests[k],
                quantity=quantities[k],
            )
            for k in range(len(dests))
        ]
        env.reset(packages = packages, time_budget = time_budget)
        res = {
            'd' : env.destinations,
            'r' : env.initial_routes
        }
        print(f'{id} done')
        q.put((id, res))
        return
        
    if retain_rate:
        comment = f'_retain{retain_rate}'
    else:
        comment = ''
    with open(f'./game_K{env._game.num_packages}{comment}.pkl', 'wb') as f:
        pickle.dump(env._game, f, -1)
    
    q = mp.Manager().Queue()
    
    env.reset()
    shape = env.initial_routes.shape
    routes = np.zeros((nb_routes, shape[0], shape[1]))
    destinations = np.zeros((nb_routes, env._game.num_packages))
    destinations[0] = env.destinations
    routes[0] = env.initial_routes
    retained_dests = list(np.random.choice(
        env.destinations,
        size=int(retain_rate*len(env.destinations)),
        replace=False
    ))
    retained_dests.append(env._game.hub)
    ps = []
    
    for i in range(1, nb_routes):
        ps.append(mp.Process(target = process, args = (deepcopy(env), i, q, retained_dests,)))
        ps[-1].start()
        
    for p in ps:
        p.join()
        
    print('all done !')
    while not q.empty():
        i, d = q.get()
        routes[i] = d['r']
        destinations[i] = d['d']
        
    
    for i in range(len(destinations)):
        if i >= len(destinations):
            break
        if len(destinations[i]) != len(set(destinations[i])):
            destinations = np.delete(destinations, i, 0)
            routes = np.delete(routes, i, 0)

    destinations = destinations[:1000]
    routes = routes[:1000]
    print('operation succeeded!')
    
    np.save(f'routes_K{env._game.num_packages}{comment}', routes)
    np.save(f'destinations_K{env._game.num_packages}{comment}', destinations)
    

def create_labels(K = 100):
    path = 'TransportersDilemma/RL/'
    with open(path+f'game_K{K}.pkl', 'rb') as f:
        g = pickle.load(f)
    routes = np.load(path+f'routes_K{K}.npy')
    dests = np.load(path+f'destinations_K{K}.npy')
    
    def process(g, id, q):
        np.random.seed(id)
        
        env = RemoveActionEnv(game = g, saved_routes = routes, saved_dests=dests, 
        #   obs_mode='action', 
          change_instance = False, rewards_mode='normalized_terminal', instance_id = id)
        env.reset()
        action_SA, *_ = recuit(env._env, 5000, 1, 0.999, H=100_000)
        q.put((id, action_SA))
        print(f'{id} done')
    
    q = mp.Manager().Queue()
    
    y = np.zeros((len(dests), K))
    ps = []
    
    for i in range(len(y)):
        ps.append(mp.Process(target = process, args = (deepcopy(g), i, q,)))
        ps[-1].start()
        
    for i in range(len(ps)):
        ps[i].join()
        
    print('all done !')
    while not q.empty():
        i, a = q.get()
        y[i] = a
        
    
    np.save(f'y_K{K}', y)
    

def create_x(K = 100):
    path = 'TransportersDilemma/RL/'
    with open(path+f'game_K{K}.pkl', 'rb') as f:
        g = pickle.load(f)
    routes = np.load(path+f'routes_K{K}.npy')
    dests = np.load(path+f'destinations_K{K}.npy')
    
    def process(g, id, q):
        np.random.seed(id)
        
        env = RemoveActionEnv(game = g, saved_routes = routes, saved_dests=dests, 
        #   obs_mode='action', 
          change_instance = False, rewards_mode='normalized_terminal', instance_id = id)
        obs, _ = env.reset()
        q.put((id, obs))
        print(f'{id} done')
    
    q = mp.Manager().Queue()
    env = RemoveActionEnv(game = g, saved_routes = routes, saved_dests=dests, 
        #   obs_mode='action', 
          change_instance = False, rewards_mode='normalized_terminal', instance_id = 0)
    obs, _ = env.reset()
    x = np.zeros((len(dests), *obs.shape))
    ps = []
    
    for i in range(len(x)):
        ps.append(mp.Process(target = process, args = (deepcopy(g), i, q,)))
        ps[-1].start()
        
    for i in range(len(ps)):
        ps[i].join()
        
    print('all done !')
    while not q.empty():
        i, a = q.get()
        x[i] = a
        
    
    np.save(f'x_K{K}', x)
    

def create_quantities(nb_routes = 5_000, time_budget = 1):
    
    def process(env: AssignmentEnv, id, q, dests):
        np.random.seed(id)
        
        quantities = np.ones(len(dests), dtype=int)
        C = env._game.max_capacity * env._game.num_vehicles - env._game.num_packages
        c = (C*np.random.dirichlet(np.ones(len(dests)))).astype(int)
        quantities += c
        packages = [
            Package(
                destination=dests[k],
                quantity=quantities[k],
            )
            for k in range(len(dests))
        ]
        env.reset(packages = packages, time_budget = time_budget)
        res = {
            'd' : dests,
            'q' : quantities,
            'r' : env.initial_routes
        }
        print(f'{id} done')
        q.put((id, res))
        return
       
    g = AssignmentGame(
            grid_size=12,
            max_capacity=25,
            Q = 10,
            K=20,
            emissions_KM = [0., .1, .3, .3],
            costs_KM = [1, 1, 1, 1],
            seed=42
        )
    
    comment = f'_retain{1.}'
    with open(f'./game_K{g.num_packages}{comment}.pkl', 'wb') as f:
        pickle.dump(g, f, -1)
    
    q = mp.Manager().Queue()
    env = AssignmentEnv(g)
    env.reset()
    shape = env.initial_routes.shape
    routes = np.zeros((nb_routes, shape[0], shape[1]))
    destinations = np.zeros((nb_routes, env._game.num_packages))
    quantities = np.ones((nb_routes, env._game.num_packages), dtype=int)
    destinations[0] = env.destinations
    routes[0] = env.initial_routes
    retained_dests = env.destinations
    ps = []
    
    for i in range(1, nb_routes):
        env = AssignmentEnv(deepcopy(g))
        ps.append(mp.Process(target = process, args = (deepcopy(env), i, q, retained_dests.copy(),)))
        ps[-1].start()
        
    for p in ps:
        p.join()
        
    print('all done !')
    while not q.empty():
        i, d = q.get()
        routes[i] = d['r']
        destinations[i] = d['d']
        quantities[i] = d['q']
        
    for i in range(len(destinations)):
        if i >= len(destinations):
            break
        if len(destinations[i]) != len(set(destinations[i])):
            destinations = np.delete(destinations, i, 0)
            routes = np.delete(routes, i, 0)
            quantities = np.delete(quantities, i, 0)

    destinations = destinations[:1000]
    routes = routes[:1000]
    print('operation succeeded!')
    
    np.save(f'routes_K{env._game.num_packages}{comment}', routes)
    np.save(f'destinations_K{env._game.num_packages}{comment}', destinations)
    np.save(f'quantities_K{env._game.num_packages}{comment}', quantities)
    

def test():
    g = AssignmentGame(
            grid_size=12,
            max_capacity=25,
            Q = 10,
            K=20,
            emissions_KM = [0., .1, .3, .3],
            costs_KM = [1, 1, 1, 1],
            seed=42
        )
    env = AssignmentEnv(g)
    np.random.seed(42)
    env.reset()
    dests = list(np.random.choice(
        env.destinations,
        size=20,
        replace=False
    ))

    assert len(dests) == len(set(dests))
    
    assert len(dests) == env._game.num_packages
    
    quantities = np.ones(len(dests), dtype=int)
    C = env._game.max_capacity * env._game.num_vehicles - env._game.num_packages
    c = (C*np.random.dirichlet(np.ones(len(dests)))).astype(int)
    quantities += c
    packages = [
        Package(
            destination=dests[k],
            quantity=quantities[k],
        )
        for k in range(len(dests))
    ]
    env.reset(packages = packages, time_budget = 5)
    print(env.quantities)
    

if __name__ == '__main__':
    # g = AssignmentGame(
    #         grid_size=12,
    #         max_capacity=15,
    #         Q = 25,
    #         K=50,
    #         emissions_KM = [0., .1, .3, .3],
    #         costs_KM = [1, 1, 1, 1],
    #         seed=42
    #     )
    # env = AssignmentEnv(g)
    # create_routes(env, 1000, time_budget=60)
    
    # g = AssignmentGame(
    #         grid_size=12,
    #         max_capacity=25,
    #         Q = 10,
    #         K=20,
    #         emissions_KM = [0., .1, .3, .3],
    #         costs_KM = [1, 1, 1, 1],
    #         seed=42
    #     )
    # env = AssignmentEnv(g)
    # create_routes(env, 1050, time_budget=60, retain_rate=1, change_quantity=True)
    # create_labels()
    # create_x()
    # test()
    # create_quantities(1100, time_budget=10)
    
    g = AssignmentGame(
            grid_size=12,
            max_capacity=8,
            Q = 6,
            K=16,
            emissions_KM = [.1, .3],
            costs_KM = [1, 1],
            seed=42
        )
    env = AssignmentEnv(g)
    create_routes(env, 1000, time_budget=5)