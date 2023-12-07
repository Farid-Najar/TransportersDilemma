from copy import deepcopy
from assignment import AssignmentEnv, AssignmentGame, Package, RemoveActionEnv
import numpy as np
import pickle
import multiprocess as mp
from SA_baseline import recuit

def create_routes(env : AssignmentEnv, nb_routes = 5_000, retain_rate = 0.75):
    
    def process(env, id, q, retained_dests):
        np.random.seed(id)
        
        dests = retained_dests+list(np.random.choice(
            [
            i for i in range(env._game.grid_size**2)
                if i not in retained_dests
            ],
            size=env._game.num_packages - len(retained_dests)+1
        ))
        dests.remove(env._game.hub)
        assert len(dests) == env._game.num_packages
        packages = [
            Package(
                destination=d,
                quantity=1,#TODO
            )
            for d in dests
        ]
        env.reset(packages = packages)
        res = {
            'd' : env.destinations,
            'r' : env.initial_routes
        }
        q.put((id, res))
        print(f'{id} done')
    
    with open(f'./game_K{env._game.num_packages}_retain{retain_rate}.pkl', 'wb') as f:
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
        size=int(retain_rate*len(env.destinations))
    ))
    retained_dests.append(env._game.hub)
    ps = []
    
    for i in range(1, nb_routes):
        ps.append(mp.Process(target = process, args = (deepcopy(env), i, q, retained_dests,)))
        ps[-1].start()
        
    for i in range(len(ps)):
        ps[i].join()
        
    print('all done !')
    while not q.empty():
        i, d = q.get()
        routes[i] = d['r']
        destinations[i] = d['d']
        
    
    np.save(f'routes_K{env._game.num_packages}_retain{retain_rate}', routes) 
    np.save(f'destinations_K{env._game.num_packages}_retain{retain_rate}', destinations) 
    

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
    

if __name__ == '__main__':
    # g = AssignmentGame(
    #         grid_size=15,
    #         max_capacity=25,
    #         Q = 35,
    #         K=100
    #     )
    # env = AssignmentEnv(g)
    # create_routes(env, 2_500, retain_rate=0.8)
    # create_labels()
    create_x()