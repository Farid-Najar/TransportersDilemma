from copy import deepcopy
from dataclasses import dataclass
import networkx as nx
import numpy as np
from numba import njit
from numba.typed import List
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn import preprocessing

from transporter import Transporter

from typing import Any, Dict, Optional
from time import time
import gymnasium as gym

@dataclass
class Package:
    destination : int
    origin : int = 0
    quantity : int = 1

class AssignmentGame:
    def __init__(self, 
                 transporter : Transporter = None,
                 grid_size : int = 12,
                 hub : int = 100,
                 seed = None,
                 max_capacity = 15,
                 horizon = 1_000,
                 is_VRP = True,
                 emissions_KM = [0, .15, .3, .3],
                 costs_KM = [4, 4, 4, 4],
                 CO2_penalty = 10_000,
                 K = 50,
                 Q = 25,
                 ):
        
        # if np.all(np.array(transporters_hubs) >= size**2):
            # raise('transporters\' locations are not smaller than nodes')
        
        assert len(emissions_KM) == len(costs_KM)
        
        num_vehicles = len(emissions_KM)
        
        self.total_capacity = max_capacity * num_vehicles
        
        self.emissions_KM = np.array(emissions_KM)#, dtype=int)
        self.costs_KM = np.array(costs_KM)
        self.Q = Q
        
        self.info = dict()
        
        self.grid_size = grid_size
        self.hub = hub
        self.transporters_vehicles_colors = ('lightcoral', 'lightgreen', 'lightyellow', 'lightblues')
        
        self.G = nx.grid_2d_graph(grid_size, grid_size)
        distances = {
            e : {
                'distance' : np.random.binomial(12, 0.1)+1
            }
            for e in self.G.edges
        }
        nx.set_edge_attributes(self.G, distances)
        self.distance_matrix = nx.floyd_warshall_numpy(self.G, weight = 'distance')
        self.time_matrix = self.distance_matrix/40 #In cities, the average speed is 40 km/h
        self.mask = None
        
        
        self.omission_cost = (2*np.max(self.distance_matrix) +1)*np.max(self.costs_KM)
        #np.sum(self._get_costs(2*np.max(self.distance_matrix) +1, 0))
        self.CO2_penalty = max(CO2_penalty, 2*self.omission_cost)
        
        self.cost_matrix = np.array([
            (self.costs_KM[m] + self.CO2_penalty*self.emissions_KM[m])*self.distance_matrix
            for m in range(num_vehicles)
        ], dtype=int)
        
        
        self.num_vehicles = num_vehicles
        
        self.max_capacity = max_capacity
        
        self.horizon = horizon
        
        self.num_packages = K
        
        if transporter is not None:
            self.transporter = transporter
            
        else:
            if is_VRP:
                self.transporter = [
                    Transporter(
                        self.distance_matrix,
                        self.cost_matrix,
                        transporter_hub=hub,
                        omission_cost=self.omission_cost,
                        max_capacity=max_capacity,
                        num_vehicles=num_vehicles
                )]
            else:
                self.transporter = [
                    Transporter(
                        self.distance_matrix,
                        self.cost_matrix, 
                        transporter_hub=hub,
                        omission_cost=self.omission_cost,
                        max_capacity=max_capacity,
                        num_vehicles=1
                    )
                    for _ in range(num_vehicles)
                ]
        
        self.ids = [
            str(i) for i in range(len(self.transporter))
        ]

        self.seed(seed)
        self.solutions = [[] for _ in range(len(self.transporter))]
        
    def seed(self, seed = None):
        np.random.seed(seed)
        # super().seed(seed)
        
    @property
    def num_actions(self):
        return len(self.transporter) + 1
        
    def get_ids(self):
        return self.ids

    def render(self):
        # _, ax = plt.subplots()
        # nx.draw_networkx(self.G, 
        #                  pos = dict( (n, n) for n in self.G.nodes()),  
        #                  ax=ax, 
        #                  font_size=5, 
        #                  with_labels=False, 
        #                  node_size=100, 
        #                  node_color=self.G_ncolors, 
        #                  node_shape='s')
        # # plt.draw()
        # # return super().render()
        # return ax
        pass
        
    
    def _compute_cost(self, actions, time_budget, call_OR):
        
        # nodes = [[] for _ in self.transporter]
        # quantities = [[] for _ in self.transporter]
        
        # @njit
        # def _aux(
        #     a,
        #     packages,
        #     mask,
        #     n_vehicles,
        #     time_budget,
        #     mask_is_None,
        #     call,
        #     d_matrix,
        #     t_matrix,
        #     solutions,
        #     transporter,
        #     num_packages,
            
        # ):
        #     deliveries = np.array([
        #         [
        #             packages[k]
        #             for k in range(len(a))
        #             if a[k] == m+1
        #         ]
        #         for m in range(n_vehicles)
        #     ])
            
        #     nodes = np.array([
        #         [d.destination for d in deliveries[m]]
        #         for m in range(n_vehicles)
        #     ])
            
        #     quantities = np.array([
        #         [d.quantity for d in deliveries[m]]
        #         for m in range(n_vehicles)
        #     ])
            
        #     omitted = \
        #         np.sum([p.quantity for p in packages])\
        #         - \
        #         np.sum(quantities)
            
        #     # for k in range(len(actions)):
        #     #     if actions[k]:
        #     #         nodes[actions[k]-1].append(self.packages[k].destination)
        #     #         quantities[actions[k]-1].append(self.packages[k].quantity)
        #     #     else:
        #     #         omission_penalty += self.omission_cost*self.packages[k].quantity
        #     #         omitted += 1
        #     # # print(nodes)
            
        #     if mask_is_None:
        #         if np.sum(a) == num_packages:
        #             l = [self.hub] + list(nodes[0])
        #             mask = np.ix_(l, l)
        #         else:
        #             print("You have to call the OR-routing at least once with all packages included !")
            
        #     # sol = None
            
        #     if call:
        #         for m in range(n_vehicles):#TODO parallelize
        #             #TODO for the TSP case
        #             sol = transporter[m].compute_cost(nodes[m], quantities[m], time_budget)
                    
        #             if np.sum(actions) == num_packages:
        #                 solutions[m] = sol
            
        #     else:#TODO for the TSP case
                
                
        #         omitted_packages = np.where(a == 0)[0] + 1 # important to add 1 to ignore the hub's index 0
                
                
        #         sol = [
        #             [
        #                 solutions[0][m][i]
        #                 for i in range(len(solutions[0][m]))
        #                 if solutions[0][m][i] not in omitted_packages
        #             ]
        #             for m in range(len(solutions[0]))
        #         ]
                
        #     distance_matrix = d_matrix[mask]
        #     time_matrix = t_matrix[mask]
        #     distance = np.zeros(n_vehicles)
        #     time = np.zeros(n_vehicles)
        #     for m in range(len(sol)):
        #         for i in range(len(sol[m])-1):
        #             distance[m] += distance_matrix[sol[m][i], sol[m][i+1]]
        #             time[m] += time_matrix[sol[m][i], sol[m][i+1]]
                    
        #     return sol, distance, time, omitted, mask
        
        deliveries = [
            [
                self.packages[k]
                for k in range(len(actions))
                if actions[k] == m+1
            ]
            for m in range(len(self.transporter))
        ]
        
        nodes = [
            [d.destination for d in deliveries[m]]
            for m in range(len(self.transporter))
        ]
        quantities = [
            [d.quantity for d in deliveries[m]]
            for m in range(len(self.transporter))
        ]
        
        omitted = \
            np.sum([p.quantity for p in self.packages])\
            - \
            np.sum(quantities)
        omission_penalty = self.omission_cost*omitted
        
        # for k in range(len(actions)):
        #     if actions[k]:
        #         nodes[actions[k]-1].append(self.packages[k].destination)
        #         quantities[actions[k]-1].append(self.packages[k].quantity)
        #     else:
        #         omission_penalty += self.omission_cost*self.packages[k].quantity
        #         omitted += 1
        # # print(nodes)
        
        if self.mask is None:
            if np.sum(actions) == self.num_packages:
                l = [self.hub] + list(nodes[0])
                self.mask = np.ix_(l, l)
            else:
                raise("You have to call the OR-routing at least once with all packages included !")
        
        sol = None
        
        if call_OR:
            for m in range(len(self.transporter)):#TODO parallelize
                #TODO for the TSP case
                sol = self.transporter[m].compute_cost(nodes[m], quantities[m], time_budget)
                
                if np.sum(actions) == self.num_packages:
                    self.solutions[m] = sol
        
        else:#TODO for the TSP case
            
            
            omitted_packages = np.where(actions == 0)[0] + 1 # important to add 1 to ignore the hub's index 0
            
            
            sol = [
                [
                    self.solutions[0][m][i]
                    for i in range(len(self.solutions[0][m]))
                    if self.solutions[0][m][i] not in omitted_packages
                ]
                for m in range(len(self.solutions[0]))
            ]
            
        distance_matrix = self.distance_matrix[self.mask]
        time_matrix = self.time_matrix[self.mask]
        distance = np.zeros(self.num_vehicles)
        time = np.zeros(self.num_vehicles)
        for m in range(len(sol)):
            for i in range(len(sol[m])-1):
                distance[m] += distance_matrix[sol[m][i], sol[m][i+1]]
                time[m] += time_matrix[sol[m][i], sol[m][i+1]]
                
        # mask_is_None = self.mask is None
        # if mask_is_None:
        #     self.mask = np.zeros(self.num_packages+1)
            
        # sol, distance, time, omitted, self.mask = _aux(
        #     actions,
        #     self.packages,
        #     self.mask,
        #     self.num_vehicles,
        #     time_budget,
        #     mask_is_None,
        #     call_OR,
        #     self.distance_matrix,
        #     self.time_matrix,
        #     self.solutions,
        #     self.transporter,
        #     self.num_packages,
        # )
        omission_penalty = self.omission_cost*omitted
        
        
        total_costs =     np.sum(self._get_costs(distance, time))
        total_emissions = np.sum(self._get_emissions(distance, time))
            
            
            
        self.info['solution_found'] = np.any(time != 0)
        self.info['costs'] = total_costs
        self.info['time_per_vehicle'] = time
        self.info['distance_per_vehicle'] = distance
        self.info['excess_emission'] = total_emissions - self.Q
        self.info['omitted'] = omitted
        self.info['solution'] = self.solutions if sol is None else sol
            
        return total_costs + max(0, total_emissions - self.Q)*self.CO2_penalty + omission_penalty
        
    
    # def _get_state(self, random_qantity=False):
    #     # print('ids are : ', self.ids)
    #     if self.t == 0:
    #         self.obs = {
    #             i : np.zeros(5)
    #             for i in self.ids
    #         }
        
    #     # The destination of the client
    #     # node = self.available_nodes.pop(np.random.randint(len(self.available_nodes)))
        
    #     #Compute the manhattan distance between the hubs and the destination
    #     # position_node = np.array([node//self.size, node%self.size]) 

    #     # compute the quantity
    #     if random_qantity :
    #         quantity = np.random.randint(self.max_capacity)
    #     else:
    #         quantity = 1
            
    #     #Construct the state
    #     for i in self.obs.keys():
    #         self.obs[i][0] = self.t
    #         # self.obs[i][self.t] = node
    #         self.obs[i][-3] = quantity

    #     return self.obs
    
    
    def reset(self, packages = None, seed = None):
        
        np.random.seed(seed)
        
        # if num_packages is None:
        #     num_packages = self.max_capacity * self.num_vehicles
        # else:
        assert self.num_packages <= self.max_capacity * self.num_vehicles
            
        # super().reset(seed=seed)
        if packages is None:
            destinations = np.sort(np.random.choice(
                [i for i,_ in enumerate(self.G.nodes) if i!=self.hub], size=self.num_packages, replace=False
            ))

            self.packages = [
                Package(
                    destination=d,
                    quantity=1,#TODO
                )
                for d in destinations
            ]
        else:
            assert len(packages) == self.num_packages
            self.packages = sorted(packages, key=lambda p : p.destination)
            assert self.total_capacity >= np.sum([
                p.quantity
                for p in packages
            ])
        
        for transporter in self.transporter:
            transporter.reset()
            
        # self.G_ncolors = ['#1f78b4' for _ in range(len(self.G.nodes))]
        # self.G_ncolors[self.transporters_hubs[0]] = 'red'
        # self.G_ncolors[self.transporters_hubs[1]] = 'green'
        
        # self.available_nodes = list(range(len(self.G_ncolors)))
        # self.available_nodes.remove(self.transporters_hubs[0])
        # self.available_nodes.remove(self.transporters_hubs[1])
        
        self.t = 0
        self.info = dict()
        
        # self.num_packages = num_packages
        
        return self.info
    
    def _get_emissions(self, d, t):
        return self.emissions_KM*d
    
    def _get_costs(self, d, t):
        return self.costs_KM*d
        
    def _get_rewards(self, actions, time_budget, call_OR):
        #print('obs is :', self.obs)
        #print('ids is :', self.ids)
        
        costs = self._compute_cost(actions, time_budget, call_OR)
        
        # assert rewards[str(winner)]>=0
        
        return -costs
    
        
    def step(self, actions, time_budget = 1, call_OR = True):
        self.t += 1
        r = self._get_rewards(actions, time_budget, call_OR)
        
        done = self.info['excess_emission'] <= 0

        return r, done, self.info


# @njit
# def sol_to_routes(d, sol : List, n_vehicles, max_capacity):
#     routes = np.zeros((n_vehicles, 2*max_capacity-1))
#     for i in range(len(sol)):
#         for j in range(len(sol[i])-1):
#             routes[i, 2*j] = sol[i][j]
#             routes[i, 2*j + 1] = d[sol[i][j], sol[i][j+1]]

@njit
def get_d_t(
    a,
    distance_matrix,
    costs_matrix,
    time_matrix,
    initial_solution,
    quantities,
    omission_cost,
    n_vehicles,
    ):
    
    omitted = np.where(a == 0)[0]
    omission_penalty = omission_cost*np.sum(quantities[omitted])
    
    distance = np.zeros(n_vehicles)
    time = np.zeros(n_vehicles) #TODO
        
    omitted += 1 # important to add 1 to ignore the hub's index 0
    
    routes = np.zeros(initial_solution.shape)
    
    for i in range(len(initial_solution)):
        j = 0
        k = 0
        l = 2
        
        routes[i, k] = initial_solution[i, j]
        # distance[i] += distance_matrix[
        #     int(initial_solution[i, 0]),
        #     int(initial_solution[i, l]),
        # ]
        # print(initial_solution)
        while k<len(routes[i]):
            if initial_solution[i][l] not in omitted:
                # print(initial_solution[i, j])
                routes[i, k+1] = costs_matrix[i][#distance_matrix[
                    int(initial_solution[i, j]),
                    int(initial_solution[i, l]),
                ]
                distance[i] += distance_matrix[
                    int(initial_solution[i, j]),
                    int(initial_solution[i, l]),
                ]
                if initial_solution[i, l] == 0:
                    break
                j = l
                # routes[i, k+2] = initial_solution[i, l]
                k += 2
                routes[i, k] = initial_solution[i, j]
                
            l += 2
        # routes[i, k+1] = costs_matrix[i][#distance_matrix[
        #     int(initial_solution[i, l-2]),
        #     int(initial_solution[i, l]),
        # ]
        # distance[i] += distance_matrix[
        #     int(initial_solution[i, l-2]),
        #     int(initial_solution[i, l]),
        # ]
        # print(routes)
            
    #     for j in range(len(sol[i])-1):
    #         routes[i, 2*j + 1] = distance_matrix[sol[i][j], sol[i][j+1]]
    # sol = [
    #     [
    #         initial_solution[m][i]
    #         for i in range(len(initial_solution[m]))
    #         if initial_solution[m][i] not in omitted
    #     ]
    #     for m in range(len(initial_solution))
    # ]
        
    
    # for m in range(n_vehicles):
    #     distance[m] = sum([initial_solution[m, j] for j in range(1, len(initial_solution[m]), 2)])
            # time[m] += time_matrix[sol[m][i], sol[m][i+1]]
            
    return routes, distance, time, omitted, omission_penalty

@njit
def get_elimination_gain(cost_matrix, routes, obs_shape, normalize):
    observation = np.zeros(obs_shape)
    for m in range(len(routes)):
        j = 0
        while not (j > 0 and routes[m, j] == 0):
            if routes[m, j] != 0:
                observation[int(routes[m, j])-1] = routes[m, j-1] + routes[m, j+1] -(
                        cost_matrix[m, int(routes[m, j-2]), int(routes[m, j+2])]
                )
            j+=2
    if normalize:
        observation = np.clip(observation / np.max(observation), 0, 1)
    return observation


class AssignmentEnv(gym.Env):
    def __init__(self, 
                 game : AssignmentGame = None, 
                 saved_routes = None,
                 saved_dests = None,
                 obs_mode = 'routes', # possible values ['multi', 'cost_matrix','routes', 'action', 'elimination_gain', 'assignment', 'game']
                 change_instance = True,
                 instance_id = 0,
                 ):
        
        super().__init__()
        
        if game is None:
            self._game = AssignmentGame()
        else:            
            self._game = game
        
        self.num_actions = 2
        self.K = self._game.num_packages
        self.omission_cost = self._game.omission_cost
        
        d = len(self._game.distance_matrix)
        if obs_mode == 'cost_matrix':
            self.obs_dim = (d, d)
            
        elif obs_mode == 'multi':
            self.obs_dim = {
                "costs" : (1, d, d),
                "other" : (self._game.max_capacity + 2)*self._game.num_vehicles + 1
            }
            
        elif obs_mode == 'routes':
            self.obs_dim = ((2*(self._game.max_capacity+2)-1)*self._game.num_vehicles) + 2*self._game.num_packages +1
            
        elif obs_mode == 'action' or obs_mode == 'elimination_gain' or obs_mode == 'assignment' or obs_mode == 'game':
            self.obs_dim = self._game.num_packages 
            
        else:
            raise("The given obs mode is not recognised !")
        
        self.obs_mode = obs_mode
        if saved_routes is not None:
            assert saved_dests is not None
            assert len(saved_routes) == len(saved_dests)
            self.order = np.arange(len(saved_dests), dtype=int)
            if change_instance:
                np.random.shuffle(self.order)
            
        self.change_instance = change_instance
        self.saved_routes = saved_routes
        self.saved_dests = saved_dests
        # self.new_routes = []
        # self.new_dests = []
        self.reset_counter = instance_id
        # print(self.obs_dim)
        
        
        if obs_mode == 'action':
            self.observation_space = gym.spaces.MultiBinary(self._game.num_packages)
        elif obs_mode == 'assignment':
            self.observation_space = gym.spaces.MultiDiscrete(
                (self._game.num_vehicles+1)*np.ones(d, dtype=int)
            )
        elif obs_mode == 'multi':
            d = self._game.num_packages + 1
            self.observation_space = gym.spaces.Dict({
                "costs" : gym.spaces.Box(0, 1, (1, d, d,), np.float64),
                "other" : gym.spaces.Box(0, 1, (self.obs_dim["other"],), np.float64)
            })
        elif obs_mode == 'cost_matrix':
            self.observation_space = gym.spaces.Box(0, 1, (1, d, d,), np.float64)
        elif obs_mode == 'elimination_gain':
            self.observation_space = gym.spaces.Box(0, 1, (self.obs_dim,), np.float64)
        else:
            self.observation_space = gym.spaces.Box(0, 1e10, (self.obs_dim,), np.float64)
        self.action_space = gym.spaces.MultiBinary(self._game.num_packages)
        
    def reset(self, 
              seed: int | None = None,
              packages = None,
              time_budget = 1,
              *args,
              **kwargs,
              ) -> tuple[np.ndarray, dict[str, Any]]:
        
        if self.saved_routes is not None:
            # np.zeros((self._game.num_vehicles, 2*(self._game.max_capacity+2)-1))
            if self.reset_counter == len(self.saved_routes):
                self.reset_counter = 0
                # self.saved_dests = np.concatenate([
                #     self.saved_dests,
                #     np.array(self.new_dests)
                # ])
                # self.saved_routes = np.concatenate([
                #     self.saved_routes,
                #     np.array(self.new_routes)
                # ])
                # self.new_routes = []
                # self.new_dests = []
                np.random.shuffle(self.order)
                assert self.saved_routes.shape[1:] == (self._game.num_vehicles, 2*(self._game.max_capacity+2)-1)
                # name = str(time())
                # np.save(name + '_routes', np.array(self.new_routes))
                # np.save(name + '_dests', np.array(self.new_dests))
                
            self.destinations = np.array(self.saved_dests[self.order[self.reset_counter]], dtype=int)
            # self.destinations.sort()
            packages = [
                Package(
                    destination=d,
                    quantity=1,#TODO
                )
                for d in self.destinations
            ]
        
        self._game.reset(packages, seed)
        
        if self.saved_routes is None:# or self.reset_counter%500 == 0:
            self.destinations = np.array([
                p.destination for p in self._game.packages
            ])
        
        
        self.quantities = np.array([
            p.quantity for p in self._game.packages
        ])
        
        # Costs + emission penalty
        l = [self._game.hub] + list(self.destinations)
        self.mask = np.ix_(l, l)
        self.distance_matrix = self._game.distance_matrix[self.mask]
        self.costs_matrix = np.array([
            (self._game.costs_KM[m] + self._game.CO2_penalty*self._game.emissions_KM[m])*self.distance_matrix
            for m in range(len(self._game.costs_KM))
        ])# TODO for more complexe cost functions
        
        self.time_matrix = self._game.time_matrix[self._game.mask]
        
        a = np.ones(self._game.num_packages, dtype=int)
        if self.saved_routes is None:# or self.reset_counter%500 == 0:
            *_, info = self._game.step(
                a,
                time_budget=time_budget,
            )
            
        else:
            self.initial_routes = self.saved_routes[self.order[self.reset_counter]]
            _, distance, _, omitted, _ = get_d_t(
                a,
                self.distance_matrix,
                self.costs_matrix,
                self.time_matrix,
                self.initial_routes,
                self.quantities,
                self._game.omission_cost,
                self._game.num_vehicles,
            )
            
            total_costs =     np.sum(self._game._get_costs(distance, time))
            total_emissions = np.sum(self._game._get_emissions(distance, time))
            info = dict()
            info['solution_found'] = np.any(time != 0)
            info['costs'] = total_costs
            info['time_per_vehicle'] = time
            info['distance_per_vehicle'] = distance
            info['excess_emission'] = total_emissions - self._game.Q
            info['omitted'] = omitted
            
            # info['solution'] = self.solutions if sol is None else sol
        
        
        # self.initial_routes = List()
        # for lst in self._game.solutions[0]:
        #     l = List()
        #     for e in lst:
        #         l.append(e)
        #     self.initial_routes.append(l)
        
        
        
        # print( 2*(self._game.max_capacity) - 1)
        if self.saved_routes is None:# or self.reset_counter%500 == 0:
            self.initial_routes = np.zeros((self._game.num_vehicles, 2*(self._game.max_capacity+2)-1))
            for i in range(len(self._game.solutions[0])):
                for j in range(len(self._game.solutions[0][i])-1):
                    self.initial_routes[i, 2*j] = self._game.solutions[0][i][j]
                    self.initial_routes[i, 2*j + 1] = self.costs_matrix[i][self._game.solutions[0][i][j], self._game.solutions[0][i][j+1]]
                    #self._game.distance_matrix[self._game.solutions[0][i][j], self._game.solutions[0][i][j+1]]
                    
            # self.new_routes.append(deepcopy(self.initial_routes))
            # self.new_dests.append(deepcopy(self.destinations))
                    
        # else:
        #     self.initial_routes = self.saved_routes[self.order[self.reset_counter]]
        
        if self.change_instance:
            self.reset_counter += 1
            
        if self.obs_mode == 'cost_matrix' or self.obs_mode == 'multi':
            
            M = np.zeros((1, *self.distance_matrix.shape))
            # tmp = [self._game.hub] + list(self.destinations)
            
            for m in range(len(self.initial_routes)):
                l = []
                # ll = []
                for i in range(0, len(self.initial_routes[m]), 2):
                    # d.add(env.initial_routes[l, 2*i])
                    if self.initial_routes[m, i]:
                        # l.append(int(self.destinations[int(self.initial_routes[m, i])-1]))
                        l.append(int(self.initial_routes[m, i])-1)
                # M[0][np.ix_(l, tmp)] = self.costs_matrix[m][l, :]
                M[0][l, :] = self.costs_matrix[m][l, :]
                M[0][:, l] = self.costs_matrix[m][:, l]
                # M[0][np.ix_(tmp, l)] = self.costs_matrix[m][:, l]
            # del tmp
            M[0] = normalize(M[0])
            if self.obs_mode == 'cost_matrix':
                self.observation = M#np.array(M*255, dtype=np.uint8)
            else:
                self.observation = self.observation_space.sample()
                self.observation["costs"] = M
                self.observation['other'][:-1] = np.array([
                    self.initial_routes[m, j] 
                    for m in range(len(self.initial_routes))
                    for j in range(0, len(self.initial_routes[m]), 2)
                ])
                self.observation['other'][-1] = info['excess_emission']
                self.observation['other'] /= np.max(self.observation['other'])
                
        if self.obs_mode == 'routes':#TODO finish the work
            self.observation = np.reshape(np.concatenate([
                self.initial_routes.reshape(-1),
                # self.destinations, #destinations for each package
                self.quantities, #quantites for each package
                [
                  info['excess_emission']
                ], #complemantary informations
                a, # actions
            ]), (-1))
            
        if self.obs_mode == 'action':
            self.observation = a
        elif self.obs_mode == 'assignment':
            self.observation = np.zeros(self.observation_space.shape, dtype=int)
            for m in range(len(self.initial_routes)):
                for j in range(0, len(self.initial_routes[m]), 2):
                    if self.initial_routes[m, j]:
                        self.observation[int(self.destinations[int(self.initial_routes[m, j])-1])] = m+1
            
        if self.obs_mode == 'elimination_gain':
            self.observation = get_elimination_gain(
                self.costs_matrix, self.initial_routes, self.observation_space.shape, True
            )#np.zeros(self.observation_space.shape)
            # for m in range(len(self.initial_routes)):
            #     j = 0
            #     while not (j > 0 and self.initial_routes[m, j] == 0):
            #         if self.initial_routes[m, j] != 0:
            #             self.observation[int(self.initial_routes[m, j])-1] = self.initial_routes[m, j-1] + self.initial_routes[m, j+1] -(
            #                     self.costs_matrix[m, int(self.initial_routes[m, j-2]), int(self.initial_routes[m, j+2])]
            #             )
            #         j+=2
            # self.observation /= np.max(self.observation)
            
        if self.obs_mode == 'game':
            self.observation = get_elimination_gain(
                self.costs_matrix, self.initial_routes, self.observation_space.shape, False
            )
            
        if self.obs_mode != 'multi':
            assert self.observation.shape == self.observation_space.shape
        else:
            assert self.observation["costs"].shape == self.observation_space["costs"].shape
            assert self.observation["other"].shape == self.observation_space["other"].shape
        
        # print(self.obs_dim)
        # print(self.observation.shape)
            
        return self.observation.copy(), info
    
    
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        
        info = dict()
        # self.initial_routes = List()
        # for lst in self._game.solutions[0]:
        #     self.initial_routes.append(lst)
        
            
            
        routes, distance, time, omitted, omission_penalty = get_d_t(
            action,
            self.distance_matrix,
            self.costs_matrix,
            self.time_matrix,
            self.initial_routes,
            self.quantities,
            self._game.omission_cost,
            self._game.num_vehicles,
        )
        
        if self.obs_mode == 'elimination_gain':
            self.observation = get_elimination_gain(
                self.costs_matrix, routes, self.observation_space.shape, True
            )#np.zeros(self.observation_space.shape)
            # for m in range(len(routes)):
            #     j = 0
            #     while not (j > 0 and routes[m, j] == 0):
            #         if self.initial_routes[m, j] != 0:
            #             self.observation[int(routes[m, j])-1] = routes[m, j-1] + routes[m, j+1] -(
            #                     self.costs_matrix[m, int(routes[m, j-2]), int(routes[m, j+2])]
            #             )
            #         j+=2
            # self.observation /= np.max(self.observation)
            
        if self.obs_mode == 'action' and self.obs_mode == 'routes' :
            self.observation[-len(action):] = action
        if self.obs_mode == 'routes':
            self.observation[:routes.size] = routes.reshape(-1)
        
            
        # if self.obs_mode == 'elimination_gain':
        #     self.observation[action == 0] = 0.
            
        # print(self.observation.shape)
        costs = self._game._get_costs(distance, time)
        emissions = self._game._get_emissions(distance, time)
        total_costs =     np.sum(costs)
        total_emissions = np.sum(emissions)
        
        if self.obs_mode == 'game':
            alpha = np.zeros(self._game.num_packages, dtype=int)
            for m in range(len(routes)):
                for j in range(0, len(routes[m]), 2):
                    if routes[m, j]:
                        alpha[int(routes[m, j])-1] = m+1
            self.observation = get_elimination_gain(
                self.costs_matrix, self.initial_routes, self.observation_space.shape, False
            )
        
            info['LCF'] = np.concatenate([[0], costs + emissions*self._game.CO2_penalty])
            info['GCF'] = np.sum(info['LCF'])
            r = self.observation.copy() + np.maximum(0, info['LCF'][alpha] - info['GCF']/self._game.num_packages) + (alpha == 0)*self._game.omission_cost
            
        #total_costs + max(0, total_emissions - self._game.Q)*self._game.CO2_penalty
        # + omission_penalty
        
        info['solution_found'] = np.any(time != 0)
        info['costs'] = total_costs
        info['time_per_vehicle'] = time
        info['distance_per_vehicle'] = distance
        info['excess_emission'] = total_emissions - self._game.Q
        info['omitted'] = omitted
        # info['solution'] = self.solutions if sol is None else sol
        if self.obs_mode == 'routes':
            self.observation[-len(action)-1] = info['excess_emission']
            
        if self.obs_mode == 'multi':
            self.observation['other'][:-1] = np.array([
                    routes[m, j] 
                    for m in range(len(routes))
                    for j in range(0, len(routes[m]), 2)
            ])
            self.observation['other'][-1] = info['excess_emission']
            self.observation['other'] /= np.max(self.observation['other'])
        
        if self.obs_mode != 'game':
            r = -(total_costs + max(0, total_emissions - self._game.Q)*self._game.CO2_penalty + omission_penalty)
        else:
            info['r'] = -(total_costs + max(0, total_emissions - self._game.Q)*self._game.CO2_penalty + omission_penalty)
        done = bool(info['excess_emission']<=0)
        # print(type(done))
        
        return self.observation.copy(), r, done, done, info

    
class AlterActionEnv(gym.Env):
    
    def __init__(self, 
                 game : AssignmentGame = None, 
                 H = 500,
                 rewards_mode = 'heuristic'
                 ):
        
        # super().__init__()
        
        self._env = AssignmentEnv(game)
        self.rewards_mode = rewards_mode
        self.observation_space = self._env.observation_space
        self.action_space = gym.spaces.Discrete(self._env._game.num_packages)
        self.action = np.ones(self._env._game.num_packages, dtype=int)
        
        self.H = H
        
    # @property
    # def invalid_actions(self):
    #     return np.where(self.action == 0)[0]
    
    # @property
    # def n_invalid_actions(self):
    #     return self.action.size - np.sum(self.action)
    
    # def action_masks(self):
    #     return self.action == 1
        
    def reset(self, *args, **kwargs
              ) -> tuple[np.ndarray, dict[str, Any]]:
        
        self.action = np.ones(self._env._game.num_packages, dtype=int)
        self.t = 0

        return self._env.reset(*args, **kwargs)
    
    def step(self, a: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        
        self.action[a] = 0 if self.action[a] else 1
        self.t += 1
        obs, r, d, _, info = self._env.step(self.action)
        done = d or bool(self.t > self.H)
        
        if self.rewards_mode == 'terminal':
            r *= float(done)
        return obs, r, done, done, info
    
    
class RemoveActionEnv(gym.Env):
    
    def __init__(self, 
                 rewards_mode = 'normalized_terminal',
                 action_mode = 'destinations', # possible values ['destinations', 'all_nodes']
                 *args,
                 **kwargs,
                #  game : AssignmentGame = None,
                #  saved_routes = None,
                #  saved_dests = None,
                #  obs_mode = 'routes'
                 ):
        
        super().__init__()
        
        self._env = AssignmentEnv(*args, **kwargs)#game, saved_routes, saved_dests, obs_mode)
        self.rewards_mode = rewards_mode
        # self.action = np.ones(self._env._game.num_packages, dtype=int)
        
        if action_mode == 'all_nodes':
            self.action_mask = np.zeros(self._env._game.grid_size**2, dtype=bool)
            self.action_space = gym.spaces.Discrete(len(self.action_mask))
            if self._env.obs_mode == 'cost_matrix':
                self.observation_space = self._env.observation_space
            else:
                self.observation_space = gym.spaces.MultiBinary(len(self.action_mask))
                
        else:
            if self._env.obs_mode == 'cost_matrix':
                d = self._env._game.num_packages + 1
                self.observation_space = gym.spaces.Box(0, 1, (1, d, d,), np.float64)
            else:
                self.observation_space = self._env.observation_space
            self.action_space = gym.spaces.Discrete(self._env._game.num_packages)
        self.invalid_actions = []
        self.n_invalid_actions = 0
        
        self.action_mode = action_mode
        self.H = self._env._game.num_packages
        
    # @property
    # def invalid_actions(self):
    #     return np.where(self.action == 0)[0]
    
    # @property
    # def n_invalid_actions(self):
    #     return self.action.size - np.sum(self.action)
    
    def action_masks(self):
        if self.action_mode == 'all_nodes':
            return self.action_mask
        return self.action == 1
        
    def reset(self, *args, **kwargs
              ) -> tuple[np.ndarray, dict[str, Any]]:
        
        obs, info = self._env.reset(*args, **kwargs)
        self.obs = obs.copy()
        if self.action_mode == 'destinations' and self._env.obs_mode == 'cost_matrix':
            self.obs = self.obs[0][self._env.mask].reshape(self.observation_space.shape)
            obs = self.obs
        
        self.destinations = np.array(self._env.destinations, dtype=np.int16)
        if self.action_mode == 'all_nodes':
            self.action_mask = np.zeros(self._env._game.grid_size**2, dtype=bool)
            self.action_mask[self.destinations] = True
            if self._env.obs_mode != 'cost_matrix':
                obs = self.action_mask.astype(int)
            
        self.action = np.ones(self._env._game.num_packages, dtype=int)
        self.t = 0
        self.invalid_actions = []
        self.n_invalid_actions = 0

        return obs, info
    
    def step(self, a: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        
        self.action[a] = 0
        self.invalid_actions.append(a)
        self.n_invalid_actions += 1
        self.t += 1
        obs, r, d, _, info = self._env.step(self.action)
        
        if self._env.obs_mode == 'cost_matrix':
            self.obs[0, a, :] = 0
            self.obs[0, :, a] = 0
            # ii = np.array(a).reshape(-1)
            # self.obs[0][np.ix_(ii, ii)] = 0
            # print(a)
            # print(self.obs[0][np.ix_(ii, ii)])
            # print(20*'-')
            # del ii
        elif self._env.obs_mode == 'elimination_gain':
            # self.obs[a] = self._env.observation#0.
            self.obs = self._env.observation#0.
            # self._env.observation = self.obs
        elif self._env.obs_mode == 'assignment':
            self.obs[a] = 0
            self._env.observation = self.obs

        if self.action_mode == 'all_nodes':
            self.action_mask[a] = False
            a = (self.destinations[:, None] == a).argmax(axis=0)
        
        if self.action_mode == 'all_nodes':
            obs = self.action_mask.astype(int)
        else:
            obs = self.obs
            
        done = d or bool(self.t > (self.H-1))
        
        normalizer_const = self._env._game.num_packages*self._env._game.omission_cost
        
        if self.rewards_mode == 'terminal':
            r = float(done)*(r+normalizer_const)
            
        elif self.rewards_mode == 'normalized_terminal':
            r = np.clip(float(done)*(r+normalizer_const)/normalizer_const, 0, 1)
            
        elif self.rewards_mode == 'penalize_length':
            r = -float(not done) + float(done)*10
        
            
        return obs, r, done, done, info
    
class CombActionEnv(gym.Env):
    
    def __init__(self, 
                 game : AssignmentGame = None, 
                 rewards_mode = 'heuristic',
                 *args,
                 **kwargs
                 ):
        
        super().__init__()
        
        self._env = AssignmentEnv(game)
        self.rewards_mode = rewards_mode
        self.observation_space = self._env.observation_space
        self.action_space = gym.spaces.MultiBinary(self._env._game.num_packages)
        # self.action = np.ones(self._env._game.num_packages, dtype=int)
        # self.invalid_actions = []
        # self.n_invalid_actions = 0
        
        self.H = self.action_space.n
        
    # @property
    # def invalid_actions(self):
    #     return np.where(self.action == 0)[0]
    
    # @property
    # def n_invalid_actions(self):
    #     return self.action.size - np.sum(self.action)
    
    # def action_masks(self):
    #     return self.action == 1
        
    def reset(self, *args, **kwargs
              ) -> tuple[np.ndarray, dict[str, Any]]:
        
        # self.action = np.ones(self._env._game.num_packages, dtype=int)
        self.t = 0
        # self.invalid_actions = []
        # self.n_invalid_actions = 0

        return self._env.reset(*args, **kwargs)
    
    def step(self, a: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        
        # self.action[a] = 0
        # self.invalid_actions.append(a)
        # self.n_invalid_actions += 1
        self.t += 1
        obs, r, d, _, info = self._env.step(a)
        done = d or bool(self.t > (self.H-1))
        
        if self.rewards_mode == 'terminal':
            r = float(done)*(r+2e4)
            
        elif self.rewards_mode == 'normalized_terminal':
            r = float(done)*(r+2e4)/2e4
            
        elif self.rewards_mode == 'penalize_length':
            r = -float(not done)
            
        return obs, r, done, done, info

# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-10, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
        
def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class NormalizedEnv(gym.core.Wrapper):
    def __init__(self, env, ret=False, clipob=1., cliprew=1., gamma=0.99, epsilon=1e-8):
        super(NormalizedEnv, self).__init__(env)
        self.routes_size = (self.env._env._game.num_vehicles)*(2*(self.env._env._game.max_capacity+2)-1)
        self.ob_rms = RunningMeanStd(shape=(self.routes_size,))
        self.ret_rms = RunningMeanStd(shape=(1,)) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(())
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = self.env._env._game.Q
        self.observation_space = gym.spaces.Box(0, 1, (self.env._env.observation_space.shape[-1],), np.float64)
        self.action_masks = self.unwrapped.action_masks
        


    def step(self, action):
        obs, rews, d, t, info = self.env.step(action)
        info['real_reward'] = rews
        # print("before", self.ret)
        self.ret = self.ret * self.gamma + rews
        # print("after", self.ret)
        obs[:self.routes_size] = self._obfilt(obs[:self.routes_size])
        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret].copy()))
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), 0, self.cliprew)
        self.ret = self.ret * (1-float(d))
        
        obs[-len(self.env.action)-1] = np.clip(
            info['excess_emission']/self.Q,
            0, 1
        )
        return obs, rews, d, t, info

    def _obfilt(self, obs):
        self.ob_rms.update(obs)
        obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), 0, self.clipob)
        return obs

    def reset(self, *args, **kwargs):
        self.ret = np.zeros(())
        obs, info = self.env.reset(*args, **kwargs)
        obs[:self.routes_size] = self._obfilt(obs[:self.routes_size])
        
        obs[self.routes_size:-len(self.env.action)-1] = preprocessing.normalize(
                [obs[self.routes_size:-len(self.env.action)-1]]
        )[0]
        obs[-len(self.env.action)-1] = np.clip(
            info['excess_emission']/self.Q,
            0, 1
        )
        return obs, info
    
    
class GameEnv(gym.Env):
    def __init__(self, 
                 env : AssignmentEnv = None,
                 is_0_allowed = False,
                #  saved_routes = None,
                #  saved_dests = None,
                #  change_instance = True,
                #  instance_id = 0,
                 ):
        
        self._env = env
        self.K = self._env._game.num_packages
        self.omission_cost = self._env._game.omission_cost
        self.CO2_penalty = self._env._game.CO2_penalty
        self.Q = self._env._game.Q
        self.emissions_KM = self._env._game.emissions_KM
        self.costs_KM = self._env._game.costs_KM
        self.max_capacity = self._env._game.max_capacity
        self.is_0_allowed = is_0_allowed
        self.num_actions = len(self.emissions_KM) + 1 if is_0_allowed else len(self.emissions_KM)
        # self.instance_id = instance_id
        # self.saved_routes = saved_routes
        # self.saved_dests = saved_dests
        # self.change_instance = change_instance
        
    
    def reset(self):
        res = self._env.reset()
        self.distance_matrix = self._env.distance_matrix
        self.cost_matrix = self._env.costs_matrix
        
        return res
    
    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        info = dict()
        routes = []
        costs = np.zeros(len(self.cost_matrix))
        emissions = np.zeros(len(self.cost_matrix))
        info['LCF'] = np.zeros(len(self.cost_matrix))
        obs = np.zeros(action.shape)
        
        if not self.is_0_allowed:
            action += 1
        
        a = action.copy()
        
        for i in range(len(self.cost_matrix)+1):
            routes.append([])
            alpha = list(np.where(action == i)[0] + 1)
            if i:
                routes[i].append(0)
                while True:
                    if len(alpha) == 0:
                        break
                    if len(routes[i]) > self.max_capacity:
                        # routes[0] += alpha
                        info['LCF'][i-1] += self.omission_cost
                        a[np.array(alpha) - 1] = 0
                        # break
                    
                    j = np.argmin(self.cost_matrix[i-1, routes[i][-1], alpha])
                    dest = alpha.pop(j)
                    costs[i-1] += self.distance_matrix[routes[i][-1], dest]*self.costs_KM[i-1]
                    emissions[i-1] += self.distance_matrix[routes[i][-1], dest]*self.emissions_KM[i-1]
                    info['LCF'][i-1] += self.cost_matrix[i-1, routes[i][-1], dest]
                    routes[i].append(dest)
                    if len(routes[i]) > 2:
                        obs[routes[i][-2] - 1] = self.cost_matrix[i-1, routes[i][-3], routes[i][-2]] + self.cost_matrix[i-1, routes[i][-2], routes[i][-1]] - (
                            self.cost_matrix[i-1, routes[i][-3], routes[i][-1]]
                        )
                    
                costs[i-1] += self.distance_matrix[routes[i][-1], 0]*self.costs_KM[i-1]
                emissions[i-1] += self.distance_matrix[routes[i][-1], 0]*self.emissions_KM[i-1]
                info['LCF'][i-1] += self.cost_matrix[i-1, routes[i][-1], 0]
                routes[i].append(0)
                if len(routes[i]) > 2:
                    obs[routes[i][-2] - 1] = self.cost_matrix[i-1, routes[i][-3], routes[i][-2]] + self.cost_matrix[i-1, routes[i][-2], routes[i][-1]] - (
                        self.cost_matrix[i-1, routes[i][-3], routes[i][-1]]
                    )
                    
            else:
                routes[i] += alpha
        
        # info['LCF'] = np.concatenate([[0], costs + emissions*self.CO2_penalty])
        # info['GCF'] = np.sum(info['LCF'])
        
        # r = obs + np.maximum(0, info['LCF'][action] - info['GCF']/self.K) + (action == 0)*self.omission_cost
        if not self.is_0_allowed:
            action -= 1
            
        r = obs/np.max(obs) + (info['LCF'][action] - np.min(info['LCF']))/(np.max(info['LCF']) - np.min(info['LCF']))
        
        # normalizer_const = self.K*self.omission_cost
            
        total_emission = np.sum(emissions)
        info['r'] = -(np.sum(costs) + max(0, total_emission - self.Q)*self.CO2_penalty + np.sum(a == 0)*self.omission_cost)
        info['a'] = a
        info['routes'] = routes
        # info['r'] = np.clip((normalizer_const + info['r'])/normalizer_const, 0, 1)
        
        return obs, r, total_emission <= self.Q, False, info
    
def test_assignment_game(game = None, K = 500, log = True, plot = True):
    
    if game is None:
        game = AssignmentGame(
                Q=0,
                grid_size=45,
                max_capacity=125,
                K=K,
            )
    # game = AssignmentGame()
    rewards = []
    game.reset()
    
    done = False
    i = 0
    actions = np.ones(game.num_packages, dtype=int)
    r, done, info = game.step(actions)
    while not done:
        rewards.append(r)
        if log:
            print(info)
            print(info['solution'])
        actions[i] = 0
        i += 1
        if i == K:
            break
        r, done, info = game.step(actions, call_OR=False)
        
    if log:
        print('rewards', rewards)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(rewards)
        plt.show()
        

def test_assignment_AlterAction_env(game = None, K = 500, log = True, plot = True):
    if game is None:
        game = AssignmentGame(
                Q=0,
                grid_size=45,
                max_capacity=125,
                K = K
            )
    env = AlterActionEnv(game)
    rewards = []
    env.reset()
    # K = env._game.num_packages
    
    
    done = False
    i = 0
    while not done:
        # actions[2] = 0
        action = env.action_space.sample()
        
        _, r, done, _, info = env.step(action)
        # done = True
        i += 1
        if i > env.H:
            print(env.t)
            assert done
        rewards.append(r)
        if log:
            print(info)

    if log:
        print('rewards', rewards)
        
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(rewards)
        plt.show()

def test_assignment_env(game = None, K = 500, log = True, plot = True):
    if game is None:
        game = AssignmentGame(
                Q=0,
                grid_size=45,
                max_capacity=125,
                K = K
            )
    env = AssignmentEnv(game)
    rewards = []
    env.reset()
    K = env._game.num_packages
    
    
    done = False
    i = 0
    actions = np.ones(game.num_packages, dtype=int)
    while not done:
        # actions[2] = 0
        _, r, done, _, info = env.step(actions)
        # done = True
        actions[i] = 0
        i += 1
        if i >= K:
            break
        rewards.append(r)
        if log:
            print(info)

    if log:
        print('rewards', rewards)
        
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(rewards)
        plt.show()
if __name__ == '__main__':
    game = AssignmentGame(
            Q=0,
            K = 100,
            grid_size=25,
            max_capacity=125
        )
    
    # env = AssignmentEnv(game)
    # print(env.obs_dim)
    
    # test_assignment_game(game, log = False)
    # print('game ok!')
    # test_assignment_env(game, log = False)
    test_assignment_AlterAction_env(game, log = False)
    print('env ok!')