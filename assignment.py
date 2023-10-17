from dataclasses import dataclass
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ortools
from numba import njit

from transporter import Transporter

from typing import Any, List, Dict, Optional

import gymnasium as gym
from ray.rllib.env import MultiAgentEnv

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
                 CO2_penalty = 1_000,
                 Q = 25,
                 ):
        
        # if np.all(np.array(transporters_hubs) >= size**2):
            # raise('transporters\' locations are not smaller than nodes')
        
        assert len(emissions_KM) == len(costs_KM)
        
        num_vehicles = len(emissions_KM)
        
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
        
        self.omission_cost = 4*np.max(self.distance_matrix) +1
        
        self.CO2_penalty = max(CO2_penalty, 2*self.omission_cost)
        
        self.cost_matrix = np.array([
            (self.costs_KM[m] + self.CO2_penalty*self.emissions_KM[m])*self.distance_matrix
            for m in range(num_vehicles)
        ], dtype=int)
        
        self.num_vehicles = num_vehicles
        
        self.max_capacity = max_capacity
        
        self.horizon = horizon
        
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
    
    
    def reset(self, num_packages = None, packages = None, seed = None):
        
        np.random.seed(seed)
        
        if num_packages is None:
            num_packages = self.max_capacity * self.num_vehicles
        else:
            assert num_packages <= self.max_capacity * self.num_vehicles
        # super().reset(seed=seed)
        if packages is None:
            destinations = np.random.choice([i for i,_ in enumerate(self.G.nodes) if i!=85], size=num_packages, replace=False)

            self.packages = [
                Package(
                    destination=d,
                    quantity=1,#TODO
                )
                for d in destinations
            ]
        else:
            self.packages = packages
        
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
        
        self.num_packages = num_packages
        
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


class AssignmentEnv(gym.Env):
    def __init__(self, game : AssignmentGame, num_packages):
        self._game = game
        
    def reset(self, *, seed: int | None = None, packages = None, num_packages = None) -> tuple[np.ndarray, dict[str, Any]]:
        info = self._game.reset(num_packages, packages, seed)
        return self._get_observation(), info
    
    def step(self, action: np.ndarray, time_budget = 2) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        r, done, info = self._game(action, time_budget)
        return self._get_observation(), r, done, done, info
    
    def _get_observation(self):
        return None#TODO

if __name__ == '__main__':
    game = AssignmentGame()
    K = 50
    rewards = []
    game.reset(num_packages = K)
    
    done = False
    while not done:
        actions = np.ones(K, dtype=int)
        # actions[2] = 0
        r, d, info = game.step(actions, time_budget=1)
        done = True
        rewards.append(r)
        print(info)
        print(info['solution'])
        
        actions[4] = 0
        actions[5] = 0
        # actions[2] = 0
        r, d, info = game.step(actions, call_OR=False)
        rewards.append(r)
        print(info)
        print(info['solution'])
        
        r, d, info = game.step(actions)
        rewards.append(r)
        print(info)
        print(info['solution'])
    print('rewards', rewards)