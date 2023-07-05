from dataclasses import dataclass
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ortools

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
        
        self.omission_cost = 2*np.max(self.distance_matrix) +1
        
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
        
    
    def _compute_cost(self, actions, time_budget):
        
        # nodes = [[] for _ in self.transporter]
        # quantities = [[] for _ in self.transporter]
        
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
            np.sum(quantities)#TODO precise quantities
        omission_penalty = self.omission_cost*omitted
        
        # for k in range(len(actions)):
        #     if actions[k]:
        #         nodes[actions[k]-1].append(self.packages[k].destination)
        #         quantities[actions[k]-1].append(self.packages[k].quantity)
        #     else:
        #         omission_penalty += self.omission_cost*self.packages[k].quantity
        #         omitted += 1
        # # print(nodes)
        total_costs = 0
        total_emissions = 0
        
        for m in range(len(self.transporter)):#TODO parallelize
            distance, time, solution = self.transporter[m].compute_cost(nodes[m], quantities[m], time_budget)
            total_costs +=     np.sum(self._get_costs(distance, time))
            total_emissions += np.sum(self._get_emissions(distance, time))
            
        self.info['solution_found'] = np.any(time != 0)
        self.info['costs'] = total_costs
        self.info['time_per_vehicle'] = time
        self.info['distance_per_vehicle'] = distance
        self.info['excess_emission'] = total_emissions - self.Q
        self.info['omitted'] = omitted
        self.info['solution'] = solution
            
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
        
    def _get_rewards(self, actions, time_budget):
        #TODO
        #print('obs is :', self.obs)
        #print('ids is :', self.ids)
        
        costs = self._compute_cost(actions, time_budget)
        

        
        # assert rewards[str(winner)]>=0
        
        return -costs
    
        
    def step(self, actions, time_budget = 1):
        self.t += 1
        
        done = self.t >= self.horizon

        return self._get_rewards(actions, time_budget), done, self.info


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
        r, d, info = game.step(actions, time_budget=10)
        done = True
        rewards.append(r)
        print(info)
        print(info['solution'])
    print('rewards', rewards)