import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ortools

from transporter import Transporter

from typing import List, Dict, Optional
# from dataclasses import dataclass

import gym
    

class TransportEnv(gym.Env):
    def __init__(self, 
                 transporters : Optional[List[Transporter]] = None,
                 price_range = (1, 100),
                 size : int = 12,
                 transporters_hubs : list | set | tuple = (27, 116),
                 seed = 42,
                 nb_prices = 20,
                 max_capacity = 50,
                 horizon = 128,
                 ):
        
        if np.all(np.array(transporters_hubs) >= size**2):
            raise('transporters\' locations are not smaller than nodes')
        
        super().__init__()
        self.size = size
        self.transporters_hubs = transporters_hubs
        self.transporters_clients_color = ('lightcoral', 'lightgreen')
        self.G = nx.grid_2d_graph(size, size)
        self.distance_matrix = nx.floyd_warshall_numpy(self.G)
        
        self.action_space = gym.spaces.Discrete(nb_prices)
        self.prices = np.linspace(*price_range, nb_prices).astype(int)
        self.max_capacity = max_capacity
        self.delegation_cost = price_range[-1]
        
        self.horizon = horizon
        
        if transporters is not None:
            self.transporters = transporters
        else:
            self.transporters = [
                Transporter(
                    self.distance_matrix, 
                    transporter_hub=transporters_hubs[i],
                    delegation_cost=self.delegation_cost
                )
                for i in range(len(transporters_hubs))
            ]
        
        np.random.seed(seed)
        
    def render(self):
        fig, ax = plt.subplots()
        nx.draw_networkx(self.G, 
                         pos = dict( (n, n) for n in self.G.nodes()),  
                         ax=ax, 
                         font_size=5, 
                         with_labels=False, 
                         node_size=100, 
                         node_color=self.G_ncolors, 
                         node_shape='s')
        # return super().render()
        
        
    def _create_new_demand(self):
        # Delivery only with the same size for every one
        node = np.random.choice(self.available_nodes)
        quantity = np.random.randint(self.capacity)
        return np.array([node, quantity])
    
    def _compute_cost(self, node, quantity):
        return [self.transporters[i].compute_cost(node, quantity)
                for i in range(len(self.transporters))]
        
    def _get_state(self, random_qantity=False):
        self.state = {
            i : np.zeros(3)
            for i in range(len(self.transporters))
        }
        
        node = self.available_nodes.pop(np.random.randint(len(self.available_nodes)))
        if random_qantity :
            quantity = np.random.randint(self.max_capacity)
        else:
            quantity = 1
        for i in self.state.keys():
            self.state[i][0] = node
            self.state[i][1] = quantity
            self.state[i][2] = self.t
        return self.state
    
    
    def reset(self):
        
        self.G_ncolors = ['#1f78b4' for _ in range(len(self.G.nodes))]
        self.G_ncolors[self.transporters_hubs[0]] = 'red'
        self.G_ncolors[self.transporters_hubs[1]] = 'green'
        
        self.available_nodes = list(range(len(self.G_ncolors)))
        self.available_nodes.remove(self.transporters_hubs[0])
        self.available_nodes.remove(self.transporters_hubs[1])
        
        self.t = 0
        
        return self._get_state()
        
    def _get_rewards(self, actions):
        
        node = int(self.state[0][0])
        quantity = self.state[0][1]
        costs = [transporter.compute_cost(node, quantity) for transporter in self.transporters]
        a = [
            np.argmax(
            self.prices
            >
            np.maximum(costs[i], self.prices[actions[i]]) 
            )
        for i in actions.keys()
        ]
        if a[0] == a[1]:
            winner = np.random.randint(2)
        else:
            winner = np.argmin(a)
        rewards = {
            i : 0
            for i in actions.keys()
        }
        rewards[winner] = self.prices[a[winner]] - costs[winner]
        self.transporters[winner].new_order(node, quantity)
        self.G_ncolors[node] = self.transporters_clients_color[winner]
        return rewards
    
    def _get_info(self):
        # TODO
        # return {
        #     i : {}
        #     for i in range(len(self.transporters))
        # }
        return {}
        
    def step(self, actions : Dict):
        self.t += 1
        done = {
            i : self.t >= self.horizon
            for i in range(len(self.transporters))
        }
        return self._get_state(), self._get_rewards(actions), done, self._get_info()
    