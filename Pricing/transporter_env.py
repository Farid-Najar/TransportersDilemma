import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ortools

from transporter import Transporter

from typing import List, Dict, Optional
# from dataclasses import dataclass

import gymnasium as gym
from ray.rllib.env import MultiAgentEnv

class TransportEnv(gym.Env):
    def __init__(self, 
                 transporters : Optional[List[Transporter]] = None,
                 size : int = 12,
                 transporters_hubs : list | set | tuple = (27, 116),
                 seed = 42,
                 nb_margins = 10,
                 max_capacity = 10,
                 horizon = 100,
                 ):
        
        if np.all(np.array(transporters_hubs) >= size**2):
            raise('transporters\' locations are not smaller than nodes')
        
        super().__init__()
        self.size = size
        margin_range = (-.1, 1) # The maximum cost is 2*size in a grid + the maximum profit
        self.transporters_hubs = transporters_hubs
        self.transporters_clients_color = ('lightcoral', 'lightgreen')
        self.G = nx.grid_2d_graph(size, size)
        self.distance_matrix = nx.floyd_warshall_numpy(self.G)
        
        self.obs_dim = 4+horizon
        self.action_space = gym.spaces.Discrete(nb_margins)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.obs_dim),
            high=1000*np.ones(self.obs_dim)
        )
        
        self.margins =  1. + np.linspace(*margin_range, nb_margins)#.astype(int)
        self.max_capacity = max_capacity
        self.delegation_cost = margin_range[-1]
        
        self.horizon = horizon
        
        if transporters is not None:
            self.transporters = transporters
        else:
            self.transporters = [
                Transporter(
                    self.distance_matrix, 
                    transporter_hub=transporters_hubs[i],
                    delegation_cost=self.delegation_cost,
                    max_capacity=max_capacity,
                )
                for i in range(len(transporters_hubs))
            ]
        
        self.ids = [
            str(i) for i in range(len(self.transporters))
        ]

        self.seed(seed)
        
    def seed(self, seed = 42):
        np.random.seed(seed)
        # super().seed(seed)
        
    def get_ids(self):
        return self.ids

    def render(self):
        _, ax = plt.subplots()
        nx.draw_networkx(self.G, 
                         pos = dict( (n, n) for n in self.G.nodes()),  
                         ax=ax, 
                         font_size=5, 
                         with_labels=False, 
                         node_size=100, 
                         node_color=self.G_ncolors, 
                         node_shape='s')
        # plt.draw()
        # return super().render()
        return ax
        
    
    def _compute_cost(self, node, quantity):
        res = []
        for transporter in self.transporters:
            try:
                cost = transporter.compute_marginal_cost(node, quantity)
            except Exception:
                cost = self.margins[-1]
            res.append(cost)
        return res
    def _get_state(self, random_qantity=False):
        # print('ids are : ', self.ids)
        if self.t == 0:
            self.obs = {
                i : np.zeros(self.obs_dim)
                for i in self.ids
            }
        
        # The destination of the client
        node = self.available_nodes.pop(np.random.randint(len(self.available_nodes)))
        
        #Compute the manhattan distance between the hubs and the destination
        position_node = np.array([node//self.size, node%self.size]) 
        position_hub1 = np.array([self.transporters_hubs[0]//self.size, self.transporters_hubs[0]%self.size])
        position_hub2 = np.array([self.transporters_hubs[1]//self.size, self.transporters_hubs[1]%self.size])
        distance_from_hub1 = np.linalg.norm(
            position_node
            -
            position_hub1,
            ord=1
        ) 
        distance_from_hub2 = np.linalg.norm(
            position_node
            - 
            position_hub2,
            ord=1
        ) 
        # compute the quantity
        if random_qantity :
            quantity = np.random.randint(self.max_capacity)
        else:
            quantity = 1
            
        #Construct the state
        for i in self.obs.keys():
            self.obs[i][0] = self.t
            self.obs[i][self.t] = node
            self.obs[i][-3] = quantity
            self.obs[i][-2] = distance_from_hub1
            self.obs[i][-1] = distance_from_hub2
            
        return self.obs
    
    
    def reset(self, seed=42, options=None):
        # super().reset(seed=seed)
        for transporter in self.transporters:
            transporter.reset()
            
        self.G_ncolors = ['#1f78b4' for _ in range(len(self.G.nodes))]
        self.G_ncolors[self.transporters_hubs[0]] = 'red'
        self.G_ncolors[self.transporters_hubs[1]] = 'green'
        
        self.available_nodes = list(range(len(self.G_ncolors)))
        self.available_nodes.remove(self.transporters_hubs[0])
        self.available_nodes.remove(self.transporters_hubs[1])
        
        self.t = 0
        
        return self._get_state(), self._get_info()
        
    def _get_rewards(self, actions):
        
        #print('obs is :', self.obs)
        #print('ids is :', self.ids)
        
        node = int(self.obs["0"][self.t - 1])
        quantity = self.obs["0"][-3]
        self.costs = np.array(self._compute_cost(node, quantity))
        # print(costs)
        # The transporters can propose a price greater than the cost
        # a = [
        #     np.argmax(
        #     self.margins
        #     >
        #     np.maximum(costs[int(i)], self.margins[actions[i]]) 
        #     )
        # if actions[i] < len(self.margins) -1
        # else actions[i]
        # for i in actions.keys()
        # ]
        # print(actions.values())
        a = self.costs * self.margins[list(actions.values())]
        
        # Random allocation if equal margins
        if a[0] == a[1]:
            winner = np.random.randint(2)
        else:
            # smallest price wins otherwise
            winner = np.argmin(a)
        rewards = {
            i : 0
            for i in actions.keys()
        }
        # The reward is the profit
        
        rewards[str(winner)] = a[winner] - self.costs[winner]
        
        # assert rewards[str(winner)]>=0
        
        self.transporters[winner].new_order(node, quantity)
        # We update the colors
        self.G_ncolors[node] = self.transporters_clients_color[winner]
        return rewards
    
    def _get_info(self):
        # TODO
        return {
            i : dict()
            for i in self.ids
        }
        # return dict()
        
    def step(self, actions : Dict):
        self.t += 1
        done = {
            i : self.t >= self.horizon
            for i in self.ids
        }
        trunc = {
            i : False
            for i in self.ids
        }
        return self._get_state(), self._get_rewards(actions), done, trunc, self._get_info()
    
class OneDynamicTransporter(gym.Env):
    def __init__(self, other_transporter_policy = None, *args, **kwargs):
        # super().__init__()
        if other_transporter_policy is None:
            self.other = lambda x : 1
        else:
            self.other = other_transporter_policy
        self.ma_env = TransportEnv(*args, **kwargs)
        self.action_space = self.ma_env.action_space
        self.observation_space = self.ma_env.observation_space
        
    def reset(self, *, seed: int | None = 42, options = None):
        obs, info = self.ma_env.reset(seed, options)
        return obs["0"], info["0"]
    
    def step(self, action):
        actions = dict()
        actions["0"] = action
        actions["1"] = self.other(self.ma_env.obs['1'])
        
        obs, reward, done, tr, info = self.ma_env.step(actions)
        return obs["0"], reward["0"], done["0"], tr["0"], info["0"]
    
    def seed(self, seed = None):
        self.ma_env.seed(seed)
        
    def render(self):
        return self.ma_env.render()
    @property
    def t(self):
        return self.ma_env.t
        
        
class MAT(MultiAgentEnv):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.env = TransportEnv(*args, **kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # assert len(self.env.ids ) == 2
        self.ids = self.env.ids
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        return self.env.reset(seed, options)
    
    def step(self, actions : Dict):
        return self.env.step(actions)
    
    def render(self):
        return self.env.render()
    

if __name__ == '__main__':
    env = MAT(max_capacity = 10)
    rewards = []
    env.reset()
    env.render()
    done = False
    while not done:
        actions = {
            i : 0
            for i in range(2)
        }
        _, r, d, *_ = env.step(actions)
        done = d["0"]
        rewards.append(r)
    env.render()
    t1_r = [r[0] for r in rewards]
    t2_r = [r[1] for r in rewards]
    print('transporter 1 total profit', sum(t1_r))
    print('transporter 2 total profit', sum(t2_r))