import numpy as np

class Actor:
    def __init__(self, model):
        self.model = model
    
    def act(self, observation):
        return self.model.act(observation)
    

class Trainer:
    def __init__(self, env, actor : Actor):
        self.actor = actor
        self.env = env
        
    def train(self, 
              epoch = 1_000):
        
        for i in range(epoch):
            obs = self.env.reset()