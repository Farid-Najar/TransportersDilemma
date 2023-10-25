import sys
import os
from copy import deepcopy
# direc = os.path.dirname(__file__)
from pathlib import Path
path = Path(os.path.dirname(__file__))
# pri&
# caution: path[0] is reserved for script path (or '' in REPL)
# print(str(path)+'/ppo')
sys.path.insert(1, str(path.parent.absolute()))#'/Users/faridounet/PhD/TransportersDilemma')

from assignment import AssignmentGame, AssignmentEnv, test_assignment_AlterAction_env, RemoveActionEnv
import numpy as np
import unittest

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import logging
logging.basicConfig(
    filename='./test_RL.log',
    filemode='w',
)

def test_PPO_mask():
    env = Monitor(RemoveActionEnv(rewards_mode='terminal'))
    env.reset()
    model = MaskablePPO(MaskableActorCriticPolicy, env, gamma=0.4, seed=32, verbose=1)
    action = 0
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
    
    # self.assertTrue(mean_reward < 15_000)
    env.reset()
    while True:
        action_masks = get_action_masks(env)
        assert((action_masks == env.action_masks()).all())
        if not(action not in env.invalid_actions):
            print(env.action)
            assert False
        obs, reward, terminated, truncated, info = env.step(action)
        action, _states = model.predict(obs, action_masks=action_masks)
        # logging.info(f'invalid actions : {env.invalid_actions}')
        # logging.info(f'action : {action}')
        
        if info ['excess_emission']<= 0:
            assert(terminated)
        if terminated:
            # print(reward)
            assert(env.t < env.H)
            break
        
if __name__ == '__main__':
    test_PPO_mask()