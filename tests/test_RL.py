from assignment import AssignmentGame, AssignmentEnv, test_assignment_AlterAction_env, RemoveActionEnv
import numpy as np
import unittest

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.maskable.evaluation import evaluate_policy

# import logging
# logging.basicConfig(
#     filename='test_RL.log',
#     filemode='w',
# )

class TestRL(unittest.TestCase):

    def test_PPO_mask(self):
        env = Monitor(RemoveActionEnv(rewards_mode='terminal'))
        env.reset()
        model = MaskablePPO(MaskableActorCriticPolicy, env, gamma=0.4, seed=32, verbose=1)
        action = 0
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
        
        # self.assertTrue(mean_reward < 15_000)
        for _ in range(100):
            env.reset()
            while True:
                obs, reward, terminated, truncated, info = env.step(action)
                action_masks = get_action_masks(env)
                self.assertTrue((action_masks == env.action_masks()).all())
                action, _states = model.predict(obs, action_masks=action_masks)
                # logging.info(f'invalid actions : {env.invalid_actions}')
                # logging.info(f'action : {action}')
                self.assertTrue(action not in env.invalid_actions)
                if info ['excess_emission']<= 0:
                    self.assertTrue(terminated)
                if terminated:
                    # print(reward)
                    self.assertTrue(env.t < env.H)
                    break


if __name__ == '__main__':
    unittest.main()