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

import argparse
import pickle

import numpy as np

# import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.logger import Figure
# from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.monitor import Monitor
import torch.nn as nn

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.evaluation import evaluate_policy as evaluate_maskable
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
# from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from assignment import CombActionEnv, RemoveActionEnv, AssignmentGame, NormalizedEnv
import logging
# logging.basicConfig(
#     filename='RL.log',
#     filemode='w',
#     format='%(levelname)s - %(name)s - %(message)s'
# )

# class FigureRecorderCallback(BaseCallback):
#     def __init__(self, *args, verbose = 1, **kwargs):
#         print('args : ')
#         print(args)
#         print('kwargs : ')
#         print(kwargs.keys())
#         assert False
#         super().__init__(verbose)

#     def _on_step(self):
#         # Plot values (here a random variable)
#         figure = plt.figure()
#         figure.add_subplot().plot(np.random.random(3))
#         # Close the figure after logging it
#         self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
#         plt.close()
#         return True

# def eval_model(model, vec_env:VecMonitor, n_eval = 10):
#     obs, _ = vec_env.reset()


def make_env(env, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env2 = deepcopy(env)
        env2.reset(seed=seed + rank)
        return env2
    set_random_seed(seed)
    return _init

def train_RL(
    vec_env,
    algo = PPO,
    policy = "MlpPolicy",
    policy_kwargs = {},
    callbackClass=EvalCallback,
    algo_file : str|None = None,
    algo_dir = str(path)+'/ppo',
    budget = 1000,
    n_eval = 10,
    save = True,
    eval_freq = 200,
    progress_bar =True,
    n_steps = 128,
    gamma = 0.99,
):
    
    # Instantiate the agent
    if algo_file is not None:
        try:
            model = algo.load(algo_file+'/best_model', env=vec_env)
            assert model.policy_kwargs == policy_kwargs
        except Exception as e:
            logging.warning(f'couldnt load the model because this exception has been raised :\n{e}')
            
            print(f'path is {path}')
            raise('couldnt load the model!')
    else:   
        model = algo(
            policy,
            vec_env,
            policy_kwargs=policy_kwargs,
            n_steps=n_steps,
            gamma=gamma,
            batch_size=n_steps*os.cpu_count(),
            # n_epochs=50,
            # learning_rate=5e-5,
            verbose=1,
            tensorboard_log=algo_dir+"/"
        )
    logging.info(f"the model parameters :\n {model.__dict__}")
    # Train the agent and display a progress bar
    if issubclass(algo, PPO):
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval)
        logging.info(f'Before training :\n mean, std = {mean_reward}, {std_reward}')
    else:
        mean_reward, std_reward = evaluate_maskable(model, model.get_env(), n_eval_episodes=n_eval)
        logging.info(f'Before training :\n mean, std = {mean_reward}, {std_reward}')
    
    # checkpoint_callback = CheckpointCallback(
    # #   save_freq=1000,
    #   save_path="./logs/",
    #   name_prefix="rl_model",
    #   save_replay_buffer=True,
    #   save_vecnormalize=True,
    # )
    
    eval_callback = callbackClass(vec_env, best_model_save_path=algo_dir,
                             log_path=algo_dir, eval_freq=eval_freq,
                             deterministic=True)
    
    model.learn(
        total_timesteps=budget,
        progress_bar=progress_bar,
        log_interval=100,
        callback=eval_callback,
        # tb_log_name="ppo",
    )
    # Save the agent
    if save:
        model.save(f'{str(path)}/{algo.__name__}')
    # del model  # delete trained model to demonstrate loading
    return model


def train_PPO(
    env_kwargs = dict(
        H=50,
        rewards_mode = 'heuristic', # possible values ['heuristic', 'terminal']
    ),
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        share_features_extractor=True,
        net_arch=[2048, 2048, 1024, 256, 128]#dict(
        #    pi=[2048, 2048, 1024, 256, 64], 
        #    vf=[2048, 2048, 1024, 256, 64])
    ),
    n_eval = 100,
    budget = int(2e4),
    save = True,
    save_path = None,
    eval_freq = 200,
    progress_bar =True,
    algo_file = str(path),
    n_steps = 128,
    gamma = 0.99,
    *args,
    **kwargs
):
    if save_path is None:
        save_path = str(path)+'/ppo'
        
    logging.basicConfig(
        filename=save_path+f'/ppo.log',
        filemode='w',
        format='%(levelname)s - %(name)s :\n%(message)s \n',
        level=logging.INFO,
    )
    logging.info('Train PPO started !')
    # Create environment
    env = CombActionEnv(**env_kwargs)
    check_env(env)
    # check_env(env)
    logging.info(
        f"""
        Environment information :
        
        Grid size = {env._env._game.grid_size}
        Q = {env._env._game.Q}
        K = {env._env._game.num_packages}
        n_vehicles = {env._env._game.num_vehicles}
        """
    )
    # Create environment
    num_cpu = os.cpu_count()
    logging.info(f'Number of CPUs = {num_cpu}')
    vec_env = SubprocVecEnv([make_env(env, i) for i in range(num_cpu)])
    vec_env = VecMonitor(vec_env, save_path+"/")
    # log(type(env))
    
    model = train_RL(
        vec_env,
        algo=PPO,
        policy_kwargs=policy_kwargs,
        budget=budget,
        n_eval=n_eval,
        save = save,
        algo_dir=save_path,
        eval_freq =     eval_freq ,
        progress_bar =    progress_bar,
        algo_file = algo_file,
        n_steps =     n_steps ,
        gamma = gamma,
    )

    

    

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `log_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, log_system_info=True)
    # model = PPO.load("ppo", env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    logging.info(f'After training :\n mean, std = {mean_reward}, {std_reward}')
    
    
############################################################################

def train_PPO_mask(
    env_kwargs = dict(
        rewards_mode = 'terminal', # possible values ['heuristic', 'terminal']
    ),
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        share_features_extractor=True,
        net_arch=[2048, 2048, 1024, 128]#dict(
        #    pi=[2048, 2048, 1024, 256],#, 128], 
        #    vf=[2048, 2048, 1024, 256])#, 128])
    ),
    n_eval = 100,
    budget = int(2e4),
    save = True,
    save_path = None,
    algo_file = str(path),
    eval_freq = 200,
    progress_bar =True,
    n_steps = 128,
    gamma = 0.99,
    normalize = True,
    **kwargs
):
    if save_path is None:
        save_path = str(path)+'/ppo_mask'
    logging.basicConfig(
        filename=save_path+f'/ppo_mask.log',
        filemode='w',
        format='%(levelname)s - %(name)s :\n%(message)s \n',
        level=logging.INFO,
    )
    logging.info('Train PPO maskable started !')
    # Create environment
    if normalize:
        env = NormalizedEnv(RemoveActionEnv(**env_kwargs))#rewards_mode = 'terminal')
        logging.info(
            f"""
            Environment information :

            Grid size = {env.env._env._game.grid_size}
            Q = {env.env._env._game.Q}
            K = {env.env._env._game.num_packages}
            n_vehicles = {env.env._env._game.num_vehicles}
            """
        )
    else:
        env = RemoveActionEnv(**env_kwargs)
    # check_env(env)
    
    num_cpu = os.cpu_count()
    logging.info(f'Number of CPUs = {num_cpu}')
    vec_env = SubprocVecEnv([make_env(env, i) for i in range(num_cpu)])
    vec_env = VecMonitor(vec_env, save_path+"/")
    # log(type(env))
    
    model = train_RL(
        vec_env,
        algo=MaskablePPO,
        policy=MaskableActorCriticPolicy,
        policy_kwargs=policy_kwargs,
        callbackClass=MaskableEvalCallback,
        budget=budget,
        n_eval=n_eval,
        save = save,
        algo_dir=save_path,
        algo_file = algo_file,
        eval_freq =     eval_freq ,
        progress_bar =    progress_bar,
        n_steps =     n_steps ,
        gamma = gamma,
    )

    mean_reward, std_reward = evaluate_maskable(model, model.get_env(), n_eval_episodes=10)
    logging.info(f'After training :\n mean, std = {mean_reward}, {std_reward}')
    # Create environment

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default="ppo_mask", choices=['ppo', 'ppo_mask'],
                        help='Selects the RL algorithm')
    parser.add_argument('--r_mode', default="normalized_terminal", choices=['heuristic', 'terminal', 'normalized_terminal', 'penalize_length'],
                        help='Selects the reward function')
    parser.add_argument('--obs_mode', default="routes", choices=['routes', 'action'],
                        help='Selects the observation of the agent.')
    parser.add_argument('--action_mode', default="all_nodes", choices=['destinations', 'all_nodes'],
                        help='Selects the actions of the agent.')
    
    parser.add_argument('--n_steps', type=int, default=256,
                       help='the number of steps done on an environment before updating the model')
    
    # parser.add_argument('--batch_size', type=int, default=2048,
    #                    help='the batch size of ppo')
    # parser.add_argument('--minibatch_size', type=int, default=256,
    #                    help='the mini batch size of ppo')
    parser.add_argument('--n_eval', type=int, default=25,
                       help='the sample size for the policy evaluation')
    parser.add_argument('--steps', type=int, default=50_000,
                       help='the maximum steps done by the algorithm')
    parser.add_argument('--verbose', type=int, default=0,
                       help='the verbosity')
    parser.add_argument('--save', type=bool, default=True,
                       help='save the model')
    parser.add_argument('--load', type=bool, default=False,
                       help='load the model')
    parser.add_argument('--eval_freq', type=int, default=200,
                       help='the number of steps before an evaluation')
    parser.add_argument('--progress_bar', type=bool, default=False,
                       help='the progress bar appearance')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='the discount factor gamma')
    
    parser.add_argument('--Q', type=float, default=30,
                       help='the emission quota')
    parser.add_argument('--K', type=int, default=50,
                       help='the number of deliveries')
    parser.add_argument('--C', type=int, default=15,
                       help='the capacity of each vehicle')
    parser.add_argument('--load_game', type=bool, default=True,
                       help='if the game is saved already')
    parser.add_argument('--instance_id', type=int, default=0,
                       help='The id of the saved instance')
    parser.add_argument('--change_instance', type=bool, default=False,
                       help='if the model should try to solve other instances')
    
    
    args = parser.parse_args()

    # train_PPO(budget=15000, n_eval=1, save = False)
    # r_mode = 'penalize_length'
    # save_dir = str(path)+f'/ppo_mask/rewardMode({r_mode})_steps({budget})'
    # os.mkdir(save_dir)
    # train_PPO_mask(env_kwargs = dict(
    #     rewards_mode = r_mode, # possible values ['heuristic', 'terminal', 'penalize_length']
    # ),
    # budget=budget, n_eval=25, save = True, save_path=save_dir
    # )
    comment = ''
    if not args.change_instance:
        comment += f'_instanceID{str(args.instance_id)}'
    if args.algo == 'ppo':
        train_algo = train_PPO
        save_dir = str(path)+f'/ppo/rewardMode({args.r_mode})_steps({args.steps})'+comment
    else:
        train_algo = train_PPO_mask
        save_dir = str(path)+f'/ppo_mask/rewardMode({args.r_mode})_steps({args.steps})'+comment
    os.makedirs(save_dir, exist_ok=True)
    

    try:
        if args.load_game:
            with open(f'TransportersDilemma/RL/game_K{args.K}.pkl', 'rb') as f:
                g = pickle.load(f)
            routes = np.load(f'TransportersDilemma/RL/routes_K{args.K}.npy')
            dests = np.load(f'TransportersDilemma/RL/destinations_K{args.K}.npy')
        else:
            assert False
    except Exception as e:
        print(e)
        raise('couldnt load')
        g = AssignmentGame(
            grid_size=15,
            max_capacity=args.C,
            Q = args.Q,
            K=args.K
        )
        with open(save_dir+'/game.pkl', 'wb') as f:
                pickle.dump(g, f, -1)
        routes = None,
        dests = None,
        
    if args.load:
        algo_file = str(path)
    else:
        algo_file = None
        
    train_algo(
        env_kwargs = dict(
            game = g,
            rewards_mode = args.r_mode, # possible values ['heuristic', 'terminal', 'normalized_terminal', 'penalize_length']
            action_mode = args.action_mode,
            saved_routes = routes,
            saved_dests = dests,
            obs_mode = args.obs_mode,
            change_instance = args.change_instance,
            instance_id = args.instance_id,
        ),
        policy_kwargs = dict(
            activation_fn=nn.ReLU,#LeakyReLU,
            share_features_extractor=True,
            net_arch= [1024, 1024, 256, 128] if args.obs_mode == 'action' and args.action_mode == 'destinations' else
            [2048, 2048, 1024, 512]#, 128]#dict(
            # [2048, 2048, 1024, 256]#, 128]#dict(
            #    pi=[2048, 2048, 1024, 256],#, 128], 
            #    vf=[2048, 2048, 1024, 256])#, 128])
        ),
        budget=args.steps, n_eval=args.n_eval, save = args.save, save_path=save_dir,
        eval_freq = args.eval_freq, progress_bar =args.progress_bar, n_steps = args.n_steps,
        gamma = args.gamma, algo_file = algo_file,
        normalize=False if args.obs_mode == 'action' else True
    )
    
    # r_mode = 'heuristic'
    # save_dir = str(path)+f'/ppo_mask/rewardMode({r_mode})_steps({budget})'
    # os.mkdir(save_dir)
    # train_PPO_mask(env_kwargs = dict(
    #         rewards_mode = r_mode, # possible values ['heuristic', 'terminal', 'penalize_length']
    #     ),
    #     budget=30_000, n_eval=25, save = True, save_path=save_dir
    # )
    
    # /opt/homebrew/bin/python3.10 /Users/faridounet/PhD/TransportersDilemma/RL/train_RL.py --verbose 1 --progress_bar True --steps 500000 --change_instance True
    # /opt/homebrew/bin/python3.10 /Users/faridounet/PhD/TransportersDilemma/RL/train_RL.py --verbose 1 --progress_bar True --steps 100000  --obs_mode action --K 100