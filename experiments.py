import multiprocess as mp
from tqdm import tqdm
import subprocess

from compare_game_SA import run_SA_TSP, compare
from GameLearning import make_different_sims, EXP3, LRI





if __name__ == '__main__' :
    
    compare(n_simulation=100, K=20, retain=1., real_data=True)
    
    # compare(n_simulation=100, K=50, real_data=True)
    # compare(n_simulation=100, K=50, retain=.8, real_data=True)
    # compare(n_simulation=100, K=100, real_data=True)
    
    # run_SA_TSP(n_simulation=100, K=20, retain=1., real_data=True)
    # run_SA_TSP(n_simulation=100, K=50, real_data=True)
    # run_SA_TSP(n_simulation=100, K=100, real_data=True)
    
    # K = 50
    # make_different_sims(K = K, strategy = LRI, n_simulation=100, T=10_000, log=False, tsp=True, comment = '_tsp', real_data = True)
    # make_different_sims(K = K, strategy = EXP3, n_simulation=100, T=10_000, log=False, tsp=True, comment = '_tsp', real_data = True)
    
    # K = 100
    # make_different_sims(K = K, strategy = LRI, n_simulation=100, T=15_000, log=False, tsp=True, comment = '_tsp', real_data = True)
    # make_different_sims(K = K, strategy = EXP3, n_simulation=100, T=15_000, log=False, tsp=True, comment = '_tsp', real_data = True)
    
    # K = 20
    # make_different_sims(K = K, strategy = LRI, n_simulation=100, T=10_000, log=False, tsp=True, comment = '_tsp', real_data = True)
    # make_different_sims(K = K, strategy = EXP3, n_simulation=100, T=10_000, log=False, tsp=True, comment = '_tsp', real_data = True)
    
    # import os
    # To run RL experiments
    # Make sure that you use the correct virtual env
    # os.system("python3 RL/train_RL.py --verbose 1 --progress_bar True --steps 20000  --obs_mode multi --K 50")
    # # print(subprocess.run(["python3", "RL/train_RL.py --verbose 1 --progress_bar True --steps 20000  --obs_mode multi --K 50"],capture_output=True))
    # os.system("python3 /TransportersDilemma/RL/train_RL.py --verbose 1 --progress_bar True --steps 20000  --obs_mode multi --K 100")
    # os.system("python3 /TransportersDilemma/RL/train_RL.py --verbose 1 --progress_bar True --steps 1000000  --obs_mode multi --K 50 --change_instance True")
    # os.system("python3 /TransportersDilemma/RL/train_RL.py --verbose 1 --progress_bar True --steps 1000001 --K 50 --retain_rate 0.8 --change_instance True")
    # os.system("python3 /TransportersDilemma/RL/train_RL.py --verbose 1 --progress_bar True --steps 1000001 --K 20 --retain_rate 1. --change_instance True --obs_mode routes")
    