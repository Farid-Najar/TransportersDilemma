import multiprocess as mp
from tqdm import tqdm

from compare_game_SA import run_SA_TSP, compare
from GameLearning import make_different_sims, EXP3, LRI





if __name__ == '__main__' :
    # compare_SA_DP(n_simulation=50, K=50)
    # compare(n_simulation=100, K=16)
    compare(n_simulation=100, K=20, retain=1., real_data=True)
    # compare(n_simulation=100, K=30, real_data=True)
    compare(n_simulation=100, K=50, real_data=True)
    compare(n_simulation=100, K=50, retain=.8, real_data=True)
    compare(n_simulation=100, K=100, real_data=True)
    # compare(n_simulation=100, K=250, real_data=True)
    # run_DP(n_simulation=50, K=100, real_data=True)
    run_SA_TSP(n_simulation=100, K=20, retain=1., real_data=True)
    run_SA_TSP(n_simulation=100, K=50, real_data=True)
    run_SA_TSP(n_simulation=100, K=100, real_data=True)
    
    K = 50
    make_different_sims(K = K, strategy = LRI, n_simulation=100, T=10_000, log=False, tsp=True, comment = '_tsp', real_data = True)
    make_different_sims(K = K, strategy = EXP3, n_simulation=100, T=10_000, log=False, tsp=True, comment = '_tsp', real_data = True)
    
    K = 100
    make_different_sims(K = K, strategy = LRI, n_simulation=100, T=15_000, log=False, tsp=True, comment = '_tsp', real_data = True)
    make_different_sims(K = K, strategy = EXP3, n_simulation=100, T=15_000, log=False, tsp=True, comment = '_tsp', real_data = True)
    
    K = 20
    make_different_sims(K = K, strategy = LRI, n_simulation=100, T=10_000, log=False, tsp=True, comment = '_tsp', real_data = True)
    make_different_sims(K = K, strategy = EXP3, n_simulation=100, T=10_000, log=False, tsp=True, comment = '_tsp', real_data = True)
    