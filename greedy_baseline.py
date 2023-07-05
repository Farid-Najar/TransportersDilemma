import numpy as np
from assignment import AssignmentGame

def greedy(game : AssignmentGame):
    
    vehicles_by_priority = np.argsort(game.emissions_KM)
    
    q = np.ones(len(game.packages)+1)
    q[1:] = np.array([
        d.quantity for d in game.packages
    ])
    
    nodes = [
        d.destination for d in game.packages
    ]
    
    action = np.ones(game.num_packages, dtype=bool)
    rewards = np.zeros(game.num_packages)
    
    l = [game.hub] + nodes
    x, y = np.ix_(l, l)
    
    A = game.distance_matrix[x, y]@np.diag(1/q)
    indices = np.flip(np.argsort(np.min(A[1:, 1:] + np.max(A[1:, 1:])*np.eye(len(A[1:, 1:])), axis=0))) 
    
    for t in range(len(indices)):
        i = indices[t]
        r, d, info = game.step(action.astype(int))
        action[i] = not action[i]
        print(info)
        rewards[t] = r
    # for m in vehicles_by_priority:
    #     frm = game.hub
    return rewards
    
if __name__ == '__main__':
    # q = (np.random.randint(5, size=(5)) + 1)
    # Q = np.diag(1/q)
    # A = np.random.randint(50, size=(5, 5))
    # np.fill_diagonal(A,0)
    # B = A@np.diag(1/q)

    # print(np.all(B == A@Q))
    import matplotlib.pyplot as plt
    
    game = AssignmentGame()
    game.reset(50)
    
    r = greedy(game)
    plt.plot(r)
    plt.show()