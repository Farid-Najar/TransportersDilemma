from assignment import AssignmentGame, AssignmentEnv, test_assignment_AlterAction_env, RemoveActionEnv, CombActionEnv
import numpy as np
import unittest



def test_assignment_game(game = None, K = 500, log = True, plot = True):
    try:
        if game is None:
            game = AssignmentGame(
                    # Q=0,
                    # grid_size=45,
                    # max_capacity=125,
                    # K=K,
                )
        # game = AssignmentGame()
        rewards = []
        game.reset()
        K = game.num_packages
        
        
        done = False
        i = 0
        actions = np.ones(game.num_packages, dtype=int)
        r, done, info = game.step(actions)
        while not done:
            rewards.append(r)
            if log:
                print(info)
                print(info['solution'])
            actions[i] = 0
            i += 1
            if i == K:
                break
            r, done, info = game.step(actions, call_OR=False)
            
        if log:
            print('rewards', rewards)
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(rewards)
            plt.show()
        return rewards
    except Exception:
        return None
    
        

def test_assignment_env(game = None, K = 50, log = False, plot = False):
    if game is None:
        game = AssignmentGame(
                # Q=0,
                # grid_size=45,
                # max_capacity=125,
                # K = K
            )
    env = AssignmentEnv(game)
    rewards = []
    env.reset()
    
    K = game.num_packages
    
    for l in range(len(env._game.solutions[0])):
        for i in range(len(env._game.solutions[0][l])):
            assert env.initial_routes[l, 2*i] == env._game.solutions[0][l][i]
    
    
    done = False
    i = 0
    actions = np.ones(game.num_packages, dtype=int)
    while not done:
        # actions[2] = 0
        _, r, done, _, info = env.step(actions)
        # done = True
        actions[i] = 0
        i += 1
        if i == K:
            break
        rewards.append(r)
        if log:
            print(info)

    if log:
        print('rewards', rewards)
        
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(rewards)
        plt.show()
    
    return rewards

class TestAssignment(unittest.TestCase):

    def test_game(self):
        # game = AssignmentGame(
        #     Q=0,
        #     K = 500,
        #     grid_size=45,
        #     max_capacity=125
        # )
        r = test_assignment_game(log = False, plot = False)
        # print('game ok!')
    
        self.assertFalse(r is None)
        if r is not None:
            self.assertFalse(np.max(r) == np.min(r))

    def test_env(self):
        # game = AssignmentGame(
        #     Q=0,
        #     K = 500,
        #     grid_size=45,
        #     max_capacity=125
        # )
        r = test_assignment_env(log = False, plot = False)
        self.assertFalse(r is None)
        if r is not None:
            self.assertFalse(np.max(r) == np.min(r))
        # print('env ok!')
        
    def test_AlterAction(self):
        test_assignment_AlterAction_env(log = False, plot = False)
        
    def test_RemoveAction(self):
        env = RemoveActionEnv()
        self.assertTrue(env.n_invalid_actions == 0)
        self.assertTrue(len(env.invalid_actions) == 0)
        env.reset()
        i = 0
        k = np.sum(env.action_masks())
        while True:
            self.assertTrue(np.sum(env.action_masks()) == env.H-i)
            o, r, d, _, info = env.step(i)
            self.assertTrue((o[-env.H:] == env.action).all())
            if info['excess_emission'] <= 0:
                self.assertTrue(d)
                # print(r)
                # print(i)
            if i >=env.H-1:
                self.assertTrue(d)
                self.assertTrue(np.sum(env.action_masks()) == 0)
                break
            i+=1
            self.assertTrue(env.n_invalid_actions == i)
            self.assertTrue(len(env.invalid_actions) == i)
            
    def test_CombAction(self):
        env = CombActionEnv()
        # self.assertTrue(env.n_invalid_actions == 0)
        # self.assertTrue(len(env.invalid_actions) == 0)
        env.reset()
        i = 0
        # k = np.sum(env.action_masks())
        while True:
            # self.assertTrue(np.sum(env.action_masks()) == env.H-i)
            a = env.action_space.sample()
            o, r, d, _, info = env.step(a)
            self.assertTrue((o[-env.H:] == a).all())
            self.assertTrue(o[-env.H-1] == info['excess_emission'])
            if info['excess_emission'] <= 0:
                self.assertTrue(d)
                # print(r)
                # print(i)
            if i >=env.H-1:
                self.assertTrue(d)
                # self.assertTrue(np.sum(env.action_masks()) == 0)
                break
            i+=1
            # self.assertTrue(env.n_invalid_actions == i)
            # self.assertTrue(len(env.invalid_actions) == i)
        

if __name__ == '__main__':
    unittest.main()

