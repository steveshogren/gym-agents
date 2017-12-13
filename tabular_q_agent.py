#from gym.spaces import Discrete, Tuple
from collections import defaultdict
from gym import error
from gym.spaces import discrete
import argparse, numpy as np
import pickle as pickle

class TabularQAgent(object):
    """
    Agent implementing tabular Q-learning.
    """


    def __init__(self, observation_space, action_space, **userconfig):
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = action_space.n
        self.config = {
            "init_mean" : 0.0,      # Initialize Q values with this mean
            "init_std" : 0.0,       # Initialize Q values with this standard deviation
            "learning_rate" : 1.0,  # learning rate 1.0 - 0.0  where 1.0 is for perfectly deterministic scenarios
            "eps": 0.98,            # Epsilon in epsilon greedy policies - 1.0 infinitely long negative traits
            "discount": 0.50,
            "n_iter": 1000000}        # Number of iterations
        self.config.update(userconfig)
        self.makeDefaultDict()

    def makeDefaultDict(self):
        output = open('data.pkl', 'rb')
        data1 = pickle.load(output)
        output.close()
        # print(data1)
        #data1 = False
        if (data1):
            print("Starting with: " + str(data1))
            self.q = defaultdict(lambda: self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"], data1)
        else:
            self.q = defaultdict(lambda: self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"])

    def saveState(self):
        output = open('data.pkl', 'wb')
        d = dict(self.q)
        pickle.dump(d, output)
        output.close()

    def printState(self):
        print(self.q)

    def chooseAction(self, observation, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        return np.argmax(self.q[observation]) if np.random.random() > eps else self.action_space.sample()

    def learn(self, env, envStep):
        config = self.config
        obs = env.reset()
        q = self.q
        currentSize = 0
        highestReward = 0
        for t in range(config["n_iter"]):
            action = self.chooseAction(obs)
            ## obs needs to take into account the current position on the tape
            ## and the current letter at that position!!!!!
            obs2, reward, done, _ = envStep(env, action)
            future = 0.0

            if not done:
                future = np.max(q[obs2])

            # update q
            q[obs][action] = \
               ((1-self.config["learning_rate"]) * q[obs][action]) + self.config["learning_rate"] * (reward + (config["discount"] * future))

            if done:
                hreward = env.env.episode_total_reward
                if (hreward > highestReward):
                    highestReward = hreward
                currentSize = currentSize + 1
                if currentSize % 10000 == 0 and hreward > 0:
                    self.saveState()
                    print("Current highest reward: " + str(highestReward))
                    print (str(currentSize) + "/" + str(config["n_iter"]))
                    env.render()
                # either a failure or success, reset the env
                obs2 = env.reset()

            obs = obs2
