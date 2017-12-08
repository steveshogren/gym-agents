import argparse
import logging
import sys

import gym
from gym import wrappers
from tabular_q_agent import TabularQAgent
import helpers as h

# gutted to use TabularQAgent instead with the Reverse-v0
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        # print str(self.action_space)
        return self.action_space.sample()

if __name__ == '__main__':
    env = gym.make('Reverse-v0')

    env.seed(0)

    episode_count = 100
    reward = 0
    done = False

    ob = env.reset()
    # convert the tuple action space into Discrete
    perms = h.triplePerms(env.action_space)
    actionSize = h.tupleSize(env.action_space)

    agent = TabularQAgent(ob, actionSize)

    agent.learn(env, lambda action: env.step(perms[action]))

    env.render()
    env.close()

