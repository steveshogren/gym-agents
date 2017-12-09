import argparse
import logging
import sys

import gym
from gym import wrappers
from tabular_q_agent import TabularQAgent
import helpers as h

# gutted to use TabularQAgent instead with the Reverse-v0

if __name__ == '__main__':
    env = gym.make('Reverse-v0')

    env.seed(0)

    reward = 0
    done = False

    ob = env.reset()
    # convert the tuple action space into Discrete
    # action space from this env is a triple of 3 discrete actions
    #       1. Move read head left 0 or right 1
    #       2. Write 1 or not 0
    #       3. Which character to write. A=0, B=1,..
    print(env.action_space)
    print(ob)
    
    perms = h.triplePerms(env.action_space)

    actionSize = h.tupleSize(env.action_space)

    agent = TabularQAgent(ob, actionSize)

    agent.learn(env, lambda action: env.step(perms[action]))

    env.render()
    env.close()

