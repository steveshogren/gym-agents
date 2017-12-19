import argparse
import logging
import sys

import gym
from gym import wrappers
from tabular_q_agent import TabularQAgent
import helpers as h

# gutted to use TabularQAgent instead with the Reverse-v0

def convertObsToTuple(env, obs):
    currentLetter = obs
    atEnd =(env.env.input_width-env.env.read_head_position)==1
    anyWritten = env.env.write_head_position==0
    # print (currentLetter, currentPosition, countWritten)
    return (currentLetter, atEnd, anyWritten)

def step(env, action):
    obs,reward,done,_ = env.step(perms[action])
    return convertObsToTuple(env, obs),reward,done,_

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
    #print("action_space: " + str(env.action_space))
    #print("obs: " + str(ob))
    #print("input_width: " + str(env.env.input_width))
    #print("read_head_position: " + str(env.env.read_head_position))
    #print("last_action: " + str(env.env.last_action))
    #print("last_reward: " + str(env.env.last_reward))
    
    perms = h.triplePerms(env.action_space)

    actionSize = h.tupleSize(env.action_space)

    agent = TabularQAgent(ob, actionSize)

    agent.learn(env, step, convertObsToTuple)

    #env.render()

    env.close()

