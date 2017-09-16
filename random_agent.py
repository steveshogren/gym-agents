import argparse
import logging
import sys

import gym
from gym import wrappers
from tabular_q_agent import TabularQAgent


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        # print str(self.action_space)
        return self.action_space.sample()

if __name__ == '__main__':
    env = gym.make('Copy-v0')

    env.seed(0)
    #agent = TabularQAgent(ob, env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            agent = TabularQAgent(ob, env.action_space)

            #agent.learn(env)
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.render()
    env.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    #logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    #gym.upload(outdir)
