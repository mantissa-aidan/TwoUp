import numpy as np

import argparse
import sys

import gym
from gym import wrappers, logger

from DQN import DQNAgent

import shutil
import tempfile

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def action(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='TwoUp-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)
    # agent = DQNAgent(env)

    episode_count = 1
    reward = 0
    done = False

    # ob = env.reset()
    # action = agent.act(ob, reward, done)
    # print(action)

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.action(ob, reward, done)
            if action == -1:
                break
            ob, reward, done, _ = env.step(action)
            env.render()
            if done:
                break


    #Close the env and write monitor result info to disk
    env.close()
