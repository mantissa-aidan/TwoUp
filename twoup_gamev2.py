import shutil
import gym
import tempfile
from DQN import DQNAgent
from tqdm import *


def train(environment, model_name=None, key=None):
    tdir = tempfile.mkdtemp()
    env = gym.make(environment)
    env = gym.wrappers.Monitor(env, tdir, force=True)
    agent = DQNAgent(env)
    env.seed(0)
    # agent.load_model("TwoUp-v0_model.h5")
    EPISODES = 5000
    for episode in trange(EPISODES):
        state, reward, done = env.reset(), 0.0, False
        action = agent.action(state, reward, done, episode)
        while not done:
            # env.render()
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            state = next_state
            action = agent.action(state, reward, done, episode)
        if model_name and (episode == EPISODES - 1 or episode % 10 == 0):
            agent.save_model(filename=model_name)
            pass
    env.close()
    if key:
        gym.upload(tdir, api_key=key)
    shutil.rmtree(tdir)


def run(environment, model_name, key=None):
    tdir = tempfile.mkdtemp()
    env = gym.make(environment)
    env = gym.wrappers.Monitor(env, tdir, force=True)
    agent = DQNAgent(env, trained_model=model_name)
    EPISODES = 100
    for episode in range(EPISODES):
        state, reward, done = env.reset(), 0.0, False
        action = agent.action(state, reward, done, episode, training=False)
        while not done:
            #env.render()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            action = agent.action(state, reward, done, episode, training=False)
    env.close()
    if key:
        gym.upload(tdir, api_key=key)
    shutil.rmtree(tdir)


if __name__ == "__main__":
    environment = 'TwoUp-v0'
    api_key = ""
    my_model = environment + '_model.h5'

    train(environment=environment, key=api_key, model_name=my_model)
    #run(environment=environment, key=api_key, model_name=my_model)