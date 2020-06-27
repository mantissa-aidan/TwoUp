import h5py
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

#For GPU training on windows
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

### For gpu training on WSL
# import tensorflow.compat.v1 as tf 
# tf.enable_eager_execution(tf.ConfigProto(log_device_placement=True)) 

class DQNAgent:
    def __init__(self, environment, trained_model=None):
        # Initialize constant
        self.environment = environment
        self.obs_size = environment.observation_space.shape[0]
        self.action_size = environment.action_space.n
        self.consecutive_episodes = 100

        # Hyperparameters of the training
        self.learning_rate = 0.0005
        self.gamma = 0.99  # discount factor
        self.replay_memory = 50000
        self.replay_size = 128

        # Initialize neural network model
        if trained_model:
            self.model = self.load_model(filename=trained_model)
        else:
            self.model = self.build_model()

        # Exploration/exploitations parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.001
        self.episode_b4_replay = 32

        # Define variable
        self.storage = deque(maxlen=self.replay_memory)
        self.sum_reward, self.rewards_lst = 0.0, []


    def build_model(self):
        # Neural Network for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64 * self.obs_size, input_dim=self.obs_size, use_bias=True, activation='relu'))
        model.add(Dense(64 * self.obs_size, use_bias=True, activation='relu'))
        model.add(Dense(self.action_size, use_bias=True, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


    def store(self, state, action, reward, next_state, done):
        # Save history to storage for replay
        cnt_state = np.reshape(state, [1, self.obs_size])
        new_state = np.reshape(next_state, [1, self.obs_size])
        self.storage.append((cnt_state, np.array([action]), np.array([reward]), new_state, np.array([done])))

        
    def action(self, state, reward, done, episode, training=True):
        # Update cumulative reward
        self.sum_reward += reward

        # Episode ends
        if done:
            self.rewards_lst.append(self.sum_reward)
            avg_reward = np.mean(self.rewards_lst[-self.consecutive_episodes: ])
            print ('Episode %4d, Reward: %5d, Average rewards %5d' %(episode, self.sum_reward, avg_reward))
            self.sum_reward = 0.0
            self.epsilon = max(self.epsilon_decay * self.epsilon, self.epsilon_min)
            return -1

        # Episode not ends: return next action
        else:
            cnt_state = np.reshape(state, [1, self.obs_size])
            # Train agent
            if training:
                if episode >= self.episode_b4_replay:
                    self.replay()
                    if np.random.random() < self.epsilon:
                        action = self.environment.action_space.sample()
                    else:
                        act_values = self.model.predict(cnt_state)
                        action = np.argmax(act_values[0])
                else:
                    action = self.environment.action_space.sample()

            # Run trained agent
            else:
                act_values = self.model.predict(cnt_state)
                action = np.argmax(act_values[0])

            return action


    def replay(self):
        minibatch_idx = np.random.permutation(len(self.storage))[: self.replay_size]

        states      = np.concatenate([self.storage[i][0] for i in minibatch_idx], axis=0)
        actions     = np.concatenate([self.storage[i][1] for i in minibatch_idx], axis=0)
        rewards     = np.concatenate([self.storage[i][2] for i in minibatch_idx], axis=0)
        next_states = np.concatenate([self.storage[i][3] for i in minibatch_idx], axis=0)
        dones       = np.concatenate([self.storage[i][4] for i in minibatch_idx], axis=0)

        X_batch = np.copy(states)
        Y_batch = np.zeros((self.replay_size, self.action_size), dtype=np.float64)

        qValues_batch = self.model.predict(states)
        qValuesNewState_batch = self.model.predict(next_states)

        targetValue_batch = np.copy(rewards)
        targetValue_batch += (1 - dones) * self.gamma * np.amax(qValuesNewState_batch, axis=1)

        for idx in range(self.replay_size):
            targetValue = targetValue_batch[idx]
            Y_sample = qValues_batch[idx]
            Y_sample[actions[idx]] = targetValue
            Y_batch[idx] = Y_sample

            if dones[idx]:
                X_batch = np.append(X_batch, np.reshape(np.copy(next_states[idx]), (1, self.obs_size)), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards[idx]] * self.action_size]), axis=0)

        self.model.fit(X_batch, Y_batch, batch_size=len(X_batch), epochs=1, verbose=0)


    def save_model(self, filename):
        self.model.save(filename)


    def load_model(self, filename):
        return load_model(filename)