import gym
import minerl
import numpy as np
import math
import random

class CpBrain:

    #Initializes CP's brain
    def __init__(self, env):

        self.env = env

        #Initializes an empty Q-Table
        self.q_table = np.zeros([len(env.observation_space.spaces), len(env.action_space.spaces)])

        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.1

        self.all_epochs = []
        self.all_penalties = []

    #Executes a single episode
    def execute_episode(self):
        state = self.env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False
        
        while not done:
            if random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample() # Explore action space
            else:
                action = np.argmax(self.q_table[state]) # Exploit learned values

            next_state, reward, done, info = self.env.step(action) 
            
            old_value = self.q_table[state, action]
            next_max = np.max(self.q_table[next_state])
            
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            self.q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1
            
