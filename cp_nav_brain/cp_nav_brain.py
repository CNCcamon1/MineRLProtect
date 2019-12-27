import gym
import minerl
import numpy as np
import math
import random
from .cp_nav_table import NavTable

class CpNavBrain:

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
        #Restart the env
        state = self.env.reset()
        epochs, penalties, reward, = 0, 0, 0
        nav_table = NavTable()
        done = False
        previous_reward = 0
        max_reward = 0
        
        while not done:
            learned_action = nav_table.get_action_by_state(state)
            if random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample()
            elif (learned_action):
                action = learned_action
            else:
                action = self.env.action_space.noop()
                action['forward'] = 1
            
            #Perform the action
            next_state, reward, done, info = self.env.step(action)
            print("Reward: ", reward)

            if (reward < previous_reward):
                nav_table.insert_node(state, action, next_state, 40.0)
                print("Confidence: 40")
            elif (reward > max_reward):
                nav_table.insert_node(state, action, next_state, 60.0)
                max_reward = reward
                print("Confidence: 60")
            else:
                nav_table.insert_node(state, action, next_state, 50.0)
                print("Confidence: 50")
            
            previous_reward = reward
            
        
        nav_table.adjust_confidence(reward)

            


            
