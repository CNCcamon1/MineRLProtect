import gym
import minerl
import logging
import random
import numpy as np
from cp_nav_brain.cp_nav_brain import CpNavBrain
from taxi_brain.taxi_brain import TaxiBrain

MINERL_DATA_ROOT = "/e/Documents/Programming/MineRLProtect/data"

logging.basicConfig(level=logging.DEBUG)
#minerl.data.download(directory=MINERL_DATA_ROOT, experiment='MineRLNavigate-v0')

def run_minerl():
    env = gym.make("MineRLNavigateDense-v0")
    player_brain = CpNavBrain(env)
    player_brain.execute_episode()

def run_taxi():
    env = gym.make("Taxi-v3")
    player_brain = TaxiBrain(env)
    for i in range(1, 100001):
        reward = player_brain.execute_episode()
        if i % 100 == 0:        
            print(reward)
            print(f"Episode: {i}")

            

def train_from_data():
    minerl.data.download(directory=MINERL_DATA_ROOT, experiment='MineRLNavigate-v0')
    data = minerl.data.make(
        'MineRLNavigate-v0',
        data_dir=MINERL_DATA_ROOT
    )
    for current_state, action, reward, next_state, done \
        in data.sarsd_iter(
            num_epochs=1, max_sequence_len=32
        ):

        #print(current_state['pov'][0])
        print(reward[-1])
        print(done[-1])

if __name__=='__main__':
    #run_taxi()
    run_minerl()