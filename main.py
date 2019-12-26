import gym
import minerl
import logging
import random
import numpy as np

MINERL_DATA_ROOT = "/e/Documents/Programming/MineRLProtect/data"

logging.basicConfig(level=logging.DEBUG)
minerl.data.download(directory=MINERL_DATA_ROOT, experiment='MineRLNavigate-v0')

def main():
    env = gym.make("Taxi-v3")
    env.render()
    #state  = env.reset()
    done = False
    net_reward = 0

    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    for i in range(1, 1000):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = env.step(action) 
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1
            
        if i % 100 == 0:
            print("Reward: ", reward)
            print(f"Episode: {i}")
            

def build_data_set():
    data = minerl.data.make(
        'MineRLNavigate-v0',
        data_dir=MINERL_DATA_ROOT
    )
    for current_state, action, reward, next_state, done \
        in data.sarsd_iter(
            num_epochs=1, max_sequence_len=32
        ):

        print(current_state['pov'][0])
        print(reward[-1])
        print(done[-1])

if __name__=='__main__':
    main()