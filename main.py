import gym
import minerl
import logging

MINERL_DATA_ROOT = "/e/Documents/Programming/MineRLProtect/data"

logging.basicConfig(level=logging.DEBUG)
minerl.data.download(directory=MINERL_DATA_ROOT)

def main():
    env = gym.make("MineRLTreechop-v0")
    obs = env.reset()
    done = False
    net_reward = 0

    while not done:
        obs, reward, done, _ = env.step(env.action_space.noop())
        

def build_data_set():
    data = minerl.data.make(
        'MineRLObtainDiamond-v0',
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