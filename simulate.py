import numpy as np
import argparse
from train import train_agents
from nash import nash_equilibrium
import shutil
import os


def simulate(train_steps):
    env = train_agents(num_episodes=train_steps)
    env.set_status('gui')
    state = env.reset()
    max_steps_per_episode = 5
    for step in range(max_steps_per_episode):
        print(env.t)
        ne = nash_equilibrium(env.q_table[state], choice="random")
        if ne is None:
            print('Ugh')
            break
        action = ne[0]
        new_state, reward, done, info = env.step(action)
        state = new_state
        if done == True:
            break

if __name__ == "__main__":
    # Instantiate the parser
    if os.path.exists(os.path.join(os.getcwd(), 'images')):
        shutil.rmtree(os.path.join(os.getcwd(), 'images'))
    os.mkdir("./images")
    parser = argparse.ArgumentParser(description='Simulate Environment')
    parser.add_argument('train_steps', default=10000, nargs='?')
    args = parser.parse_args()
    train_steps = args.train_steps
    
    simulate(train_steps)
