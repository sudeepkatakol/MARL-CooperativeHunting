import numpy as np
import environment 
from nash import nash_equilibrium

alpha = 0.3
discount_rate = 1
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

def train_agents(env = environment.Enviroment(), num_episodes=100, max_steps_per_episode=5):
    global alpha, discount_rate, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate
    q_table = env.q_table
    env.set_status('train')
    for episode in range(num_episodes):
        state = env.reset()
        for step in range(max_steps_per_episode):
            r = np.random.uniform(0, 1)
            ## Exploitation
            if r > exploration_rate:
                ne = nash_equilibrium(q_table[state], choice="random")
                if ne is None:
                    print('Ugh')
                    break
                action = ne[0]
            ## Exploration
            else:
                action = env.action_space.sample(seed=env._seed)
                #print(action, end=" ")
            new_state, reward, done, info = env.step(action)
            
            #print(new_state, reward, done)
            ne = nash_equilibrium(q_table[new_state], choice="random")
            if ne is None:
                print('Ugh')
                break
            values = ne[1]
            update = (1 - alpha) * q_table[state][action[0]][action[1]] + alpha * (reward + discount_rate * values)
            q_table[state][action[0]][action[1]][0] = update[0]
            q_table[state][action[0]][action[1]][1] = update[1]
            state = new_state
            if done == True:
                break
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)*np.exp(-exploration_decay_rate*episode)
    return env
