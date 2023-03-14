import gym
import numpy as np

def fitness(net):
    env = gym.make('CartPole-v1')
    obs = env.reset()
    obs=obs[0]
    done = False
    total_reward = 0
    while not done:
        
        action = net.forward(obs)
        action = np.argmax(action.flatten())

        obs, reward, trun,term, info = env.step(action)
        if trun or term:
            done = True
        total_reward += reward
           
    env.close()
    return total_reward
