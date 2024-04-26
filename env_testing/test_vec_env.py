import gymnasium
#import gym_sokoban
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
plt.ion()   # Turn on interactive mode
import time

n_envs=4
env_name = 'ALE/MontezumaRevenge-v5'
render_mode = 'rgb_array'       
max_episode_steps=500

# Define environment creation function
def make_env(env_name, render_mode, max_episode_steps):
    def _init():    #init is to create different instances intead of having the same environment every time
        env = gymnasium.make(env_name, render_mode=render_mode, max_episode_steps=max_episode_steps)
        return env
    return _init

vec_env = DummyVecEnv([make_env(env_name, render_mode, max_episode_steps) for _ in range(n_envs)])

#vec_env = make_vec_env('CartPole-v1', n_envs=n_envs)   #make_vec_env is for faster calculation in parrallel (only on GPU)


#---ENV_TESTING---#

obs = vec_env.reset()
done = np.array(False)
total_reward = 0
for _ in range(500):
    #choose a random action
    action =  np.array([vec_env.action_space.sample() for _ in range(n_envs)])
    obs, reward, done, _ = vec_env.step(action)    
    total_reward += reward
    
    print(f"action: {action}                        - action type:{type(action)}")
    print(f"reward: {reward}                      - total_reward={total_reward}\n")

    plt.clf()
    plt.imshow(vec_env.render())
    plt.draw()
    plt.pause(0.01)  # Pause for interaction

print(f"Total Reward = {total_reward}")
vec_env.close()