import gym
#import gym_sokoban
import time

env_name = 'MountainCar-v0'
render_mode = "rgb_array"       
max_episode_steps=200
env=gym.make(env_name,render_mode=render_mode, max_episode_steps=max_episode_steps)
env.reset()
env.render() 

#---ENV_TESTING---#

obs = env.reset()
done = False
total_reward = 0
while not done:
    action = env.action_space.sample()
    obs, reward, done, _ , _= env.step(action)    
    total_reward += reward
    env.render() 

    print(f"action: {action}                        - action type:{type(action)}")
    print(f"reward: {reward}                      - total_reward={total_reward}\n")

    time.sleep(0)                                 #pause time to see the agent move not too fast

# print(f"Total Reward = {total_reward}")
# env.close()
