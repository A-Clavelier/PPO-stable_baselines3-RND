import gymnasium
#import gym_sokoban
import matplotlib.pyplot as plt
plt.ion()   # Turn on interactive mode
import time

env_name = 'LunarLander-v2'
render_mode = 'rgb_array'       
max_episode_steps=500
env=gymnasium.make(env_name,render_mode=render_mode, max_episode_steps=max_episode_steps)


#---ENV_TESTING---#

obs = env.reset()
done = False
total_reward = 0
for i in range(500):
    #choose a random action
    action = env.action_space.sample()
    obs, reward, done, _ , _= env.step(action)    
    total_reward += reward
    if done==True:
        print(f"\n ----------done={done}; resetting env...---------- \n")
        obs = env.reset()
    print(f"i={i}")
    print(f"action: {action}                        - action type:{type(action)}")
    print(f"reward: {reward}                      - total_reward={total_reward}")
    if i<100 or i>400:
        print("rendering...")
        plt.clf()
        plt.title(f"i = {i}")
        plt.imshow(env.render())
        plt.draw()
        plt.pause(0.01)  # Pause for interaction
    else:
        time.sleep(0.01)
    print()

print(f"Total Reward = {total_reward}")
env.close()
