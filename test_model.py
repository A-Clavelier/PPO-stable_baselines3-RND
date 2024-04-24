import gym
#import gym_sokoban

from stable_baselines3 import PPO

import time


env_name = 'CartPole-v0'
#set the model_logs path adequately
model_logs_dir = "./model_logs" + '/' + env_name + '/'
#choose the logfile to load
log_name = "2024-04-23-13-05~PPO_model~CartPole-v0~policy_kwargs=None~total_timesteps=51200"
#load the model
model = PPO.load(model_logs_dir+log_name)
#create environment
env=gym.make(env_name)


#---MODEL_TESTING---#

obs = env.reset()
done = False
total_reward = 0
while not done:
    prediction = model.predict(obs,deterministic=False) #deterministic = FALSE !!!!
    action, _ = prediction
    int_action = int(action)                            #action type tuple -> int

    obs, reward, done, _ = env.step(int_action)     #use prediction function of the MDP to change the state according to the action taken last step
    total_reward += reward                          #count total reward
    env.render()                                    #render environment

    #print(f"action: {action}                        - action type:{type(action)}")
    print(f"int_action: {int_action}                    - int_action type:{type(int_action)}")
    print(f"reward: {reward}                      - total_reward={total_reward}\n")

    time.sleep(0.1)                                 #pause time to see the agent move not too fast

print(f"Total Reward = {total_reward}")
env.close()
