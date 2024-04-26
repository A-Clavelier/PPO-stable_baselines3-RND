import gymnasium
#import gym_sokoban
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
plt.ion()   # Turn on interactive mode

n_envs = 4
env_name = 'CartPole-v1'
max_episode_steps=512
render_mode = 'rgb_array'  
policy_name = "MlpPolicy"         
RND_reward = True   
RND_learning_rate = 0.0001
intrinsic_importance_coef = 1000

#set the model_logs path adequately
logs_dir ="./logs" + '/' + env_name + '/' + policy_name + '/' + f"RND_reward={RND_reward}" + '/' 
if RND_reward:
    logs_dir = logs_dir + f"RND_learning_rate={RND_learning_rate}~intrinsic_importance_coef={intrinsic_importance_coef}" + '/'
logs_dir = logs_dir + "models" + '/'
#choose the logfile to load
log_name = "PPO~learning_rate=0.0003~n_steps=2048~batch_size=64~n_epochs=10~gamma=0.99~ent_coef=0.01~n_policy_updates=502024-04-26-16-54"
#load the model
model = PPO.load(logs_dir+log_name)

# Define environment creation function
def make_env(env_name, render_mode, max_episode_steps):
    def _init():    #init is to create different instances intead of having the same environment every time
        env = gymnasium.make(env_name, render_mode=render_mode, max_episode_steps=max_episode_steps)
        return env
    return _init
#create vectorized environment
vec_env = DummyVecEnv([make_env(env_name, render_mode, max_episode_steps) for _ in range(n_envs)])


#---MODEL_TESTING---#

obs = vec_env.reset()
done = np.array(False)
total_reward = 0
while not done.all():
    prediction = model.predict(obs,deterministic=False) #deterministic = FALSE !!!!
    action, _ = prediction

    print(f"action: {action}                        - action type:{type(action)}")

    obs, reward, done, _ = vec_env.step(action)     #use prediction function of the MDP to change the state according to the action taken last step
    total_reward += reward                          #count total reward

    print(f"reward: {reward}                      - total_reward={total_reward}\n")
    
    # rendering
    plt.clf()
    plt.imshow(vec_env.render())
    plt.draw()
    plt.pause(0.01)  

print(f"Total Reward = {total_reward}")
vec_env.close()
