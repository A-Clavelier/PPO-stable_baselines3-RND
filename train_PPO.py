import gymnasium
#import gym_sokoban

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import custom_policies
import custom_callbacks
import RND_model

from datetime import datetime
import os

#----------------------------------------LOAD_MODEL_OR_NEW_MODEL?----------------------------------------------#

load = False     #choose wether to load a model to continue training or create a new model
if load:         #choose the logfile to load
        log_name = ""


#----------------------------------------------PARAMETERS------------------------------------------------------#

#env parameters
n_envs = 4                      #define the number of environments running in parallel
env_name = 'CartPole-v1'        #define the environment you want to create an agent for
render_mode = 'rgb_array'       
max_episode_steps=512

if not load:
    #Policy choice
    policy_name = "MlpPolicy"                               #define policy of the PPO with the custom_policies script(default: "MlpPolicy")                 
    policy = getattr(custom_policies, policy_name)          #policy = custom_policies.[policy_name]

    #PPO model parameters           #default values
    learning_rate = 0.0003          #0.0003     Reduce if results are unstable
    n_steps =  512                  #2048       number of steps of a rollout (between each policy updates) 
    batch_size = 64                 #64         Increase for more stable but slower learning (the info of the n_steps is spread into multiple batches to perform the policy update) 
    n_epochs = 10                   #10         number of optimization epochs to perform on each batch of samples
    gamma = 0.99                    #0.99       Discount factor Adjust according to the importance of future rewards
    ent_coef = 0.01                 #0.0        Slightly higher entropy coefficient to promote exploration

#training parameters
n_policy_updates = 30                                   #number of policy updates during training
total_timesteps = n_policy_updates*n_steps*n_envs       #number of environment steps during this training

#RND intinsic rewards parameters
RND_reward = True               #wether to use the RND_reward_callback to tackle sparse rewards problems
RND_learning_rate = 0.001       #0.0001 default
intrinsic_importance_coef = 100

#rendering parameters
show = True                         #render while learning?
sleep_time = 0.01                   #sleep time between each rendered frames (slow but clear vision)
period = 5                          #how many policy updates between two rendered rollouts (1: every rollout is rendered)
episodes_redered_by_rollout = 2     #how many episodes to render in a rendered rollout

#hardware parameters
device = 'cpu'


#------------------------------------------LOGS_DIRECTORIES_AND_LOG_NAME---------------------------------------#

def dir_creation(logs_dir):
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
#set the path to the directory that will store the logs
logs_dir = "./logs"
dir_creation(logs_dir)
logs_dir = logs_dir + '/' + env_name + '/'
dir_creation(logs_dir)
logs_dir = logs_dir + '/' + policy_name + '/'
dir_creation(logs_dir)
logs_dir = logs_dir + '/' + f"RND_reward={RND_reward}" + '/'
dir_creation(logs_dir)
if RND_reward:
    logs_dir = logs_dir + '/' + f"RND_learning_rate={RND_learning_rate}~intrinsic_importance_coef={intrinsic_importance_coef}" + '/'
    dir_creation(logs_dir)
#set the log directory path for the model and the tensorboard files
model_logs_dir = logs_dir+ '/' + "models" + '/'
dir_creation(model_logs_dir)
tensorboard_logs_dir = logs_dir+ '/' + "tensorboard" + '/'
dir_creation(tensorboard_logs_dir)

if not load:    #set the name of the logfile that will be created
    log_name = (f"PPO~learning_rate={learning_rate}"
                +f"~n_steps={n_steps}"
                +f"~batch_size={batch_size}"
                +f"~n_epochs={n_epochs}"
                +f"~gamma={gamma}"
                +f"~ent_coef={ent_coef}"
                +f"~n_policy_updates={n_policy_updates}"
                +str(datetime.today().strftime('%Y-%m-%d-%H-%M')))


#----------------------------------------CREATE_TRAINING_ENVIRONMENTS------------------------------------------#

# Define environment creation function
def make_env(env_name, render_mode, max_episode_steps):
    def _init():    #init is to create different instances intead of having the same environment every time
        env = gymnasium.make(env_name, render_mode=render_mode, max_episode_steps=max_episode_steps)
        return env
    return _init
#create vectorized environment
vec_env = DummyVecEnv([make_env(env_name, render_mode, max_episode_steps) for _ in range(n_envs)])


#---------------------------------------------PPO_MODEL_CALL---------------------------------------------------#

if load:    #load the model
    model = PPO.load(model_logs_dir+log_name,
                     vec_env,
                     device=device,
                     print_system_info=True,
                     tensorboard_log=tensorboard_logs_dir)

else:       #initialise PPO model with a MlpPolicy
    model = PPO(policy=policy, 
                env=vec_env,
                device=device,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                ent_coef=ent_coef,
                verbose=1,  
                tensorboard_log=tensorboard_logs_dir)


#----------------------------------------PPO_MODEL_TRAINING---------------------------------------------------#

#if RND callback is used, create the RNDModel network (from auxiliary_models.py)
if RND_reward:
    #check if the observation space is discrete or not to let the RNDModel know if the input needs embedding
    print(f"vec_env.observation_space: {vec_env.observation_space};"
         +f"                Type: {type(vec_env.observation_space)}")

    if isinstance(vec_env.observation_space, gymnasium.spaces.Discrete):
        observation_space_is_discrete = True
        observation_space_dim = vec_env.observation_space.n
    else:
        observation_space_is_discrete = False
        observation_space_dim = vec_env.observation_space.shape[0]
    # Initialize the RND model with the appropriate state dimension
    RND_model = RND_model.RND_model(observation_space_dim,
                                    intrinsic_importance_coef,
                                    RND_learning_rate,
                                    observation_space_is_discrete, verbose=0).to(device)

#train PPO model (the training tunes the MLP so that the policy maximises its reward)
model.learn(total_timesteps=total_timesteps,
            callback=[
                custom_callbacks.RND_reward_callback(RND_model, verbose=0),
                custom_callbacks.reward_logger_callback(verbose=0),
                custom_callbacks.model_saver_callback(model, 
                                                      model_logs_dir, 
                                                      log_name, 
                                                      verbose=0),
                custom_callbacks.render_callback(vec_env, 
                                                 model, 
                                                 n_policy_updates, 
                                                 show, 
                                                 sleep_time, 
                                                 period, 
                                                 episodes_redered_by_rollout, verbose=0)
                      ],
            tb_log_name=log_name)


#------------------------------------------------PPO_MODEL_SAVE------------------------------------------------#

model.save(model_logs_dir+log_name)

