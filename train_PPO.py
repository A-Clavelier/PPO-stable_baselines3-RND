import gym
#import gym_sokoban


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import custom_policies
import custom_callbacks
import RND_model

from datetime import datetime
import os


#---LOAD_MODEL_OR_NEW_MODEL?---#

load = False     #choose wether to load a model to continue training or create a new model
if load:         #choose the logfile to load
        log_name = ""


#---PARAMETERS---#

#env parameters
n_envs = 1                      #define the number of environments running in parallel
env_name = 'CartPole-v1'        #define the environment you want to create an agent for
render_mode = "rgb_array"       
max_episode_steps=512
env_package = 'gym'             #"gym" or "gymnasium"

#PPO model creation parameters (only used if load==False)
policy_name = "MlpPolicy"                               #define policy of the PPO with the custom_policies script(default: "MlpPolicy")                 
policy = getattr(custom_policies, policy_name)          #policy = custom_policies.[policy_name]
                                #default values
learning_rate = 0.0003          #0.0003     Reduce if results are unstable
n_steps =  8192                 #2048       number of steps of a rollout (between each policy updates) 
batch_size = 256                #64         Increase for more stable but slower learning (the info of the n_steps is spread into multiple batches to perform the policy update) 
n_epochs = 20                   #10         number of optimization epochs to perform on each batch of samples
gamma = 0.99                    #0.99       Discount factor Adjust according to the importance of future rewards
ent_coef = 0.01                 #0.0        Slightly higher entropy coefficient to promote exploration

#training parameters
n_policy_updates = 5                                   #number of policy updates during training
total_timesteps = n_policy_updates*n_steps*n_envs       #number of environment steps during this training
RND_reward = True                                       #wether to use the RND_reward_callback to tackle sparse rewards problems

#rendering parameters
show = True            #render while learning?
sleep_time = 0         #sleep time between each rendered frames (slow but clear vision)
period = 5             #how many policy updates between two rendered rollouts (1: every rollout is rendered)

#hardware parameters
device = 'cpu'

#---LOGS_DIRECTORIES_AND_LOG_NAME---#

logs_dir = "./logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
logs_dir = logs_dir + '/' + env_name + '/'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
logs_dir = logs_dir + '/' + policy_name + '/'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
logs_dir = logs_dir + '/' + f"RND_reward={RND_reward}" + '/'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

model_logs_dir = logs_dir+ '/' + "models" + '/'
tensorboard_logs_dir = logs_dir+ '/' + "tensorboard" + '/'

if not load:    #set the name of the logfile that will be created
    log_name = (f"PPO~learning_rate={learning_rate}"
                +f"~n_steps={n_steps}"
                +f"~batch_size={batch_size}"
                +f"~n_epochs={n_epochs}"
                +f"~gamma={gamma}"
                +f"~ent_coef={ent_coef}"
                +f"~n_policy_updates={n_policy_updates}"
                +str(datetime.today().strftime('%Y-%m-%d-%H-%M')))


#---SET_TRAINING_ENVIRONMENTS---#

dict_envs = {}
for i in range(n_envs):
    dict_envs[f"env_{i}"] = gym.make(env_name,render_mode=render_mode, max_episode_steps=max_episode_steps)

L_envs_funcs = [lambda: env for env in dict_envs.values()]
vec_env = DummyVecEnv(L_envs_funcs)

print(f" \n dict of environments:{dict_envs} \n ")

#---PPO_MODEL_CALL---#

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


#---PPO_MODEL_TRAINING---#

#if RND callback is used, create the RNDModel network (from auxiliary_models.py)
if RND_reward:
    #check if the observation space is discrete or not to let the RNDModel know if the input needs embedding
    print(f"vec_env.observation_space: {vec_env.observation_space};"
         +f"                Type: {type(vec_env.observation_space)}")

    if isinstance(vec_env.observation_space, gym.spaces.Discrete):
        observation_space_is_discrete = True
        observation_space_dim = vec_env.observation_space.n
    else:
        observation_space_is_discrete = False
        observation_space_dim = vec_env.observation_space.shape[0]
    # Initialize the RND model with the appropriate state dimension
    RND_model = RND_model.RND_model(observation_space_dim, observation_space_is_discrete, verbose=0).to(device)

#train PPO model (the training tunes the MLP so that the policy maximises its reward)
model.learn(total_timesteps=total_timesteps,
            callback=[
                custom_callbacks.RND_reward_callback(RND_model, verbose=0),
                custom_callbacks.reward_logger_callback(verbose=0),
                custom_callbacks.model_saver_callback(model, model_logs_dir, log_name, verbose=0),
                custom_callbacks.render_callback(dict_envs, model, n_policy_updates, show, sleep_time, period, verbose=1)
                      ],
            tb_log_name=log_name)


#---PPO_MODEL_SAVE---#

model.save(model_logs_dir+log_name)

