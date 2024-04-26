from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()   # Turn on interactive mode
import numpy as np
np.set_printoptions(precision=3, floatmode='fixed') # Set global print options for numpy arrays

""" This custom callback is responsible for: 
        -the addition of RND intrinsic reward to the extrinsic reward when specified
        -the logging of mean (extr/intr/tot) reward by episode
        -the rendering when specified
        -saving the model after a rollout when a new best mean extr reward by episode is scored"""

class custom_callback(BaseCallback):
    def __init__(self, 
                 RND_reward, RND_model,
                 PPO_model, PPO_model_logs_dir, log_name,
                 show, n_envs, vec_env, sleep_time, period, episodes_redered_by_rollout, n_policy_updates,
                 verbose=0):
        super(custom_callback, self).__init__(verbose)
        #adjust quantity of printed info
        self.verbose=verbose

        # RND model that computes the intrinsic reward
        self.RND_reward=RND_reward
        self.RND_model=RND_model

        #PPO_model
        self.PPO_model=PPO_model
        self.PPO_model_logs_dir=PPO_model_logs_dir
        self.log_name=log_name

        #Rendering
        self.show=show
        self.n_envs=n_envs
        self.vec_env=vec_env
        self.sleep_time=sleep_time
        self.period=period
        self.episodes_redered_by_rollout=episodes_redered_by_rollout
        self.n_policy_updates=n_policy_updates
        self.rollout_length = self.PPO_model.n_steps
        self.rollout_number = 1
        self.episode_number = 1

        #------------ reward logging attributes------------#
        # arrays containing the current (extr/intr/tot) reward value for the episode of each env (len=n_envs)       
        self.current_extrinsic_rewards = np.zeros(n_envs, dtype=np.float32)     
        self.current_intrinsic_rewards = np.zeros(n_envs, dtype=np.float32)      
        self.current_total_rewards = np.zeros(n_envs, dtype=np.float32)
        # lists containing each episode's ending (extr/intr/tot) reward value (len=n_episodes)
        # (when one environment of the n_ens is finished it is a new episode)
        self.episode_extrinsic_reward_list = []
        self.episode_intrinsic_reward_list = []
        self.episode_total_reward_list = []
        #track the best mean extrinsic reward for an episode
        self.Best_mean_ep_extr_reward = None


    def calculate_intrinsic_reward(self,  obs_tensor):
        # The RND model trains its predictor network with current observation.
        # The loss for this training step is the MSE between predictor network and a fixed target network.
        # So the predictor learns to replicate the target to minimize the loss 
        # on this specific observation input.
        # It also returns a numpy array of the loss from this learning step.
        # We use it as intrinsic reward since high MSE between predictor and target means 
        # new observation that should be explored and is therefore rewarded.
        #pass the obs_tensor through the RND model to get intrinsic reward
        intrinsic_reward = self.RND_model.train_step_RND(obs_tensor)  
        return intrinsic_reward
    

    def log_mean_of_rew_list(self,  log_name: str, rew_list):
        if rew_list:    #check the list is not empty                  
            mean_reward = np.mean(rew_list)  
            self.logger.record(log_name, mean_reward)
        return([])      #reset the rewards list for next rollout


    def _on_training_start(self) -> None:
        print("=======================TRAINING_START======================= \n")
        self.training_progress_bar = tqdm(total=self.n_policy_updates, desc="training", leave=False, dynamic_ncols=True)


    def _on_rollout_start(self) -> None:
        #new rollout_progress_bar creation
        self.rollout_progress_bar = tqdm(total=self.rollout_length, desc=f"rollout n°{self.rollout_number}", leave=False, dynamic_ncols=True)


    def _on_step(self) -> bool:
        # Render the environment if show 
        # and if rollout number is a multiple of period 
        # and if episode number is below the number of episodes we want rendered by rollout
        if (self.show 
            and self.rollout_number%self.period==0 
            and self.episode_number<=self.episodes_redered_by_rollout):
            plt.clf()
            plt.title(f"rollout n°{self.rollout_number} - episode n°{self.episode_number}")
            plt.imshow(self.vec_env.render())
            plt.draw()
            plt.pause(self.sleep_time) 
        # get the extrinsic reward (from the environment) for this step
        step_extrinsic_reward = self.locals['rewards']
        # get the last observation from the environment
        obs_tensor = self.locals['obs_tensor']
        if self.verbose>=5: print(f"\nobs_tensor={obs_tensor}")
        # get the intrinsic reward for this step
        if self.RND_reward:     #check if we are using RND and 
            # compute the intrinsic reward for the current observation
            step_intrinsic_reward = self.calculate_intrinsic_reward(obs_tensor)
        else:   #if we are not using RND there is no intrinsic reward
            step_intrinsic_reward = 0
        # get the total reward for this step
        step_total_reward = step_extrinsic_reward + step_intrinsic_reward
        # set the new reward as the total reward
        self.locals['rewards'] = step_total_reward
        if self.verbose>=4: print(f"\nstep    (/extr/intr/tot) rew= "
                                +f"/{step_extrinsic_reward}"
                                 +f"/{step_intrinsic_reward}"
                                 +f"/{step_total_reward}")
            

        #### MANAGEMENT OF REWARD LOGGING ATTRIBUTES ########################################################
        self.current_extrinsic_rewards += step_extrinsic_reward     # Accumulate extrinsic rewards          #
        self.current_intrinsic_rewards += step_intrinsic_reward     # Accumulate intrinsic rewards          #
        self.current_total_rewards += step_total_reward             # Accumulate total rewards              #
        if self.verbose>=3: print(f"current (/extr/intr/tot) rew= "                                         #
                                 +f"/{self.current_extrinsic_rewards}"                                      #
                                 +f"/{self.current_intrinsic_rewards}"                                      #
                                 +f"/{self.current_total_rewards}")                                         #
        # Check if an episode ended for one env                                                             #
        if self.locals['dones'].any():                                                                      #
            #update episode counter                                                                         #
            self.episode_number += 1                                                                        #
            # Skim through the environments to select the env that ended its episode                        #
            for i in range (len(self.locals['dones'])):                                                     #
                if self.locals['dones'][i]==True:                                                           #
                    if self.verbose>=2: print(f"\n\n-------------EPISODE END FOR ENV {i+1}-------------"    #
                                             +f"\nenv{i+1} ep rew: "                                        #
                                             +f"\nextr    -> {self.current_extrinsic_rewards[i]}"           #
                                             +f"\nintr    -> {self.current_intrinsic_rewards[i]}"           #
                                             +f"\ntot     ->{self.current_total_rewards[i]}"                #
                                             +f"\n-------------NEW EPISODE FOR ENV {i+1}-------------"      #
                                             +f"\nEPISODE N°{self.episode_number}\n\n")                     #
                    # append the current rewards of the env that ended its episode                          #
                    # to their respective episode rewards list for this rollout                             #
                    self.episode_extrinsic_reward_list.append(self.current_extrinsic_rewards[i])            #
                    self.episode_intrinsic_reward_list.append(self.current_intrinsic_rewards[i])            #
                    self.episode_total_reward_list.append(self.current_total_rewards[i])                    #
                    # Reset the current rewards counters of this env for its next episode                   #
                    self.current_extrinsic_rewards[i] = 0                                                   #
                    self.current_intrinsic_rewards[i] = 0                                                   #
                    self.current_total_rewards[i] = 0                                                       #
        ##################################################################################################### 
        # update the progress bar
        self.rollout_progress_bar.update(1)
        return True


    def _on_rollout_end(self) -> None:
        self.vec_env.reset()    #reset environments for new episodes next rollout
        self.rollout_number+=1  #update rollout counter
        self.episode_number=1   #reset episode counter   
        # update the progress bars
        self.rollout_progress_bar.close()
        self.training_progress_bar.update(1)
        if self.verbose>=1: print(f"\n\n####################_ROLLOUT_END_####################"                   
                                 +f"\nep_extr_rew_list of rollout= {self.episode_extrinsic_reward_list}"        
                                 +f"\nep_intr_rew_list of rollout= {self.episode_intrinsic_reward_list}"
                                 +f"\nep_tot_rew_list of rollout= {self.episode_total_reward_list}"
                                 +f"\n####################_NEW_ROLLOUT_####################\n\n") 
            
        ################### SAVING ######################################################################################################
        if self.Best_mean_ep_extr_reward == None :                                                                                      #
            self.Best_mean_ep_extr_reward = np.mean(self.episode_extrinsic_reward_list)                                                 #
        #save the PPO model in the model logs if the mean extrinsic reward for an episode is better than the last best one              #
        if np.mean(self.episode_extrinsic_reward_list)>=self.Best_mean_ep_extr_reward:                                                  #
            self.Best_mean_ep_extr_reward = np.mean(self.episode_extrinsic_reward_list)                                                 #
            self.model.save(self.PPO_model_logs_dir+self.log_name)                                                                      #
            if self.verbose>=1: print("\n\nNEW BEST MEAN EXTRINSIC REWARD FOR AN EPISODE!!!!!!!\n\n")                                   #
        #################################################################################################################################

        ################### LOGGING #####################################################################################################
        #compute and log the mean episode (extr,intr,tot) reward for this rollout, then reset their list                                #
        self.episode_extrinsic_reward_list=self.log_mean_of_rew_list('rollout/mean_ep_extr_reward',self.episode_extrinsic_reward_list)  #  
        self.episode_intrinsic_reward_list=self.log_mean_of_rew_list('rollout/mean_ep_intr_reward',self.episode_intrinsic_reward_list)  # 
        self.episode_total_reward_list=self.log_mean_of_rew_list('rollout/mean_ep_tot_reward',self.episode_total_reward_list)           #    
        #################################################################################################################################


    def _on_training_end(self) -> None:
        self.rollout_progress_bar.close()
        self.training_progress_bar.close()
        print("=======================TRAINING_END======================= \n")




