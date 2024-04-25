from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()   # Turn on interactive mode
import numpy as np


class RND_reward_callback(BaseCallback):
    """
    Custom callback implementing intrinsinc rewards for exploration through RND.
    """
    def __init__(self, RND_model, verbose=0):
        super(RND_reward_callback, self).__init__(verbose)
        # RND model that computes the intrinsic reward
        self.RND_model = RND_model  
        # reward logging attributes                
        self.current_episode_intrinsic_reward = 0
        self.current_episode_extrinsic_reward = 0
        self.episode_intrinsic_rewards_list = []
        self.episode_extrinsic_rewards_list = []
        self.verbose=verbose
    
    def calculate_intrinsic_reward(self,  obs_tensor):
        # The RND model trains its predictor network with current observation.
        # The loss for this training step is the MSE between predictor network and a fixed target network.
        # So the predictor learns to replicate the target to minimize the loss 
        # on this specific observation input.
        # It also returns a numpy array of the loss from this learning step.
        # We use it as intrinsic reward since high MSE between predictor and target means 
        # new observation that should be explored and is therefore rewarded.
        if self.verbose>=2: print(f"\nobs_tensor={obs_tensor}")
        intrinsic_reward = self.RND_model.train_step_RND(obs_tensor)     #pass the obs_tensor through the RND model
        return intrinsic_reward
    
    def _on_step(self):

        # Get the last observation from the environment
        obs_tensor = self.locals['obs_tensor']
        # Compute the intrinsic reward for the current observation
        intrinsic_reward = self.calculate_intrinsic_reward(obs_tensor)
        # get the extrinsic reward (from the environment) for this step
        extrinsic_reward = self.locals['rewards']
        
        #check if the two reward arrays are of the same shape
        if intrinsic_reward.shape != extrinsic_reward.shape:
            # Assuming intrinsic_reward should be broadcasted to the shape of extrinsic_reward
            intrinsic_reward = np.broadcast_to(intrinsic_reward, extrinsic_reward.shape)
        
        # Add the intrinsic reward to the current step's reward
        self.locals['rewards'] = intrinsic_reward + extrinsic_reward

        if self.verbose>=2: print(f"intrinsic reward={intrinsic_reward}"
                                 +f"\nextrinsic reward={extrinsic_reward}"
                                 +f"\ntotal reward={self.locals['rewards']}\n")

        #### MANAGEMENT OF LOGGING ATTRIBUTES ###########################################################
        self.current_episode_intrinsic_reward += intrinsic_reward  # Accumulate intrinsic rewards       #
        self.current_episode_extrinsic_reward += extrinsic_reward  # Accumulate extrinsic rewards       #
        # Check if the episode ended                                                                    #
        if self.locals['dones'][0]:                                                                     #
            if self.verbose>=1: print("\n-------------EPISODE END-------------"                         #
                                    +f"\nep_intr_rew={self.current_episode_intrinsic_reward}"           #
                                    +f"\nep_extr_rew={self.current_episode_extrinsic_reward}"           #
                                    + "\n-------------NEW EPISODE-------------\n")                      #
            # add current episode rewards to their episode rewards list for this rollout                #
            self.episode_intrinsic_rewards_list.append(self.current_episode_intrinsic_reward)           #
            self.episode_extrinsic_rewards_list.append(self.current_episode_extrinsic_reward)           #
            # Reset the rewards counter for the next episode                                            #
            self.current_episode_intrinsic_reward = 0                                                   #
            self.current_episode_extrinsic_reward = 0                                                   #
        #################################################################################################
        
        return True
    
    def _on_rollout_end(self) -> None:
        ################### LOGGING #####################################################################
        #compute and log the mean episode intrinsic reward for this rollout                             #
        if self.episode_intrinsic_rewards_list:                                                         #
            mean_ep_intrinsic_reward = np.mean(self.episode_intrinsic_rewards_list)                     #
            self.logger.record('rollout/mean_ep_intrinsic_reward', mean_ep_intrinsic_reward)            #                  #
        self.episode_intrinsic_rewards_list = []     #reset the rewards list for next rollout           #
                                                                                                        #
        #compute and log the mean episode extrinsic reward for this rollout                             #
        if self.episode_extrinsic_rewards_list:                                                         #
            mean_ep_extrinsic_reward = np.mean(self.episode_extrinsic_rewards_list)                     #
            self.logger.record('rollout/mean_ep_extrinsic_reward', mean_ep_extrinsic_reward)            #
        self.episode_extrinsic_rewards_list = []     #reset the rewards list for next rollout           #
        #################################################################################################



class reward_logger_callback(BaseCallback):
    """
    Custom callback for logging episodic rewards.
    """
    def __init__(self, verbose=0):
        super(reward_logger_callback, self).__init__(verbose)
        self.current_episode_reward = 0     # Initialize reward accumulator for an episode
        self.episode_rewards_list = []      # List to store rewards of each episode of a rollout

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]    # Accumulate rewards
        # Check if the episode ended
        if self.locals['dones'][0]:
            # add current episode reward to the episode reward list of this rollout                            
            self.episode_rewards_list.append(self.current_episode_reward)
            # Reset the reward counter for the next episode
            self.current_episode_reward = 0          
        return True
    
    def _on_rollout_end(self) -> None:
        if self.episode_rewards_list:                               # Ensure there are rewards to process
            mean_ep_reward = np.mean(self.episode_rewards_list)        #compute mean of episode rewards of the rollout
            self.logger.record('rollout/mean_ep_reward', mean_ep_reward)  #log the mean episode reward for this rollout
        self.episode_rewards_list = []                              #reset the List to store rewards of each episode of a rollout



class model_saver_callback(BaseCallback):
    """
    Custom callback for saving model every rollout.
    """
    def __init__(self, model, model_logs_dir, log_name, verbose=0):
        super(model_saver_callback, self).__init__(verbose)
        self.model=model
        self.model_logs_dir =model_logs_dir
        self.log_name =log_name

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.model.save(self.model_logs_dir+self.log_name)



class render_callback(BaseCallback):
    """
    Custom callback for rendering model.
    """
    def __init__(self, vec_env, model, n_policy_updates, show, sleep_time, period, episodes_redered_by_rollout, verbose=0):
        super(render_callback, self).__init__(verbose)
        self.vec_env = vec_env
        self.model = model
        self.n_policy_updates = n_policy_updates
        self.show = show
        self.sleep_time = sleep_time
        self.period = period
        self.episodes_redered_by_rollout=episodes_redered_by_rollout
        self.verbose=verbose
        self.rollout_length = self.model.n_steps
        self.rollout_number = 1
        self.episode_number = 1

    def _on_training_start(self) -> None:
        print("=======================TRAINING_START======================= \n")
        self.training_progress_bar = tqdm(total=self.n_policy_updates, desc="training iterations progress",dynamic_ncols=True)

    def _on_rollout_start(self) -> None:
        #new rollout_progress_bar creation
        self.rollout_progress_bar = tqdm(total=self.rollout_length, desc=f"timesteps progress of rollout n°{self.rollout_number}", dynamic_ncols=True)

    def _on_step(self) -> bool:
        if self.verbose>=1: 
            print(f"\nepisode_number= {self.episode_number}")
            print(f"{self.show},{self.rollout_number%self.period==0},{self.episode_number<=self.episodes_redered_by_rollout}")
        # Render the environment if show 
        # and if rollout number is a multiple of period 
        # and if episode number is below the number of episodes we waant to render in each rendered rollout
        if (self.show and self.rollout_number%self.period==0 and self.episode_number<=self.episodes_redered_by_rollout):
            if self.verbose>=1: print(f"rendering vec_env...")
            plt.clf()
            plt.title(f"rollout n°{self.rollout_number} - episode n°{self.episode_number}")
            plt.imshow(self.vec_env.render())
            plt.draw()
            plt.pause(self.sleep_time)  # Pause for interaction
                
        
        # Check if the episode ended
        if self.locals['dones'].any():
            #update episode counter
            self.episode_number += 1

        # update the progress bar
        self.rollout_progress_bar.update(1)

        return True
    
    def _on_rollout_end(self) -> None:
        #reset environments to start next rollout on new episodes
        self.vec_env.reset()
        #update rollout counter
        self.rollout_number+=1
        #reset episode counter
        self.episode_number = 1      
        # update the progress bars
        self.rollout_progress_bar.close()
        self.training_progress_bar.update(1)

    def _on_training_end(self) -> None:
        self.rollout_progress_bar.close()
        self.training_progress_bar.close()
        print("=======================TRAINING_END======================= \n")
