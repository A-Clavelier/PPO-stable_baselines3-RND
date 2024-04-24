from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.policies import ActorCriticPolicy

import torch.nn as nn


#Basic MlpPolicy implemented with stable baselines 3's PPO class
class MlpPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class optimistic_init_value_policy (ActorCriticPolicy):
    def _build_mlp_extractor(self):
        super()._build_mlp_extractor()
        last_linear_layer = None
        # Find the last linear layer
        for module in self.mlp_extractor.value_net.modules():
            if isinstance(module, nn.Linear):
                last_linear_layer = module
        # Initialize the bias of the last linear layer optimistically to promote exploration (not explored action-states have a higher predicted value)
        if last_linear_layer is not None:
            nn.init.constant_(last_linear_layer.bias, 5.0)
