import torch
import torch.nn as nn
import torch.nn.functional as F

#---TACKLING REWARD SCARCITY---#
#Random Network Distillation (RND): RND works by training a predictor network to match the outputs
#                                   of a fixed, randomly initialized target network. The difference
#                                   in output (prediction error) serves as an intrinsic reward. 
#                                   The logic is that unfamiliar inputs will likely produce higher 
#                                   prediction errors, thereby encouraging the agent to explore them.
class RND_model(nn.Module):
    def __init__(self, input_dim, intrinsic_importance_coef, RND_learning_rate, verbose=0):
        super(RND_model, self).__init__()
        self.verbose=verbose
        self.intrinsic_importance_coef=intrinsic_importance_coef
        #create predictor and target networks with random weights
        self.predictor = nn.Sequential(
                                        nn.Linear(input_dim, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 512),
                                        nn.ReLU()
                                        )
        self.target =  nn.Sequential(
                                        nn.Linear(input_dim, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 512),
                                        nn.ReLU()
                                        )
        #Freeze the target network so it does not train
        for p in self.target.parameters():
            p.requires_grad = False     
        #Initialize optimizer for the predictor network
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=RND_learning_rate)

    def forward(self, x): 
        pred = self.predictor(x)        #get predictor Network Output and compute gradient
        with torch.no_grad():           #tell PyTorch not to compute or store gradients within this block. 
            target = self.target(x)     #get target Network Output without computing gradient and updating wheights
        loss = F.mse_loss(pred, target, reduction='none') #loss is MSE(mean square error) between prediction and target.
        loss = loss.mean(dim=1)
        if self.verbose>=1:print(f"\n    RND_input={x}"
                                +f"\n    RND pred: {pred}" 
                                +f"\n    RND target: {target}" 
                                +f"\n    RND loss: {loss}")
        return loss
    
    def train_step_RND(self, input):
        self.train()                # Set the model to training mode
        self.optimizer.zero_grad()  # Clear gradients
        loss = self.forward(input)  # Compute the loss
        # Create a tensor of ones with the same shape as the loss tensor, which is needed for backward()
        grad_tensors = torch.ones_like(loss)
        # Backpropagate the gradients for each element independently
        loss.backward(gradient=grad_tensors)          
        self.optimizer.step()
        return loss.detach().numpy()*self.intrinsic_importance_coef
    
        #intrinsic reward is a conversion of the loss tensor into numpy array.
        #(for reward concatenation compatibility) 
        #And the loss tensor computed in forward()
        #is the MSE(mean square error) between prediction and target.
        #The MSE quantifies how well the predictor has learned to mimic the target. 
        #The intuition here is that for familiar inputs, the predictor should have low error 
        #because it has learned these inputs well. For novel inputs, the error will be higher, 
        #and thus exploration is rewarded.


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    iterations = 300
    window_size = 10
    indices = range(iterations-window_size+1)
    input_dim = 2
    intrinsic_importance_coef = 10000
    LRS = [0.001,0.0005, 0.0001, 0.00005,0.00001]

    def moving_average(data, window_size):
        """ Compute the moving average using a specified window size. """
        if window_size > len(data):
            raise ValueError("Window size must be less than or equal to the number of data points.")
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


    for RND_learning_rate in LRS:
        RND = RND_model(input_dim, intrinsic_importance_coef, RND_learning_rate, verbose=0)
        losses = []
        
        for _ in range(iterations):
            random_tensor = torch.rand(1,input_dim)
            loss=RND.train_step_RND(random_tensor)
            losses.append(loss[0])
        smoothed_losses = moving_average(losses, window_size)
        plt.plot(indices,smoothed_losses, label=f"lr={RND_learning_rate}")
    plt.legend()
    plt.show()
    #after several trainings with the same input, the loss should be close to 0
