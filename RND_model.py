import torch
import torch.nn as nn
import torch.nn.functional as F

#this file stores pytorch neural network models. 
#Other scripts can import it to create instances of networks defined here (eg. RND)


#---TACKLING REWARD SCARCITY---#

#Random Network Distillation (RND): RND works by training a predictor network to match the outputs
#                                   of a fixed, randomly initialized target network. The difference
#                                   in output (prediction error) serves as an intrinsic reward. 
#                                   The logic is that unfamiliar inputs will likely produce higher 
#                                   prediction errors, thereby encouraging the agent to explore them.
class RND_model(nn.Module):
    def __init__(self, input_dim, intrinsic_importance_coef, RND_learning_rate, input_is_discrete, verbose=0):
        super(RND_model, self).__init__()
        self.verbose=verbose
        self.intrinsic_importance_coef=intrinsic_importance_coef

        # Tackle discrete input problem by adding an embedding layer
        self.input_is_discrete = input_is_discrete
        if self.verbose>=1: print(f"RND input is discrete: {input_is_discrete}")
        if input_is_discrete:
            self.embed = nn.Embedding(num_embeddings=input_dim, embedding_dim=10)  # embedding dim 10
            input_dim = 10  # Output dimension of embedding layer becomes the input dim for the next layers
        
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

        # Tackle discrete input problem by adding an embedding layer
        if self.input_is_discrete:
            x = self.embed(x.long())    # Ensure x is long type for embedding
    
        pred = self.predictor(x)        #get predictor Network Output and compute gradient
        with torch.no_grad():           #tell PyTorch not to compute or store gradients within this block. 
            target = self.target(x)     #get target Network Output without computing gradient and updating wheights
        loss = F.mse_loss(pred, target) #loss is MSE(mean square error) between prediction and target.
        
        if self.verbose>=1:print(f"\n    RND_input={x}"
                                +f"\n    RND pred: {pred}" 
                                +f"\n    RND target: {target}" 
                                +f"\n    RND loss: {loss}")
        
        return loss
    
    def train_step_RND(self, input):
        self.train()                # Set the model to training mode
        self.optimizer.zero_grad()  # Clear gradients

        # Tackle discrete input problem
        if self.input_is_discrete:
            input = torch.tensor([input], dtype=torch.long, device=self.device)

        loss = self.forward(input)  # Compute the loss
        loss.backward()             # Backpropagate the gradients
        self.optimizer.step()       # Update the predictor network parameters

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
    observation_space_is_discrete = False
    state_dim = 2
    RND_model = RND_model(state_dim, observation_space_is_discrete,verbose=0)
    random_tensor = torch.rand(2)                       #fix a random tensor
    for _ in range(500):
        loss=RND_model.train_step_RND(random_tensor)    #train the RND several times with the same input
        print(loss)
    #after several trainings with the same input, the loss should be 0
