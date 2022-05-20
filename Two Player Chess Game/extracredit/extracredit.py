import torch, random, math, json
import numpy as np
from extracredit_embedding import ChessDataset, initialize_weights

DTYPE=torch.float32
DEVICE=torch.device("cpu")

###########################################################################################
def forward(model, x):
    """Performs a forward pass through your neural net (evaluates f(x)).

    @param x: an (N, in_size) Tensor
    @return y: an (N, out_size) Tensor of output from the network
    """
    #raise NotImplementedError("You need to write this part!")
    y = model(x)
        
    return y #torch.ones(x.shape[0], 1)

def step(model, loss_fn, optimizer, x, y):
    """
    Performs one gradient step through a batch of data x with labels y.

    @param x: an (N, in_size) Tensor
    @param y: an (N,) Tensor
    @return L: total empirical risk (mean of losses) at this timestep as a float
    """
    #raise NotImplementedError("You need to write this part!")
        
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = forward(model, x)

    # Compute and print loss
    loss = loss_fn(y_pred, y)
    #if t % 100 == 99:
    #    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #L = loss.item()
    return

###########################################################################################
def trainmodel():
    # Well, you might want to create a model a little better than this...
    #model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(in_features=8*8*15, out_features=1))
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(8*8*15, 8*8*15),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(8*8*15, 8*8*15),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(8*8*15, 1))

    # ... and if you do, this initialization might not be relevant any more ...
    #model[1].weight.data = initialize_weights()
    #model[1].bias.data = torch.zeros(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    
    # ... and you might want to put some code here to train your model:
    trainset = ChessDataset(filename='extracredit_train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
    for epoch in range(2000):
        for x,y in trainloader:
            #pass # Replace this line with some code that actually does the training
            step(model, loss_fn, optimizer, x, y)
            
    # ... after which, you should save it as "model.pkl":
    torch.save(model, 'model.pkl')


###########################################################################################
if __name__=="__main__":
    trainmodel()
    
