# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP5. You should only modify code
within this file and neuralnet_part1.py,neuralnet_leaderboard -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        # raise NotImplementedError("You need to write this part!")
        self.lrate = lrate
        # self.model = nn.Sequential(torch.nn.Linear(in_size, 110), torch.nn.ReLU(), torch.nn.Linear(110, out_size))
        self.model = nn.Sequential(
          nn.Conv2d(3,6,4),
          nn.ReLU(),
          nn.MaxPool2d(2,2),
          nn.Conv2d(6,16,5),
          nn.ReLU(),
          nn.MaxPool2d(2,2),
          nn.Flatten(1),
          torch.nn.Linear(5*5*16, 170), 
          torch.nn.ReLU(), 
          torch.nn.Linear(170, 84),
          torch.nn.ReLU(), 
          torch.nn.Linear(84, out_size),
        )
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = lrate)
        
    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        return self.model(x.reshape(x.shape[0],3,31,31))

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        yhats = self.forward(x)
        loss = self.loss_fn(yhats,y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    rate = .0005
    lf = torch.nn.CrossEntropyLoss()
    classifier = NeuralNet(rate, lf, 2883,4) #if used for future applications 

    listOfLosses = []
    total = torch.cat((train_set,dev_set))

    mean = torch.mean(total, 0)
    stdev = torch.std(total,0)

    trainNormed = (train_set-mean)/(stdev)

    for i in range(epochs):
        for a in range(train_set.size(0)//batch_size):
            perm = torch.randperm(trainNormed.size(0))
            temp = perm[a*batch_size: (a+1) * batch_size]
            batch = trainNormed[temp]
            label_batch = train_labels[temp]
            listOfLosses.append(classifier.step(batch, label_batch))

    toReturn = np.ones(len(dev_set))
    normdev = (dev_set-mean)/(stdev)

    predictions = classifier(normdev).detach().numpy()

    for i in range(len(predictions)):
        toReturn[i] = int(np.argmax(predictions[i]))

    return listOfLosses,toReturn.astype(int), classifier
