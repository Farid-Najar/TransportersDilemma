import sys
import os
from copy import deepcopy
# direc = os.path.dirname(__file__)
from pathlib import Path
path = Path(os.path.dirname(__file__))
# pri&
# caution: path[0] is reserved for script path (or '' in REPL)
# print(str(path)+'/ppo')
sys.path.insert(1, str(path.parent.absolute()))#'/Users/faridounet/PhD/TransportersDilemma')

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from gymnasium import spaces
import gymnasium as gym
import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from assignment import CombActionEnv, RemoveActionEnv, AssignmentGame, NormalizedEnv

class NN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, 
                 observation_space: spaces.Box, 
                 hidden_layers = [2048, 2048, 1024, 512],
                 n_actions: int = 100):
        super().__init__(observation_space, n_actions)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_inputs = observation_space.shape[0]
        hidden_layers.insert(0, n_inputs)
        layers = []
        for l in range(len(hidden_layers)-1):
            layers += [
                nn.Linear(hidden_layers[l], hidden_layers[l+1]),
                nn.ReLU()
            ]
            
        layers += [
            nn.Linear(hidden_layers[-1], n_actions),
            nn.Sigmoid()
        ]

        self.linear = nn.Sequential(
            *layers
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(observations)
    

def train(x, y, epochs = 10):
    mps_device = torch.device("mps")
    
    pivot = 2048
    print(x[:pivot, :].shape)
    print(y[:pivot, :].shape)
    training_set = TensorDataset(x[:pivot, :], y[:pivot, :])
    validation_set = TensorDataset(x[pivot:, :], y[pivot:, :])
    
    model = NN(spaces.Box(0, 1, (x.shape[1],))).to(mps_device)
    loss_fn = nn.MSELoss()#nn.BCELoss(reduction='mean')#
    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = DataLoader(
        training_set, batch_size=256, shuffle=True, num_workers=os.cpu_count(),
    )
    validation_loader = DataLoader(
        validation_set, batch_size=256, shuffle=False, num_workers=os.cpu_count(),
    )

    optimizer = torch.optim.Adam(model.parameters())
    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            # inputs.to(mps_device)
            # labels.to(mps_device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs.to(mps_device)).to(mps_device)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels.to(mps_device))
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 8 == 7:
                last_loss = running_loss / 8 # loss per batch
                # print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    best_vloss = 1_000_000.

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch, writer)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                # vinputs.to(mps_device)
                # vlabels.to(mps_device)
                voutputs = model(vinputs.to(mps_device))#.round().to(mps_device)
                vloss = loss_fn(voutputs, vlabels.to(mps_device))#torch.sum(voutputs == vlabels.to(mps_device))/vlabels.nelement()
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

    

if __name__ == '__main__':
    # mps_device = torch.device("mps")
    # torch.set_default_device(mps_device)
    
    path = 'TransportersDilemma/RL/'
    x = np.load(path+'x_K100.npy')
    y = np.load(path+'y_K100.npy')
    x /= np.amax(x, axis=1).reshape(-1, 1)
    assert len(x) == len(y)
    print(x.shape)
    
    train(torch.Tensor(x), torch.Tensor(y), 500)