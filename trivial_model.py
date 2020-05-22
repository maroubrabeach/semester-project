import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils
import torch.nn as nn
# import torch.nn.functional as F
from os import listdir
from os.path import isfile, join
import pdb
import logging

DATA_PATH = "./xfoil/data"


# Regress C_L, C_D from camber, distance, thickness

dir = listdir(DATA_PATH)
trainFrac = 0.8
trainSize = int(round(trainFrac*len(dir), 0))

class XfoilTrainDataset(Dataset):
    """Xfoil train dataset."""

    def __init__(self, data_path = DATA_PATH):
        """
        Args:
            data_path (string): Path to folder with profiles.
        """
        trainData = dir[:trainSize]
        self.data = [join(data_path, f) for f in trainData if isfile(join(data_path, f)) and f.endswith(".npy")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dataTrain = np.load(self.data[idx], allow_pickle=True)
        input = torch.cat((torch.tensor(dataTrain.item()["c"]).float(), torch.tensor(dataTrain.item()["d"]).float(), torch.tensor(dataTrain.item()["t"]).float()))
        output = torch.cat((torch.tensor(dataTrain.item()["cl"]).float(), torch.tensor(dataTrain.item()["cd"]).float()))
        return input, output

class XfoilTestDataset(Dataset):
    """Xfoil test dataset."""

    def __init__(self, data_path = DATA_PATH):
        """
        Args:
            data_path (string): Path to folder with profiles.
        """
        testData = dir[trainSize:]
        self.data = [join(data_path, f) for f in testData if isfile(join(data_path, f)) and f.endswith(".npy")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dataTest = np.load(self.data[idx], allow_pickle=True)
        input = torch.cat((torch.tensor(dataTest.item()["c"]).float(), torch.tensor(dataTest.item()["d"]).float(), torch.tensor(dataTest.item()["t"]).float()))
        output = torch.cat((torch.tensor(dataTest.item()["cl"]).float(), torch.tensor(dataTest.item()["cd"]).float()))
        return input, output

# https://medium.com/biaslyai/pytorch-linear-and-logistic-regression-models-5c5f0da2cb9#c317
# class DumbRegressor(nn.Module):
#
#     def __init__(self):
#         super(DumbRegressor, self).__init__()
#         self.fc = nn.Linear(2*160, 3)
#
#     def forward(self, x):
#         output = self.fc(x)
#         return output

# class FC(nn.Module):
#     """docstring for FC."""
#
#     def __init__(self):
#         super(FC, self).__init__()
#         self._fc1 = nn.Linear(3, 10)
#         self._fc2 = nn.Linear(10, 2)
#
#     def forward(self, x)
#         x = F.relu(self._fc1(x))
#         return self._fc2(x)


if __name__ == "__main__":

    # initialize data
    xfoil_trainData = XfoilTrainDataset()
    xfoil_loader = data_utils.DataLoader(
        xfoil_trainData,
        batch_size=2,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )
    xfoil_testData = XfoilTestDataset()
    # initialize network/model
    model = nn.Sequential(
          nn.Linear(3, 10),
          nn.ReLU(),
          nn.Linear(10, 2),
        )

    # what are the parameters of the model we want to optimize
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    n_epochs = 10
    # how many NumEpochs

    for epoch in range(0, n_epochs):

        print("epoch {}...".format(epoch))
        model.train()

        for input, output in xfoil_loader:
            # clean gradients that might be stored in parameters
            optimizer.zero_grad()
            # run forward pass
            prediction = model(input)
            # compute loss function
            loss = torch.mean((prediction-output)**2)
            print("loss {}...".format(loss.detach().numpy()))
            # compute backward pass
            loss.backward()
            # finally update parameters
            optimizer.step()


            print(input.shape, prediction.shape, output.shape)

    model.eval()
    for input, output in xfoil_testData:
        # run forward pass
        prediction = model(input)
        # compute loss function
        loss = torch.mean((prediction-output)**2)
        print("loss {}...".format(loss.detach().numpy()))

    pdb.set_trace()
    print("yolo")
