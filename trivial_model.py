import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils
import torch.nn as nn
from os import listdir
from os.path import isfile, join
import pdb
import logging

DATA_PATH = "./xfoil/data"


# REGRESS C_L from (x,y)
class XfoilDataset(Dataset):
    """xfoil dataset."""

    def __init__(self, data_path = DATA_PATH):
        """
        Args:
            data_path (string): Path to folder with profiles.
        """
        self.data = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f)) and f.endswith(".npy")][0:8]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = np.load(self.data[idx], allow_pickle=True)
        # input, outputs
        return torch.tensor(data.item()["x"]).float(), torch.tensor(data.item()["cl"]).float()


# https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb
class DumbRegressor(torch.nn.Module):
    def __init__(self):
        super(DumbRegressor, self).__init__()
        self.fc = nn.Linear(160,2)
    def forward(self, x):
        output = self.fc(x)
        return output


if __name__ == "__main__":

    # initialize data
    xfoil_data = XfoilDataset()
    xfoil_loader = data_utils.DataLoader(
        xfoil_data,
        batch_size=2,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    # initialize network/model
    model = DumbRegressor()

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
            # compute loss fuction
            loss = torch.mean((prediction-output)**2)
            print("loss {}...".format(loss.detach().numpy()))
            # compute backward pass
            loss.backward()
            # finally update parameters
            optimizer.step()


            print(input.shape, prediction.shape, output.shape)

    pdb.set_trace()
    print("yo")
