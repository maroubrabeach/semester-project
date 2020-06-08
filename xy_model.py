import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils
from os import listdir
from os.path import isfile, join
import pdb
import time
import matplotlib.pyplot as plt

DATA_PATH = "./xfoil/data"

# Regress C_L, C_D from x, y coordinates

dir = listdir(DATA_PATH)
trainFrac = 0.9
trainSize = int(round(trainFrac*len(dir), 0))

# Number of epochs to train for
num_epochs = 10


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
        # input_high_res = np.vstack((dataTrain.item()["x"], dataTrain.item()["y"]))
        # input_low_res = input_high_res[:,::4] # resize to 40 points
        # input = torch.reshape(torch.tensor(input_low_res).float(), (1,-1))
        input = torch.cat((torch.tensor(dataTrain.item()["x"]).float(), torch.tensor(dataTrain.item()["y"]).float()))
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
        # input_high_res = np.vstack((dataTest.item()["x"], dataTest.item()["y"]))
        # input_low_res = input_high_res[:,::4] # resize to 40 points
        # input = torch.reshape(torch.tensor(input_low_res).float(), (1,-1))
        input = torch.cat((torch.tensor(dataTest.item()["x"]).float(), torch.tensor(dataTest.item()["y"]).float()))
        output = torch.cat((torch.tensor(dataTest.item()["cl"]).float(), torch.tensor(dataTest.item()["cd"]).float()))
        return input, output


# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#final-thoughts-and-where-to-go-next
def train_model(model, dataloaders, optimizer, num_epochs):
    since = time.time()

    test_loss_history = []
    train_loss_history = []

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data
            for input, output in dataloaders[phase]:
                input = input.to(device)
                output = output.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    prediction = model(input)
                    loss = torch.mean((prediction-output)**2)

                    # _, preds = torch.max(prediction, 1)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * input.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'train':
                train_loss_history.append(epoch_loss)
            else:
                test_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss_history, test_loss_history



if __name__ == "__main__":

    # initialize data
    xfoil_trainData = XfoilTrainDataset()
    xfoil_trainloader = data_utils.DataLoader(
        xfoil_trainData,
        batch_size=2,
        shuffle=False,
        # num_workers=1,
    )
    xfoil_testData = XfoilTestDataset()
    xfoiltestloader = data_utils.DataLoader(
        xfoil_testData,
        batch_size=2,
        shuffle=False,
        # num_workers=1,
    )
    dataloaders_dict = {
    "train": xfoil_trainloader,
    "test": xfoiltestloader
    }

    # initialize network/model
    model = nn.Sequential(
          nn.Linear(2*160, 10),
          nn.ReLU(),
          nn.Linear(10, 2),
        )
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    print("Parameters to learn:")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    # what are the parameters of the model we want to optimize
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    # Train and evaluate
    train_loss_history, test_loss_history = train_model(model, dataloaders_dict, optimizer, num_epochs=num_epochs)

    plt.title("Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1, num_epochs+1), train_loss_history, label="Train loss")
    plt.plot(range(1, num_epochs+1), test_loss_history, label="Test loss")
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.show()


    pdb.set_trace()
