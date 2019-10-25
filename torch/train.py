import torch
from tqdm import tqdm
import torch.utils.data as data
from classes import Dataset, SQNet, ChamferLoss
from helpers import parse_csv, graph2img
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
import sys
import numpy as np

# ----- CUDA
device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print("Using device: " + device_name)
#cudnn.benchmark = True

# ----- Parameters
dataset_location = "../data/data_iso/"
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 4}
max_epochs = 100
db_split = 0.95
lr = 0.01
log_interval = 10
loss_mode = "mse" # "chamfer"

# ----- Datasets
list_ids, labels = parse_csv("../data/annotations/data_iso.csv")
print("Dataset split to: train - " + str(round(db_split * len(list_ids))) + ", val - " + str(len(list_ids) - round(db_split * len(list_ids))))
partition = {'train': list_ids[:round(db_split * len(list_ids))],
             'validation': list_ids[round(db_split * len(list_ids)):]}

# ----- Generators
training_set = Dataset(dataset_location, partition['train'], labels)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(dataset_location, partition['validation'], labels)
validation_generator = data.DataLoader(validation_set, **params)

# ----- Net initialization
net = SQNet().to(device)
summary(net, input_size=(1, 256, 256))
#graph2img(net)

# ----- Training config
optimizer = optim.Adam(net.parameters(), lr=lr)
#loss_fn = nn.MSELoss(reduction="mean")
loss_fn = ChamferLoss(64, device)


# ----- Main loop
for epoch in range(max_epochs):
    # ----- Data
    losses, val_losses = [], []

    # Training
    for batch_idx, (data, true_labels) in enumerate(training_generator):
        net.train()
        optimizer.zero_grad()

        # Transfer to GPU
        data, true_labels = data.to(device), true_labels.to(device)

        # Run forward pass
        pred_labels = net(data)

        # Calculate loss and backpropagate
        loss = loss_fn(pred_labels, true_labels)
        loss.backward()

        # Update weights and reset accumulative grads
        optimizer.step()
        np_loss = loss.item()
        print(np_loss, pred_labels, true_labels)
        losses.append(np_loss)

        if batch_idx % log_interval == 0:
            sys.stdout.write("\033[K")
            print('Train Epoch: {} Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, batch_idx * len(data), len(training_generator.dataset),
                       100. * batch_idx / len(training_generator), np.mean(losses[-10:])), end="\r")
    # Print last
    sys.stdout.write("\033[K")
    print('Train Epoch: {} Step: {} [(100%)]\tLoss: {:.6f}'.format(epoch, batch_idx, np.mean(losses)))

    # Validation
    with torch.no_grad():
        for batch_idx, (data, true_labels) in enumerate(validation_generator):
            net.eval()

            # Transfer to GPU
            data, true_labels = data.to(device), true_labels.to(device)

            # Model computations
            # Run forward pass
            pred_labels = net(data)

            # Calculate loss and backpropagate
            loss = loss_fn(pred_labels, true_labels)
            np_loss = loss.item()

            val_losses.append(np_loss)

    # Print last
    print('Validation Epoch: {} Step: {} [(100%)]\tLoss: {:.6f}'.format(epoch, batch_idx, np.mean(val_losses)))