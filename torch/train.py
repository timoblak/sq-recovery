import torch
from tqdm import tqdm
import torch.utils.data as data
from classes import H5Dataset, SQNet, ChamferLoss
from helpers import parse_csv, graph2img, plot_grad_flow, getBack
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
import sys
import cv2
import numpy as np
from time import sleep
import math

# ----- CUDA
device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print("Using device: " + device_name)
#cudnn.benchmark = True

# ----- Parameters
dataset_location = "../data/data_iso2/"
dataset_location_val = "../data/data_iso_val2/"
generator_params = {'batch_size': 32,
                    'shuffle': False,
                    'num_workers': 0}
max_epochs = 100
db_split = 0.95
lr = 1e-4
log_interval = 10
loss_mode = "mse" # "chamfer"
debug = False


# ----- Datasets
labels = parse_csv("../data/annotations/data_iso2.csv")
labels_val = parse_csv("../data/annotations/data_iso_val2.csv")
print("Dataset split to: train - " + str(len(labels)) + ", val - " + str(len(labels_val)))

# ----- Generators
training_set = H5Dataset(dataset_location, labels)
training_generator = data.DataLoader(training_set, **generator_params)

validation_set = H5Dataset(dataset_location_val, labels_val)
validation_generator = data.DataLoader(validation_set, **generator_params)

# ----- Net initialization
net = SQNet(outputs=6, clip_values=False).to(device)
summary(net, input_size=(1, 256, 256))

# ----- Training config
optimizer = optim.Adam(net.parameters(), lr=lr)
loss_fn = nn.MSELoss(reduction="mean")
loss_fn2 = ChamferLoss(32, device)


# ----- Main loop
for epoch in range(max_epochs):
    # ----- Data
    losses, val_losses = [], []
    net.train(True)
    # Training

    for batch_idx, (data, true_labels) in enumerate(training_generator):
        # Transfer to GPU
        data, true_labels = data.to(device), true_labels.to(device)
        optimizer.zero_grad()

        # Run forward pass
        pred_labels = net(data)

        # Calculate loss and backpropagate
        loss = loss_fn(pred_labels, true_labels)

        if epoch < 1:
            loss = loss_fn(pred_labels, true_labels)
        else:
            #debug = True
            loss = loss_fn2(pred_labels, true_labels)

            for g in optimizer.param_groups:
                g['lr'] = 1e-6
            #torch.nn.utils.clip_grad_norm_(net.parameters(), 1)

        #with torch.autograd.detect_anomaly():
        loss.backward()

        if debug:
            total_norm = 0
            #for p in net.parameters():
            #    param_norm = p.grad.data.norm(2)
            #    total_norm += param_norm.item() ** 2
            #total_norm = total_norm ** (1. / 2)
            print("================================================")
            print(batch_idx)
            print("---------------------NEW------------------------")
            print(pred_labels)
            print(true_labels)
            print("             --------GRADS NORM--------                ")
            #print(net.fc1.weight.grad)
            print(total_norm)
            print("             -------WEIGHTS BEFORE---------                ")
            print(net.fc1.weight)

        # Update weights and reset accumulative grads
        optimizer.step()
        np_loss = loss.item()
        losses.append(np_loss)

        if debug:
            print("             -------WEIGHTS AFTER---------                ")
            print(net.fc1.weight)
            print("\n\n")
            print("             -------LOSS---------                ")
            print(np.mean(losses[-10:]))
            sleep(0.5)

        if math.isnan(np.mean(losses[-10:])) or np.mean(losses[-10:]) == np.inf:
            #getBack(loss.grad_fn)
            debug =  True
            sleep(10)

        if batch_idx % log_interval == 0 and not debug:
            sys.stdout.write("\033[K")
            print('Train Epoch: {} Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, batch_idx * len(data), len(training_generator.dataset),
                       100. * batch_idx / len(training_generator), np.mean(losses[-10:])), end="\r")

    print("------------------------------------------------------------------------")
    print("TRAIN PREDICTIONS: ")
    print("- PRED: ", pred_labels)
    print("- TRUE: ", true_labels)
    print("------------------------------------------------------------------------")
    # Print last
    sys.stdout.write("\033[K")
    print('Train Epoch: {} Step: {} [(100%)]\tLoss: {:.6f}'.format(epoch, batch_idx, np.mean(losses)))

    net.train(True)
    # Validation
    with torch.no_grad():
        for batch_idx, (data, true_labels) in enumerate(validation_generator):

            data, true_labels = data.to(device), true_labels.to(device)


            pred_labels = net(data)
            if epoch < 1:
                loss = loss_fn(pred_labels, true_labels)
            else:
                loss = loss_fn2(pred_labels, true_labels)

            np_loss = loss.item()
            val_losses.append(np_loss)
    print("------------------------------------------------------------------------")
    print("VAL PREDICTIONS: ")
    print("- PRED: ", pred_labels)
    print("- TRUE: ", true_labels)
    print("------------------------------------------------------------------------")
    # Print last
    print('Validation Epoch: {} Step: {} [(100%)]\tLoss: {:.6f}'.format(epoch, batch_idx, np.mean(val_losses)))

training_set.close()
validation_set.close()
