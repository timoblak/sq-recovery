import sys
import math
import torch
import numpy as np
from time import sleep, time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchsummary import summary
from helpers import parse_csv
from classes import H5Dataset, SQNet, ChamferLoss, QuaternionLoss


# ----- CUDA
device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print("Using device: " + device_name)
#cudnn.benchmark = True

# ----- Parameters
dataset_location = "/media/panter/0EE434EAE434D625/Users/Panter/Documents/Tim/SUPERBLOCKS/dataset_rot.h5"
dataset_location_val = "/media/panter/0EE434EAE434D625/Users/Panter/Documents/Tim/SUPERBLOCKS/dataset_rot_val.h5"
generator_params = {'batch_size': 32,
                    'shuffle': False,
                    'num_workers': 4}
max_epochs = 1000
lr = 1e-5
log_interval = 10
loss_mode = "mse"
debug = False
pretrain_epochs = 35

# ----- Datasets
labels = parse_csv("../data/annotations/data_rot2.csv")
labels_val = parse_csv("../data/annotations/data_rot_val2.csv")
print("Dataset split to: train - " + str(len(labels)) + ", val - " + str(len(labels_val)))

# ----- Generators
training_set = H5Dataset(dataset_location, labels)
training_generator = data.DataLoader(training_set, **generator_params)

validation_set = H5Dataset(dataset_location_val, labels_val)
validation_generator = data.DataLoader(validation_set, **generator_params)

# ----- Net initialization
net = SQNet(outputs=12, clip_values=False).to(device)
summary(net, input_size=(1, 256, 256))

# ----- Training config
optimizer = optim.Adam(net.parameters(), lr=lr)
loss_mse = nn.MSELoss(reduction="mean")
loss_chamfer = ChamferLoss(16, device)
loss_quat = QuaternionLoss()

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
        #t_start = time()
        pred_labels = net(data)
        #print("Net predict: ", time() - t_start)

        if epoch < pretrain_epochs:
            pred_block, pred_quat = pred_labels
            true_block, true_quat = torch.split(true_labels, (8, 4), dim=-1)

            loss1 = loss_mse(pred_block, true_block)
            loss2 = loss_quat(pred_quat, true_quat)

            loss = loss1 + loss2
        else:
            loss = loss_chamfer(torch.cat(pred_labels, dim=-1), true_labels)
            for g in optimizer.param_groups:
                g['lr'] = 1e-8

        loss.backward()
        if torch.any(torch.isnan(net.fc1.weight.grad)):
            debug = True

        # Update weights and reset accumulative grads
        optimizer.step()
        np_loss = loss.item()
        losses.append(np_loss)

        if debug:
            print("================================================")
            print(batch_idx)
            print("---------------------NEW------------------------")
            print(pred_labels)
            print(true_labels)
            exit()

        if batch_idx % log_interval == 0 and not debug:
            l1, l2 = 0, 0
            if epoch < pretrain_epochs:
                l1, l2 = loss1.item(), loss2.item()

            sys.stdout.write("\033[K")
            print('Train Epoch: {} Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (Block: {:.6f}, Quat: {:.6f})'.format(
                epoch, batch_idx, batch_idx * len(data), len(training_generator.dataset),
                       100. * batch_idx / len(training_generator), np.mean(losses[-10:]), l1, l2), end="\r")

    # Print last
    sys.stdout.write("\033[K")
    print('Train Epoch: {} Step: {} [(100%)]\tLoss: {:.6f}'.format(epoch, batch_idx, np.mean(losses)))

    net.train(True)
    # Validation
    with torch.no_grad():
        for batch_idx, (data, true_labels) in enumerate(validation_generator):

            data, true_labels = data.to(device), true_labels.to(device)

            pred_labels = net(data)

            if epoch < pretrain_epochs:
                pred_block, pred_quat = pred_labels
                true_block, true_quat = torch.split(true_labels, (8, 4), dim=-1)

                loss1 = loss_mse(pred_block, true_block)
                loss2 = loss_quat(pred_quat, true_quat)

                loss = loss1 + loss2
            else:
                loss = loss_chamfer(torch.cat(pred_labels, dim=-1), true_labels)

            np_loss = loss.item()
            val_losses.append(np_loss)
    print("------------------------------------------------------------------------")
    print("VAL PREDICTIONS: ")
    print("- PRED: ", pred_labels)
    print("- TRUE: ", torch.split(true_labels, (8, 4), dim=-1))
    print("------------------------------------------------------------------------")
    # Print last
    l1, l2 = 0, 0
    if epoch < pretrain_epochs:
        l1, l2 = loss1.item(), loss2.item()
    print('Validation Epoch: {} Step: {} [(100%)]\tLoss: {:.6f} (Block: {:.6f}, Quat: {:.6f})'.format(epoch, batch_idx, np.mean(val_losses), l1, l2))

#training_set.close()
#validation_set.close()
