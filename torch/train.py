import sys
import math
import torch
import numpy as np
from time import sleep, time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchsummary import summary
from helpers import parse_csv, change_lr, save_model, load_model
from classes import H5Dataset, SQNet, ChamferLoss, QuaternionLoss


# ----- CUDA
device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print("Using device: " + device_name)
#cudnn.benchmark = True

# ----- Parameters
dataset_location = "/media/panter/0EE434EAE434D625/Users/Panter/Documents/Tim/SUPERBLOCKS/dataset_rot.h5"
dataset_location_val = "/media/panter/0EE434EAE434D625/Users/Panter/Documents/Tim/SUPERBLOCKS/dataset_rot_val.h5"
MODEL_LOCATION = "models/model.pt"
GENERATOR_PARAMS = {
    'batch_size': 32,
    'shuffle': False,
    'num_workers': 4
}
MAX_EPOCHS = 1000
LEARNING_RATE = 1e-5
LOG_INTERVAL = 10
RUNNING_MEAN = 30
DEBUG = False
PRETRAIN_EPOCHS = 35
CONTINUE_TRAINING = True

# ----- Datasets
labels = parse_csv("../data/annotations/data_rot2.csv")
labels_val = parse_csv("../data/annotations/data_rot_val2.csv")
print("Dataset split to: train - " + str(len(labels)) + ", val - " + str(len(labels_val)))

# ----- Generators
training_set = H5Dataset(dataset_location, labels)
training_generator = data.DataLoader(training_set, **GENERATOR_PARAMS)

validation_set = H5Dataset(dataset_location_val, labels_val)
validation_generator = data.DataLoader(validation_set, **GENERATOR_PARAMS)

# ----- Net initialization
net = SQNet(outputs=12, clip_values=False).to(device)
summary(net, input_size=(1, 256, 256))
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
starting_epoch = 0
if CONTINUE_TRAINING:
    print("Continuing with training...")
    starting_epoch, net, optimizer, _ = load_model(MODEL_LOCATION, net, optimizer)

# ----- Training config
loss_mse = nn.MSELoss(reduction="mean")
loss_chamfer = ChamferLoss(16, device)
loss_quat = QuaternionLoss()

# ----- Main loop
best_val_loss = None
for epoch in range(starting_epoch, MAX_EPOCHS):
    # ----- Data
    losses, val_losses = [], []
    l1, l2 = [], []
    net.train()
    # Training

    for batch_idx, (data, true_labels) in enumerate(training_generator):
        # Transfer to GPU
        data, true_labels = data.to(device), true_labels.to(device)
        optimizer.zero_grad()

        # Run forward pass
        #t_start = time()
        pred_labels = net(data)
        #print("Net predict: ", time() - t_start)

        if epoch < PRETRAIN_EPOCHS:
            pred_block, pred_quat = pred_labels
            true_block, true_quat = torch.split(true_labels, (8, 4), dim=-1)

            loss1 = loss_mse(pred_block, true_block)
            loss2 = loss_quat(pred_quat, true_quat)

            l1.append(loss1.item())
            l2.append(loss2.item())
            loss = loss1 + loss2
        else:
            change_lr(optimizer, 1e-8)
            loss = loss_chamfer(torch.cat(pred_labels, dim=-1), true_labels)

        loss.backward()
        if torch.any(torch.isnan(net.fc1.weight.grad)):
            DEBUG = True

        # Update weights and reset accumulative grads
        optimizer.step()
        np_loss = loss.item()
        losses.append(np_loss)

        if DEBUG:
            print("================================================")
            print(batch_idx)
            print("---------------------NEW------------------------")
            print(pred_labels)
            print(true_labels)
            exit()

        if batch_idx % LOG_INTERVAL == 0 and not DEBUG:
            sys.stdout.write("\033[K")
            print('Train Epoch: {} Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (Block: {:.6f}, Quat: {:.6f})'.format(
                epoch, batch_idx, batch_idx * len(data), len(training_generator.dataset),
                       100. * batch_idx / len(training_generator), np.mean(losses[-RUNNING_MEAN:]), np.mean(l1[-RUNNING_MEAN:]), np.mean(l2[-RUNNING_MEAN:])), end="\r")

    # Print last
    sys.stdout.write("\033[K")
    print('Train Epoch: {} Step: {} [(100%)]\tLoss: {:.6f}'.format(epoch, batch_idx, np.mean(losses)))

    # Validation
    net.eval()
    l1, l2 = [], []
    with torch.no_grad():
        for batch_idx, (data, true_labels) in enumerate(validation_generator):

            data, true_labels = data.to(device), true_labels.to(device)
            pred_labels = net(data)

            if epoch < PRETRAIN_EPOCHS:
                pred_block, pred_quat = pred_labels
                true_block, true_quat = torch.split(true_labels, (8, 4), dim=-1)

                loss1 = loss_mse(pred_block, true_block)
                loss2 = loss_quat(pred_quat, true_quat)

                l1.append(loss1.item())
                l2.append(loss2.item())
                loss = loss1 + loss2
            else:
                loss = loss_chamfer(torch.cat(pred_labels, dim=-1), true_labels)

            np_loss = loss.item()
            val_losses.append(np_loss)

    val_loss_mean = np.mean(val_losses)
    if best_val_loss is None:
        best_val_loss = np.mean(val_losses)
        save_model(MODEL_LOCATION, epoch, net, optimizer, {"loss": losses, "val_loss": val_losses})
    elif val_loss_mean < best_val_loss:
        print("New best loss achieved. Saving model..")
        best_val_loss = val_loss_mean
        save_model(MODEL_LOCATION, epoch, net, optimizer, {"loss": losses, "val_loss": val_losses})
    print("------------------------------------------------------------------------")
    print("VAL PREDICTIONS: ")
    print("- PRED: ", pred_labels)
    print("- TRUE: ", torch.split(true_labels, (8, 4), dim=-1))
    print("------------------------------------------------------------------------")
    print('Validation Epoch: {} Step: {} [(100%)]\tLoss: {:.6f} (Block: {:.6f}, Quat: {:.6f})'.format(epoch, batch_idx, val_loss_mean, np.mean(l1), np.mean(l2)))



#training_set.close()
#validation_set.close()
