import sys
import math
import torch
import numpy as np
from time import sleep, time
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchsummary import summary
from helpers import parse_csv, change_lr, save_model, load_model, save_compare_images, slerp
from classes import H5Dataset, ChamferLoss, QuaternionLoss, RotLoss, ChamferQuatLoss, IoUAccuracy
from models import GenericNetSQ, ResNetSQ
import cv2

# ----- CUDA
device_name = "cuda:0" #if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print("Using device: " + device_name)

# ----- Datasets
dataset_location = "../data/data/"
labels = parse_csv("../data/annotations/data_labels.csv")

# ----- Generators
# !!!!!! Don't forget to switch dataset mode when going from training to validation !!!!!!
dataset = H5Dataset(dataset_location, labels, train_split=0.9, dataset_file="dataset.h5")
training_generator = data.DataLoader(dataset, **{
    'batch_size': 32,
    'shuffle': True,
    'num_workers': 4
})

validation_generator = data.DataLoader(dataset, **{
    'batch_size': 32,
    'shuffle': False,
    'num_workers': 4
})

# ----- Hyperparameter initialization
MODEL_LOCATION = "trained_models/model_e4_32_32_2.pt"
MAX_EPOCHS = 20000
LEARNING_RATE = 1e-4
LOG_INTERVAL = 1
RUNNING_MEAN = 100
DEBUG = False
PRETRAIN_EPOCHS = 0
CONTINUE_TRAINING = False

# ----- Net initialization
net = ResNetSQ(outputs=4, pretrained=True).to(device)
#net = SQNet(outputs=4, fcn=64, dropout=0, clip_values=False).to(device)
summary(net, input_size=(1, 256, 256))
#exit()

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=0)

starting_epoch = 0
if CONTINUE_TRAINING:
    print("Continuing with training...")
    starting_epoch, net, optimizer, _ = load_model(MODEL_LOCATION, net, optimizer)

# ----- Training config
#loss_mse = nn.MSELoss(reduction="mean")
#loss_chamfer = ChamferLoss(17, device)
#loss_quat = QuaternionLoss()
#loss_rot = RotLoss(32, device)
loss_chamfer_quat = ChamferQuatLoss(32, device)
accuracy_estimator = IoUAccuracy(render_size=64, device=device)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4, verbose=True, threshold=1e-2)


g_clip = 0.5
#for p in net.parameters():
#    p.register_hook(lambda grad: torch.clamp(grad, -g_clip, g_clip))

# ----- Main loop
best_val_loss = None
changed = False
for epoch in range(starting_epoch, MAX_EPOCHS):
    # ----- Data
    losses, val_losses = [], []
    val_accuracies = []
    l1, l2 = [0], [0]

    net.train()
    dataset.set_mode(0)
    # Training

    for batch_idx, (data, true_labels) in enumerate(training_generator):
        
        # Transfer to GPU
        data, true_labels = data.to(device), true_labels.to(device)
        
        optimizer.zero_grad()

        # Predict
        pred_labels = net(data)

        # Calculate loss and backpropagate
        loss = loss_chamfer_quat(true_labels, pred_labels)
        
        loss.backward()

        # Update weights
        optimizer.step()

        # Save loss
        np_loss = loss.item()
        losses.append(np_loss)

        if DEBUG:
            print("================================================")
            print(batch_idx)
            print("---------------------LABELS (pred - true)------------------------")
            print(pred_labels)
            print(true_labels[:, 8:])
            print(true_labels[:, :8])
            print("---------------------LOSS------------------------")
            print(np_loss)
            sleep(0.5)

        if torch.any(torch.isnan(net.encoder.fc[0].weight.grad)):
            print("--------------- NAN GRADS!!!! ---------------")

        if batch_idx % LOG_INTERVAL == 0 and not DEBUG:
            sys.stdout.write("\033[K")
            print('Train Epoch: {} Step: {} [{}/{} ({:.0f}%)]\tLoss: {:,.6f}'.format(
                epoch, batch_idx, batch_idx * len(data), len(training_generator.dataset),
                       100. * batch_idx / len(training_generator), np.mean(losses[-RUNNING_MEAN:])), end="\r")
        
    # Print last
    sys.stdout.write("\033[K")
    print("------------------------------------------------------------------------")
    print("TRAIN PREDICTIONS: ")
    print("- PRED: ", pred_labels)
    print("- TRUE: ", true_labels[:, 8])
    print('Train Epoch: {} Step: {} [(100%)]\tLoss: {:.6f}'.format(epoch, batch_idx, np.mean(losses)))

    # Validation
    net.eval()
    dataset.set_mode(1)

    l1, l2 = [0], [0]
    with torch.no_grad():
        for batch_idx, (data, true_labels) in enumerate(validation_generator):

            data, true_labels = data.to(device), true_labels.to(device)
            pred_labels = net(data)

            loss = loss_chamfer_quat(true_labels, pred_labels)
            acc = accuracy_estimator(true_labels, pred_labels)


            if batch_idx == 0:
                trues = true_labels.cpu().detach().numpy()
                preds = pred_labels.cpu().detach().numpy()
                save_compare_images(trues, np.concatenate((trues[:, :8], preds), axis=-1))

            val_losses.append(loss.item())
            val_accuracies.append(acc.item())

    val_loss_mean = np.mean(val_losses)
    val_accuracy_mean = np.mean(val_accuracies)
    #scheduler.step(val_loss_mean)
    if best_val_loss is None:
        print("Saving first model..")
        best_val_loss = np.mean(val_losses)
        save_model(MODEL_LOCATION, epoch, net, optimizer, {"loss": losses, "val_loss": val_losses})
    elif val_loss_mean < best_val_loss:
        print("New best loss achieved. Saving model..")
        best_val_loss = val_loss_mean
        save_model(MODEL_LOCATION, epoch, net, optimizer, {"loss": losses, "val_loss": val_losses})
    print("------------------------------------------------------------------------")
    print("VAL PREDICTIONS: ")
    print(pred_labels)
    #print("- TRUE: ", true_labels[:, 8:])
    # print("- TRUE: ", torch.split(true_labels, (8, 4), dim=-1))
    print("------------------------------------------------------------------------")
    print('Validation Epoch: {} Step: {} [(100%)]\tLoss: {:,.6f}\tAccuracy: {:,.6f}'.format(epoch, batch_idx, val_loss_mean, val_accuracy_mean))
    print("========================================================================")



#training_set.close()
#validation_set.close()
