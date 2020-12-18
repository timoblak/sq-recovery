import sys
import torch
import numpy as np
import torch.optim as optim
import torch.utils.data as data
from torchsummary import summary
from helpers import parse_csv, save_model, load_model, save_compare_images, compare_images
from classes import H5Dataset, ExplicitLoss, ImplicitLoss, IoUAccuracy, LeastSquares
from models import ResNetSQ


# ----- CUDA
device_name = "cuda:0" #if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print("Using device: " + device_name)

# ----- Datasets
dataset_location = "../data/data/"
labels = parse_csv("../data/annotations/data_labels.csv")

# ----- Generators
# !!!!!! Don't forget to switch dataset mode when going from training to validation !!!!!!
# mode 0 for train and mode 1 for validation
dataset = H5Dataset(dataset_location, labels, train_split=0.9, dataset_file="dataset.h5")
training_generator = data.DataLoader(dataset, **{
    'batch_size': 32,
    'shuffle': False,
    'num_workers': 4
})

validation_generator = data.DataLoader(dataset, **{
    'batch_size': 32,
    'shuffle': False,
    'num_workers': 4
})

# ----- Hyperparameter initialization
MODEL_LOCATION = "trained_models/model_full.pt"
MAX_EPOCHS = 20000
LEARNING_RATE = 1e-4
LOG_INTERVAL = 1
RUNNING_MEAN = 100
DEBUG = False
PRETRAIN_EPOCHS = 0
CONTINUE_TRAINING = False

# ----- Net initialization
net = ResNetSQ(outputs=4, pretrained=True).to(device)
summary(net, input_size=(1, 256, 256))

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=25)


starting_epoch = 0
if CONTINUE_TRAINING:
    print("Continuing with training...")
    starting_epoch, net, optimizer, _ = load_model(MODEL_LOCATION, net, optimizer)

# ----- Training config

#loss_criterion = ExplicitLoss(32, device)
#loss_criterion = LeastSquares(64, device)
loss_criterion = ImplicitLoss(64, device, 1.5, 260)

accuracy_estimator = IoUAccuracy(render_size=64, device=device, full=True)


# ----- Main loop
best_val_loss = None
mean_losses, mean_val_losses, mean_val_accs = [], [], []
for epoch in range(starting_epoch, MAX_EPOCHS):
    losses, val_losses = [], []
    val_accuracies = []

    # Set training and data modes
    net.train()
    dataset.set_mode(0)

    for batch_idx, (data, true_labels) in enumerate(training_generator):
        
        # Transfer to GPU
        data, true_labels = data.to(device), true_labels.to(device)
        
        optimizer.zero_grad()

        # Predict
        pred_size, pred_shape, pred_position, pred_quat = net(data)
        pred_labels = torch.cat([pred_size, pred_shape, pred_position, pred_quat], dim=1)

        # Calculate loss and backpropagate
        loss = loss_criterion(data, pred_labels)
        loss.backward()

        #trues = true_labels.cpu().detach().numpy()
        #preds = pred_labels.cpu().detach().numpy()
        #compare_images(trues, preds, 1)

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
            print(true_labels)
            print("---------------------LOSS------------------------")
            print(np_loss)

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
    print('Train Epoch: {} Step: {} [(100%)]\tLoss: {:.6f}'.format(epoch, batch_idx, np.mean(losses)))

    mean_losses.append(np.mean(losses))

    # Validation mode
    net.eval()
    dataset.set_mode(1)

    with torch.no_grad():
        for batch_idx, (data, true_labels) in enumerate(validation_generator):

            data, true_labels = data.to(device), true_labels.to(device)

            # Predict
            pred_size, pred_shape, pred_position, pred_quat = net(data)
            pred_labels = torch.cat([pred_size, pred_shape, pred_position, pred_quat], dim=1) 
  
            # Calculate loss and backpropagate
            loss = loss_criterion(data, pred_labels)
            acc = accuracy_estimator(true_labels, pred_labels)
            
            if batch_idx == 0:
                trues = true_labels.cpu().detach().numpy()
                preds = pred_labels.cpu().detach().numpy()
                save_compare_images(trues, preds)

            val_losses.append(loss.item())
            val_accuracies.append(acc.item())

    val_loss_mean = np.mean(val_losses)
    val_accuracy_mean = np.mean(val_accuracies)

    mean_val_accs.append(val_accuracies)
    mean_val_losses.append(val_loss_mean)

    scheduler.step(val_loss_mean)

    if best_val_loss is None:
        print("Saving first model..")
        best_val_loss = np.mean(val_losses)
        save_model(MODEL_LOCATION, epoch, net, optimizer, {"loss": mean_losses, "val_loss": mean_val_losses, "val_acc": mean_val_accs})
    elif val_loss_mean < best_val_loss:
        print("New best loss achieved. Saving model..")
        best_val_loss = val_loss_mean
        save_model(MODEL_LOCATION, epoch, net, optimizer, {"loss": mean_losses, "val_loss": mean_val_losses, "val_acc": mean_val_accs})

    print("------------------------------------------------------------------------")
    print('Validation Epoch: {} Step: {} [(100%)]\tLoss: {:,.6f}\tAccuracy: {:,.6f}'.format(epoch, batch_idx, val_loss_mean, val_accuracy_mean))
    print("========================================================================")
