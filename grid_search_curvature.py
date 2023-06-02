import random

import numpy as np

from torchsummary import summary

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import torchmetrics

from networks.vgg16 import create_vgg16_model as network


def train_one_epoch(model, criterion, optimizer, train_loader, num_classes, device):
    training_loss = 0.0
    predicted_classes = []
    target_classes = []

    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()               # Zero gradients for batch
        output = model(data)                # Make predictions for batch
        loss = criterion(output, target)    # Compute loss for batch
        loss.backward()                     # Compute gradients for optimization
        optimizer.step()                    # Adjust learning weights

        training_loss += loss.item()*data.size(0)

        # Add predicted classes for accuracy calculation
        _, pred = torch.max(output, 1)
        predicted_classes.append(pred.cpu().numpy())
        target_classes.append(target.cpu().numpy())
    
    predicted_classes = torch.from_numpy(np.concatenate(predicted_classes))
    target_classes = torch.from_numpy(np.concatenate(target_classes))

    # Calculate the accuracy
    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
    training_accuracy = metric(predicted_classes, target_classes)
    
    return training_loss / len(train_loader.dataset), training_accuracy


def validate_one_epoch(model, criterion, validation_loader, num_classes, device):
    validation_loss = 0.0
    predicted_classes = []
    target_classes = []

    model.eval()
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            validation_loss += loss.item()*data.size(0)

            # Add predicted classes for accuracy calculation
            _, pred = torch.max(output, 1)
            predicted_classes.append(pred.cpu().numpy())
            target_classes.append(target.cpu().numpy())
    
    predicted_classes = torch.from_numpy(np.concatenate(predicted_classes))
    target_classes = torch.from_numpy(np.concatenate(target_classes))

    # Calculate the accuracy
    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
    validation_accuracy = metric(predicted_classes, target_classes)
    
    return validation_loss / len(validation_loader.dataset), validation_accuracy


# -------------------------------------
# ----------- Initial Setup -----------
# -------------------------------------
# Training settings
DATASET = 'CIFAR100'
SEED = 42
VERBOSE = True

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Check CUDA availability to train on GPU
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if device == 'cuda':
    print('[INFO] CUDA is available! Training on GPU...')
    NUM_WORKERS = 1
    PIN_MEMORY = True
else:
    print('[INFO] CUDA is not available. Training on CPU...')
    NUM_WORKERS = 4
    PIN_MEMORY = False


# -------------------------------------
# -------------- Dataset --------------
# -------------------------------------
if DATASET == 'CIFAR10':
    from datasets.CIFAR10 import generate_CIFAR10 as generate_dataset
    num_classes=10
elif DATASET == 'CIFAR100':
    from datasets.CIFAR100 import generate_CIFAR100 as generate_dataset
    num_classes=100
else:
    print('[ERROR] Currently only CIFAR10 and CIFAR100 datasets are supported. Exiting...')
    exit(1)

# Get dataset loaders and classes
train_loader, validation_loader, test_loader, classes = generate_dataset(50000, 0.0, NUM_WORKERS, PIN_MEMORY, augment=False)

for data, target in train_loader:
    X = data
    y = target


# Establish grid search parameters
# We will only search for optimal curvature
CURVATURES = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
N_FOLDS = 5

# Grid Search
split_length = int(len(X) / N_FOLDS)
for curvature in CURVATURES:
    print()
    print('Training for curvature: ', curvature)
    for i in range(N_FOLDS):
        print('Fold Number: ', i)
        print('-------------------')
        print()

        # Reload the model every fold
        model = network(curvature, 32, True, False, num_classes=num_classes)
        model = model.to(device)
        
        # Specify loss function, optimizer and LR scheduler
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=VERBOSE, patience=10)

        # Split data into folds
        X_validation = X[i*split_length : (i+1)*split_length]
        X_train = torch.from_numpy(np.concatenate([X[:i*split_length], X[(i+1)*split_length:]]))
        y_validation = y[i*split_length : (i+1)*split_length]
        y_train = torch.from_numpy(np.concatenate([y[:i*split_length], y[(i+1)*split_length:]]))

        # Transform folds into DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        validation_dataset = TensorDataset(X_validation, y_validation)
        train_loader = DataLoader(train_dataset, batch_size=64, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=64, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)

        # --------------------------------------
        # ------------ Training Loop -----------
        # --------------------------------------
        # Initialize training variables
        train_losses = []
        best_validation_loss = np.Inf
        early_stopper = 0

        for epoch in range(1, 150):
            # Train and Validate the model
            running_training_loss, running_training_accuracy = train_one_epoch(model, criterion, optimizer, train_loader, num_classes, device)
            running_validation_loss, running_validation_accuracy = validate_one_epoch(model, criterion, validation_loader, num_classes, device)
            
            # Update the LR Scheduler
            scheduler.step(running_validation_loss)

            # Print Training and Validation Statistics
            print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(epoch, running_training_loss, running_training_accuracy, running_validation_loss, running_validation_accuracy))

            # Save the model if loss has decreased
            if running_validation_loss <= best_validation_loss:
                print('Validation Loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(best_validation_loss, running_validation_loss))
                torch.save(model.state_dict(), './weights/curvature/' + DATASET + '_' + str(curvature) + '_' + str(i) + '.pth')
                best_validation_loss = running_validation_loss
                early_stopper = 0
            else:
                # Update early stopping values
                early_stopper = early_stopper + 1
            
            # Early stopping
            if early_stopper == 15:
                print('Validation loss hasnt decreased from ({:.6f}) for {} epochs. Early stopping in epoch {}...'.format(best_validation_loss, 15, epoch))
                print()
                break
