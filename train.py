import random
import argparse

import numpy as np

from torchsummary import summary

import torch
import torch.nn as nn
import torch.optim as optim

import torchmetrics


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
parser = argparse.ArgumentParser(description="Hyperbolic Embeddings for Content-Based Image Retrieval.")
parser.add_argument('--hyperbolic', action='store_true', help='Define an hyperbolic or euclidean model. Default is Euclidean.')
parser.add_argument('--weights_path', help='Path to save the model')
parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100'], help='Define which dataset to use. Currently only CIFAR10 and CIFAR100 are available.')
parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training. Default is 64)')
parser.add_argument('--validation_size', type=float, default=0.2, help='Percentage of training set to use as validation. Must be a value between 0 and 1')
parser.add_argument('--network', choices=['VGG16'], default='VGG16', help='Define the network architecture to use for embedding extraction. Currently only VGG16 is supported')
parser.add_argument("--curvature", type=float, default=1.0, help="Curvature of the Poincare ball. Default is 1.0")
parser.add_argument("--dimension", type=int, default=2, help="Dimension of the Poincare ball. Default is 2")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate. Default is 0.001)")
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train. Default is 200)")
parser.add_argument("--patience", type=int, default=15, help="Number of epochs to wait before early stopping the training. Default is 15")
parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
parser.add_argument('--verbose', action='store_false', help='Increase output verbosity. Default is True')

args = parser.parse_args()

# Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

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
if args.dataset == 'CIFAR10':
    from datasets.CIFAR10 import generate_CIFAR10 as generate_dataset
    num_classes=10
elif args.dataset == 'CIFAR100':
    from datasets.CIFAR100 import generate_CIFAR100 as generate_dataset
    num_classes=100
else:
    print('[ERROR] Currently only CIFAR10 and CIFAR100 datasets are supported. Exiting...')
    exit(1)

# Get dataset loaders and classes
train_loader, validation_loader, test_loader, classes = generate_dataset(args.batch_size, args.validation_size, NUM_WORKERS, PIN_MEMORY)


# -------------------------------------
# -------------- Network --------------
# -------------------------------------
if args.network == 'VGG16':
    from networks.vgg16 import create_vgg16_model as network
else:
    print('[ERROR] Currently only VGG16 network is supported. Exiting...')

# Generate Model
model = network(args.curvature, args.dimension, args.hyperbolic, False, num_classes=num_classes)
model = model.to(device)
    
if args.verbose:
    print('Hyperbolic Curvature: ', args.curvature)
    summary(model, (3, 32, 32))

# Specify loss function, optimizer and LR scheduler
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=args.verbose, patience=10)


# --------------------------------------
# ------------ Training Loop -----------
# --------------------------------------
# Initialize training variables
train_losses = []
best_validation_loss = np.Inf
early_stopper = 0

for epoch in range(1, args.epochs + 1):
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
        torch.save(model.state_dict(), args.weights_path)
        best_validation_loss = running_validation_loss
        early_stopper = 0
    else:
        # Update early stopping values
        early_stopper = early_stopper + 1
    
    # Early stopping
    if early_stopper == args.patience:
        print('Validation loss hasnt decreased from ({:.6f}) for {} epochs. Early stopping in epoch {}...'.format(best_validation_loss, args.patience, epoch))
        break