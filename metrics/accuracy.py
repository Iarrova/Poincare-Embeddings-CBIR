import numpy as np

from sklearn.metrics import classification_report

import torch

import torchmetrics

def measure_accuracy(model, criterion, test_loader, classes, device, num_classes=10):
    test_loss = 0.0
    predicted_classes = []
    target_classes = []

    # Get the predicted classes
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()*data.size(0)
            
            _, pred = torch.max(output, 1)
            predicted_classes.append(pred.cpu().numpy())
            target_classes.append(target.cpu().numpy())
    
    predicted_classes = torch.from_numpy(np.concatenate(predicted_classes))
    target_classes = torch.from_numpy(np.concatenate(target_classes))

    # Calculate the accuracy
    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
    accuracy = metric(predicted_classes, target_classes)

    # Calculate the per-class classification report
    clf_report = classification_report(target_classes, predicted_classes, target_names=classes)

    return test_loss/len(test_loader.dataset), accuracy.item(), clf_report 