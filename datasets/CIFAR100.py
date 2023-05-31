import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets, transforms


def generate_validation_indices(train_data, validation_size=0.2):
    '''
    Generates indices to be used as validation from training data

    Parameters:
    -----------
    train_data: PyTorch Dataset
        The Dataset that contains the training data from which we will sample the validation set
    validation_size: Float, optional
        The porcentage of the training data to be used as validation. Default=0.2
    '''
    training_size = len(train_data)
    indices = list(range(training_size))
    np.random.shuffle(indices)
    split = int(np.floor(validation_size * training_size))
    train_idx, valid_idx = indices[split:], indices[:split]

    return train_idx, valid_idx

def generate_CIFAR100(batch_size=128, validation_size=0.2, num_workers=4, pin_memory=False):
    # Convert data to a normalized torch.FloatTensor
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])


    # Choose the training and test datasets
    train_data = datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)

    # Get the validation and training indices
    train_idx, valid_idx = generate_validation_indices(train_data, validation_size)

    # Define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(valid_idx)

    # Prepare data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    validation_loader = DataLoader(train_data, batch_size=batch_size, sampler=validation_sampler, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    # Specify the image classes
    classes = [k for k in train_data.class_to_idx.keys()]

    return train_loader, validation_loader, test_loader, classes