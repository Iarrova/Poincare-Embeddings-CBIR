
import numpy as np

from torchsummary import summary
from scipy.spatial import distance_matrix

import torch
import torch.nn as nn

import torchvision.models as models

from datasets.CIFAR10 import generate_CIFAR10
from datasets.CIFAR100 import generate_CIFAR100
from networks.vgg16 import create_vgg16_model as network
from metrics.mAP import get_embeddings

np.random.seed(42)

# Check CUDA availability to use GPU
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if device == 'cuda':
    print('[INFO] CUDA is available! Using GPU...')
    NUM_WORKERS = 1
    PIN_MEMORY = True
else:
    print('[INFO] CUDA is not available. Using CPU...')
    NUM_WORKERS = 4
    PIN_MEMORY = False


def delta_hyp(dist_matrix):
    '''
    Computes the delta-hyperbolicity value from a distance matrix
    '''
    p = 0
    row = dist_matrix[p, :][np.newaxis, :]
    col = dist_matrix[:, p][:, np.newaxis]

    XY_p = 0.5 * (row + col - dist_matrix)

    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    return np.max(maxmin - XY_p)


def get_delta(model, loader, device):
    """
    Computes the delta-hyperbolicity value for image data by extracting features using VGG network.
    """
    embeddings, _ = get_embeddings(feature_extractor, loader, device)

    print('[INFO] Calculating Distance Matrix')
    dists = distance_matrix(embeddings, embeddings)

    idx = np.random.choice(len(embeddings), 1500)
    dists = embeddings[idx]

    delta = delta_hyp(dists)
    diam = np.max(dists)
    return delta, diam

# Generate the loaders for the CIFAR10 dataset
train_loader, validation_loader, test_loader, classes, superclasses = generate_CIFAR10(validation_size=0.0)
num_classes = 10

# Generate the feature extractor model
# We use a curvature of 1.0 and dimension of 32 as these present the best euclidean results for our model
model = network(1.0, 32, False, True, num_classes=num_classes)
model = model.to(device)

model.load_state_dict(torch.load('./weights/euclidean/CIFAR10_VGG16_32.pth'), strict=False)
summary(model, (3, 32, 32))

delta, diam = get_delta(train_loader, device)
print('Delta-Hiperbolicity:', delta)
print('Diameter:', diam)