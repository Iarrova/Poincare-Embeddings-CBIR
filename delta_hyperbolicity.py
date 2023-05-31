
import numpy as np

from tqdm import tqdm
from scipy.spatial import distance_matrix

import torch
import torch.nn as nn

import torchvision.models as models

from datasets.CIFAR10 import generate_CIFAR10
from datasets.CIFAR100 import generate_CIFAR100 
from metrics.mAP import get_embeddings

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


def create_feature_extractor_vgg16_bn(device):
    model = models.vgg16_bn(weights='IMAGENET1K_V1')
    model_features = model.features
    model_classifier = nn.Sequential(*list(model.classifier.children())[:-3])

    feature_extractor = nn.Sequential(model_features, nn.Flatten(), model_classifier).to(device)

    return feature_extractor


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


def get_delta(loader, device):
    """
    Computes the delta-hyperbolicity value for image data by extracting features using VGG network.
    """
    feature_extractor = create_feature_extractor_vgg16_bn(device)

    embeddings, _ = get_embeddings(feature_extractor, loader, device)

    print('[INFO] Calculating Distance Matrix')
    dists = distance_matrix(embeddings, embeddings)

    delta = delta_hyp(dists)
    diam = np.max(dists)
    return delta, diam

# Generate the loaders for the CIFAR10 dataset
train_loader, validation_loader, test_loader, classes, superclasses = generate_CIFAR10(validation_size=0.0)

delta, diam = get_delta(train_loader, device)
print('Delta-Hiperbolicity:', delta)
print('Diameter:', diam)