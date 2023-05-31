import random
import argparse

import numpy as np

from torchsummary import summary

import torch
import torch.nn as nn

from metrics.accuracy import measure_accuracy
from metrics.mAP import get_embeddings, generate_database, retrieve_n_similar_elements, mean_average_precision

# -------------------------------------
# ----------- Initial Setup -----------
# -------------------------------------
# Testing settings
parser = argparse.ArgumentParser(description="Hyperbolic Embeddings for Content-Based Image Retrieval.")
parser.add_argument('--hyperbolic', action='store_true', help='Define an hyperbolic or euclidean model. Default is Euclidean.')
parser.add_argument('--weights_path', help='Path to load the model')
parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100'], help='Define which dataset to use. Currently only CIFAR10 and CIFAR100 are available.')
parser.add_argument('--retrieval_path', help='Path to store the calculated retrieval matrix.')
parser.add_argument('--network', choices=['VGG16'], default='VGG16', help='Define the network architecture to use for embedding extraction. Currently only VGG16 is supported')
parser.add_argument("--curvature", type=float, default=1.0, help="Curvature of the Poincare ball. Default is 1.0")
parser.add_argument("--dimension", type=int, default=2, help="Dimension of the Poincare ball. Default is 2")
parser.add_argument("--query_size", type=float, default=0.1, help="Porcentage of testing embeddings to use as queries to measure mAP")
parser.add_argument('--mAP_at', type=int, default=10, help='mAP@k, define the k. Default is 10')
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
train_loader, validation_loader, test_loader, classes = generate_dataset(1000, 0.0, NUM_WORKERS, PIN_MEMORY)


# -------------------------------------
# -------------- Network --------------
# -------------------------------------
if args.network == 'VGG16':
    from networks.vgg16 import create_vgg16_model as network
else:
    print('[ERROR] Currently only VGG16 network is supported. Exiting...')

# Load Model
model = network(args.curvature, args.dimension, args.hyperbolic, False, num_classes=num_classes)
model = model.to(device)

model.load_state_dict(torch.load(args.weights_path))

# Specify loss function
criterion = nn.NLLLoss()


# -------------------------------------
# ---------- Testing Accuracy ---------
# -------------------------------------
# Get testing loss, accuracy and classification report
test_loss, accuracy, classification_report = measure_accuracy(model, criterion, test_loader, classes, device, num_classes)
print('Test Loss: {:.4f}'.format(test_loss))
print('Accuracy: {:.4f}'.format(accuracy))
print('Classification Report:\n', classification_report)


# -------------------------------------
# ----------- Testing mAP@10 ----------
# -------------------------------------
# Reload model without classification head to get embeddings
model = network(args.curvature, args.dimension, args.hyperbolic, True, num_classes=num_classes)
model = model.to(device)

model.load_state_dict(torch.load(args.weights_path), strict=False)

if args.verbose:
    summary(model, (3, 32, 32))

# Get embeddings for the testing data
embeddings, target_classes = get_embeddings(model, test_loader, device)

# Generate the database and queries to calculate mAP
queries, queries_classes, database, database_classes = generate_database(embeddings, target_classes, args.query_size)

# Get the n most similar embeddings for each query from the database, and translate it to the class
class_retrieval = retrieve_n_similar_elements(queries, queries_classes, database, database_classes, args.curvature, args.hyperbolic, args.mAP_at)
class_retrieval.to_csv(args.retrieval_path, index=False)

# Calculate the mean Average Precision for all queries
mAP = mean_average_precision(class_retrieval, args.mAP_at)
print('Mean Average Precision: {:.4f}'.format(mAP))