import argparse

import pandas as pd

from datasets.hierarchies import get_similar_classes
from metrics.mAP import average_precision

# -------------------------------------
# ----------- Initial Setup -----------
# -------------------------------------
# Testing settings
parser = argparse.ArgumentParser(description="Hyperbolic Embeddings for Content-Based Image Retrieval.")
parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100'], help='Define which dataset to use. Currently only CIFAR10 and CIFAR100 are available.')
parser.add_argument('--retrieval_path', help='Path to store the calculated retrieval matrix.')
parser.add_argument('--mAP_at', type=int, default=10, help='mAP@k, define the k. Default is 10')
parser.add_argument('--verbose', action='store_false', help='Increase output verbosity. Default is True')

args = parser.parse_args()

if args.dataset == 'CIFAR10':
    hierarchical_levels = 3
elif args.dataset == 'CIFAR100':
    hierarchical_levels = 5

class_retrieval = pd.read_csv(args.retrieval_path)

for granularity in range(hierarchical_levels, 0, -1):
    predictions = []
    for i, row in class_retrieval.iterrows():
        query_predictions = []
        for j in range(args.mAP_at):
            query_predictions.append(row[j+1] in get_similar_classes(row['Query'], granularity, 'CIFAR100'))
        predictions.append(query_predictions)

    running_sum = 0
    for prediction in predictions:
        running_sum += average_precision(prediction)
    print('Granularity {}: mAP@10: {:.4f}'.format(granularity, (running_sum / len(predictions))))