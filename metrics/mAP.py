import math
import random

import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances

import torch

from hyptorch.pmath import dist

def get_embeddings(model, loader, device):
    '''
    Calculates the embeddings for a DataLoader.
    '''
    embeddings = []
    target_classes = []

    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            embeddings.append(output.cpu().numpy())
            target_classes.append(target.cpu().numpy())
        
    embeddings = np.concatenate(embeddings)
    target_classes = np.concatenate(target_classes)
    
    return embeddings, target_classes


def generate_database(embeddings, target_classes, query_size):
    '''
    Splits a group of embeddings and their labels into queries and a database group.
    Queries will be used to measure the mAP against the database. There are no repeated elements between groups.
    '''
    # Get the subset that will be used as queries
    number_of_queries = int(math.floor(len(embeddings) * query_size))
    queries_indices = random.sample(range(len(embeddings)), number_of_queries)

    # Remove the queries from the embeddings database
    queries = embeddings[queries_indices]
    queries_classes = target_classes[queries_indices]

    database = np.delete(embeddings, queries_indices, axis=0)
    database_classes = np.delete(target_classes, queries_indices)

    return queries, queries_classes, database, database_classes


def retrieve_n_similar_elements(queries, queries_classes, database, database_classes, curvature, hyperbolic, mAP_at):
    # Define the hyperbolic distance function for pairwise_distances
    def hyperbolic_distance(X, Y, curvature=curvature):
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        return dist(X, Y, c=curvature)

    class_retrieval = []
    for query in queries:
        if hyperbolic:
            pairwise_distance = pairwise_distances([query], database, metric=hyperbolic_distance).flatten()
            
        elif not hyperbolic:
            pairwise_distance = pairwise_distances([query], database).flatten()

        # Get the index of the n-nearest embeddings
        n_nearest_indices = np.argsort(pairwise_distance)[:mAP_at]
        
        # Translate n nearest indices to corresponding class
        retrieved_elements = database_classes[n_nearest_indices]
        
        class_retrieval.append(retrieved_elements)
        
    class_retrieval = np.array(class_retrieval)
    class_retrieval = pd.DataFrame(class_retrieval)
    class_retrieval.insert(0, 'Query', queries_classes)
    
    return class_retrieval


def average_precision(query_predictions):
    '''
    Receives an array of ordered predictions with True or False, where True is a relevant document and False isn't a relevant document
    Calculates the average precision as \frac{1}{GTP} \sum_{k}^{n} P@k \cdot rel@k
    Returns the average precision, a value between 0 and 1
    '''
    correct_predictions = 0
    running_sum = 0

    for i, pred in enumerate(query_predictions):
        k=i+1
        if pred:
            correct_predictions += 1
            running_sum += (correct_predictions / k)
    
    return running_sum / len(query_predictions)

def mean_average_precision(class_retrieval, mAP_at):
    '''
    Receives a 2D array of all predictions, where each subarray has True or False values, depending on the relevancy of the retrieved document.
    Calculates the mean average precision as the mean of all average precisions.
    Returns the mean average precision, a value between 0 and 1
    '''
    # Transform class retrieval into True/False vectors
    predictions = []

    for i, row in class_retrieval.iterrows():
        query_predictions = []
        for j in range(mAP_at):
            query_predictions.append(row['Query'] == row[j])
        predictions.append(query_predictions)

    # Calculate mean Average Precision
    running_sum = 0
    for prediction in predictions:
        running_sum += average_precision(prediction)

    return running_sum / len(predictions)

