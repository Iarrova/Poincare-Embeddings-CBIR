import torch.nn as nn

import torchvision.models as models

import hyptorch.nn as hypnn

def create_vgg16_model(curvature, embedding_size, hyperbolic=False, extractor=False, num_classes=10):
    '''
    Generates a VGG16_BN model, which can be Euclidean or Hyperbolic.
    
    Parameters:
    -----------
    curvature : float
        The curvature of the Poincare Ball
    embedding_size : int
        The dimension of the embedding to use
    hyperbolic : bool, optional
        Flag to specify if the model is hyperbolic or euclidean
    extractor : bool, optional
        Flag to specify wether the model will be used as a classifier or a feature extractor
    num_classes : int, optional
        Number of classes in the dataset
    
    Returns:
    --------
    A PyTorch model.
    '''
    model = models.vgg16_bn(weights='IMAGENET1K_V1')
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    classifier = [nn.Linear(512, 512), nn.ReLU(inplace=True,), nn.Dropout(0.5), nn.Linear(512, embedding_size)]

    if hyperbolic:
        classifier.append(hypnn.ToPoincare(c=curvature, ball_dim=embedding_size))
        if (not extractor):
            classifier.append(hypnn.HyperbolicMLR(c=curvature, ball_dim=embedding_size, n_classes=num_classes))
            classifier.append(nn.LogSoftmax(dim=1))
    
    elif (not hyperbolic):
        if (not extractor):
            classifier.append(nn.ReLU(inplace=True))
            classifier.append(nn.Linear(embedding_size, num_classes))
            classifier.append(nn.LogSoftmax(dim=1))

    model.classifier = nn.Sequential(*classifier)
    return model