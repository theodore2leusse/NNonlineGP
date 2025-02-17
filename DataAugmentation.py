# sciNeurotech Lab 
# Theodore

# This file includes functions for augmenting the dataset to improve model robustness.

# import lib
import torch
from botorch.models import SingleTaskGP
from botorch.utils.transforms import standardize, normalize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import (UpperConfidenceBound, 
                                          ExpectedImprovement,
                                          NoisyExpectedImprovement, 
                                          LogExpectedImprovement,
                                          LogNoisyExpectedImprovement)
import numpy as np
from utils import *
import warnings
import time
from tqdm import tqdm
from scipy.stats import norm
import math
import itertools
import random
import pickle
import GPy



def augment_with_vertical_flip(train_X, train_Y) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Augments the dataset by adding vertically flipped images.

    Args:
        train_X (torch.Tensor): A tensor of shape [N, Cx, H, W], where:
            - N: Number of images,
            - Cx: Number of channels per image,
            - H: Height of the images,
            - W: Width of the images.
        train_Y (torch.Tensor): A tensor of shape [N, Cy, Hb, W], where:
            - N: Number of images,
            - Cy: Number of channels per image,
            - H: Height of the images,
            - W: Width of the images.
    
    Returns:
        torch.Tensor: A new tensor containing the original images and their vertically flipped versions,
                    with shape [2N, Cx, H, W].
        torch.Tensor: A new tensor containing the original images and their vertically flipped versions,
                    with shape [2N, Cy, H, W].
    """
    # Ensure the input tensor has 4 dimensions
    if train_X.dim() != 4 or train_Y.dim() != 4:
        raise ValueError("Inputs must have 4 dimensions [N, C, H, W].")
    
    # Flip the images vertically (reverse along the height dimension)
    flipped_X = torch.flip(train_X, dims=[2])  # Dimension 2 corresponds to the height (H)

    # Concatenate the original and flipped images along the batch dimension
    augmented_X = torch.cat((train_X, flipped_X), dim=0)

    # Flip the images vertically (reverse along the height dimension)
    flipped_Y = torch.flip(train_Y, dims=[2])  # Dimension 2 corresponds to the height (H)

    # Concatenate the original and flipped images along the batch dimension
    augmented_Y = torch.cat((train_Y, flipped_Y), dim=0)
    
    return augmented_X, augmented_Y

def augment_with_horizontal_flip(train_X, train_Y) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Augments the dataset by adding horizontally flipped images.

    Args:
        train_X (torch.Tensor): A tensor of shape [N, Cx, H, W], where:
            - N: Number of images,
            - Cx: Number of channels per image,
            - H: Height of the images,
            - W: Width of the images.
        train_Y (torch.Tensor): A tensor of shape [N, Cy, Hb, W], where:
            - N: Number of images,
            - Cy: Number of channels per image,
            - H: Height of the images,
            - W: Width of the images.
    
    Returns:
        torch.Tensor: A new tensor containing the original images and their horizontally flipped versions,
                    with shape [2N, Cx, H, W].
        torch.Tensor: A new tensor containing the original images and their horizontally flipped versions,
                    with shape [2N, Cy, H, W].
    """
    # Ensure the input tensor has 4 dimensions
    if train_X.dim() != 4 or train_Y.dim() != 4:
        raise ValueError("Inputs must have 4 dimensions [N, C, H, W].")
    
    # Flip the images horizontally (reverse along the width dimension)
    flipped_X = torch.flip(train_X, dims=[3])  # Dimension 3 corresponds to the width (W)

    # Concatenate the original and flipped images along the batch dimension
    augmented_X = torch.cat((train_X, flipped_X), dim=0)

    # Flip the images horizontally (reverse along the width dimension)
    flipped_Y = torch.flip(train_Y, dims=[3])  # Dimension 3 corresponds to the width (W)

    # Concatenate the original and flipped images along the batch dimension
    augmented_Y = torch.cat((train_Y, flipped_Y), dim=0)
    
    return augmented_X, augmented_Y

def augment_with_opposit_map_values(train_X, train_Y) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Augments the dataset by negating channels 0 and 3 for all images of train_X 
    and negating chanel 0 for train_Y.

    Args:
        train_X (torch.Tensor): A tensor of shape [N, Cx, H, W], where:
            - N: Number of images,
            - Cx: Number of channels per image,
            - H: Height of the images,
            - W: Width of the images.
        train_Y (torch.Tensor): A tensor of shape [N, Cy, Hb, W], where:
            - N: Number of images,
            - Cy: Number of channels per image,
            - H: Height of the images,
            - W: Width of the images.
    
    Returns:
        torch.Tensor: A new tensor containing the original images and their negated versions
                      (channel 0 and 3 multiplied by -1), with shape [2N, Cx, H, W].
        torch.Tensor: A new tensor containing the original images and their negated versions
                      (channel 0 multiplied by -1), with shape [2N, Cy, H, W].
    """
    # Ensure the input tensor has 4 dimensions
    if train_X.dim() != 4 or train_Y.dim() != 4:
        raise ValueError("Inputs must have 4 dimensions [N, C, H, W].")
    
    # Copy the input tensor to apply modifications
    opposit_X = train_X.clone()

    # Multiply channels 0 and 3 by -1
    opposit_X[:, 0, :, :] *= -1  # Negate channel 0
    opposit_X[:, 3, :, :] *= -1  # Negate channel 3

    # Concatenate the original and flipped images along the batch dimension
    augmented_X = torch.cat((train_X, opposit_X), dim=0)

    # Copy the input tensor to apply modifications
    opposit_Y = train_Y.clone()

    # Multiply channel 0 by -1
    opposit_Y[:, 0, :, :] *= -1  # Negate channel 0

    # Concatenate the original and flipped images along the batch dimension
    augmented_Y = torch.cat((train_Y, opposit_Y), dim=0)
    
    return augmented_X, augmented_Y

