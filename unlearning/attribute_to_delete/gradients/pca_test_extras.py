

from pathlib import Path
import yaml
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
# from opacus.utils.batch_memory_manager import BatchMemoryManager
from torchvision import models

import local_models
from datasets import initialize_data_CIFAR10, initialize_data_SVHN, initialize_data_MNIST

from tqdm import tqdm
from scipy.sparse.linalg import svds as scipy_sparse_svds
from sklearn.utils.extmath import randomized_svd
import sys
import logging
import pickle
import scipy
from datetime import datetime 

SVHN_MEAN = (0.4376821, 0.4437697, 0.47280442)
SVHN_STD_DEV = (0.19803012, 0.20101562, 0.19703614)

transform_SVHN = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(SVHN_MEAN, SVHN_STD_DEV),
])
