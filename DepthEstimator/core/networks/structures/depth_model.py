from __future__ import absolute_import, division, print_function
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


class Depth_Model(nn.Module):
    def __init__(self, num_layers=50):
        super(Depth_Model, self).__init__()
        pass

    def forward(self, x):
        pass