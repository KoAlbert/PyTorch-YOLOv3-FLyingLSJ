import torch
import torch.nn.functional as F
import numpy as np


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    #targets[:, 2] = 1 - targets[:, 2]
    targets[:, -4] = 1 - targets[:, -4] # this will fulfill the original one and multi-label one. Alex
    return images, targets
