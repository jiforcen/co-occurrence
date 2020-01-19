""" Co-occurrence methods needed to obtain the co-occurrence representation
"""
import math
import numpy as np
import torch

def ini_cooc_filter(channels, cooc_r):
    """ Method to obtain a co-occurrence filter
    given the number of channels and r
    """
    cooc_w = cooc_r * 2 + 1
    cooc_filter = np.ones((channels, channels, 1, 1))/math.pow(cooc_w, 2)
    for i in range(channels):
        cooc_filter[i, i, :, :] = 1e-10
    cooc_filter = torch.FloatTensor(cooc_filter).cuda()
    cooc_filter = cooc_filter.repeat(1, 1, cooc_w, cooc_w)
    return cooc_filter

def calc_spatial_cooc(tensor, cooc_filter, cooc_r):
    """ Method to obtain the co-occurrence representation
    given an activation tensor, co-occurrence filter and the size r
    """
    act_m = tensor > torch.mean(tensor)
    act = tensor * act_m.float()
    cooc_map = torch.nn.functional.conv2d(act, cooc_filter, padding=cooc_r)

    cooc_map = cooc_map / (tensor.shape[1] - 1)
    cooc_map = cooc_map * act_m.float()
    return cooc_map
