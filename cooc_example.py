""" co-occurrence calculation example for a given activation map
"""
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from cooccurrence.cooccurrences import (
    ini_cooc_filter,
    calc_spatial_cooc
)

IMAGE_PATH = "./data/all_souls_000117.jpg"
ACT_MAP_PATH = "./data/all_souls_000117.npy"
COOC_FILTER_PATH = "./data/weights_cooc_44_best_model_8192_ft.npy"

def plot_figures(img, act_m, cooc_m_f, cooc_m_l):
    """ Method to plot the example figures
    """

    _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))

    ax1.imshow(img)
    ax1.set(title='Image')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.tick_params(axis='both', which='both', length=0)

    ax2.imshow(np.sum(act_m, axis=0))
    ax2.set(title='Activations')
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.tick_params(axis='both', which='both', length=0)

    ax3.imshow(np.sum(cooc_m_f, axis=0))
    ax3.set(title='Direct co-occurrences')
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax3.tick_params(axis='both', which='both', length=0)

    ax4.imshow(np.sum(cooc_m_l, axis=0))
    ax4.set(title='Learned co-occurrences')
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    ax4.tick_params(axis='both', which='both', length=0)

    plt.show()


if __name__ == '__main__':
    """ Main """

    parser = ArgumentParser()
    parser.add_argument('--cooc_r', dest='cooc_r', type=int, default=4, help='cooc r size')
    args = parser.parse_args()

    # Load image and activation map
    image = Image.open(IMAGE_PATH)
    act_map = np.load(ACT_MAP_PATH)
    act_map_t = torch.FloatTensor(np.expand_dims(act_map, axis=0)).cuda()

    depth = act_map_t.shape[1]

    # Co-occurrences calcultation
    # Direct cooc_filter
    cooc_filter = ini_cooc_filter(depth, args.cooc_r)
    cooc_map_fix = calc_spatial_cooc(act_map_t, cooc_filter, args.cooc_r)
    cooc_map_fix = cooc_map_fix.cpu().data.squeeze().numpy()

    # Co-occurrences calcultation
    # Learned cooc_filter trained previously
    cooc_filter = np.load(COOC_FILTER_PATH)
    cooc_filter = torch.FloatTensor(cooc_filter).cuda()
    cooc_map_learned = calc_spatial_cooc(act_map_t, cooc_filter, 4)
    cooc_map_learned = cooc_map_learned.cpu().data.squeeze().numpy()


    # Graphic representation
    plot_figures(image, act_map, cooc_map_fix, cooc_map_learned)
