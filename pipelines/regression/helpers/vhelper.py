# -*- coding: utf-8 -*-
"""Datasets visualization.
"""
import os
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import seaborn as sns


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):   
    # Where to save the figures
    PROJECT_ROOT_DIR = "."
    OUT_IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "analytics_images")
    os.makedirs(OUT_IMAGES_PATH, exist_ok=True)
    
    path = os.path.join(OUT_IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)