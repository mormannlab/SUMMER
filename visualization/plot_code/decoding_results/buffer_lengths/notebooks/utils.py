import os
import numpy as np
import matplotlib.pyplot as plt
import sys

import config_buffer as config
print(f"Fontsize Labels Bar: {config.capsize_err}")

import config_colors as config_colors
from config_plot_params import *


def plot_buffer_lengths_evaluation(values, values_err, labels, metric, mode, save_path):
    # Define colors
    colors=[config.grey_color, config_colors.colors[0]]

    x = np.arange(len(labels))  # the label locations
    width = 0.8  # the width of the bars
    print(f'X: {x-width/2}')
    print(f'Type of : {width/2}')
    
    # Define index
    if metric=="Cohen's kappa":
        idx=0
        filename='KAPPA'
    elif metric=="F1 Score":
        idx=1
        filename='F1'
    elif metric=="PR-AUC":
        idx=2
        filename="PR-AUC"
    elif metric=="AUROC":
        idx=3
        filename='AUROC'

    fig, ax = plt.subplots(figsize=(config.figure_x, config.figure_y))#(figsize=(25, 10))

    # plot bars
    for i in range(len(labels)):
        if labels[i] == '32':
            bar = ax.bar(x[i], values[i][idx], yerr=values_err[i][idx], width=width, color=colors[1], capsize=config.capsize_err)
        else:
            bar = ax.bar(x[i], values[i][idx], yerr=values_err[i][idx], width=width, color=colors[0], capsize=config.capsize_err)
        ax.bar_label(container=bar, labels=[f"{np.round(values[i][idx],2)}"], padding=5, fontsize=config.fontsize_labels_bar)
    
    # X AXIS
    ax.set_xlabel('Temporal Gaps [s]', fontsize=config.x_fontsize_label, labelpad=config.labelpad)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', color='black', labelsize=config.x_fontsize_tick_labels, length=config.tick_length, width=config.tick_width)
    
    # Y AXIS
    yaxis=f"Decoding Performance \n [{metric}]"
    y=np.arange(0,1.2, 0.2)
    ax.set_yticks(y)
    ax.set_ylabel(yaxis, labelpad=config.labelpad, fontsize=config.y_fontsize_label)
    ax.tick_params(axis='y', color='black', length=config.tick_length, width=config.tick_width, labelsize=config.y_fontsize_tick_labels)
    ymin=0
    ymax=1.0
    plt.ylim(ymin,ymax)

    # GENERAL
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=config.linewidth_spines)

    plt.tight_layout()
    plt.savefig(fname=os.path.join(save_path, f"{mode}", f"buffer_lengths.jpg"), bbox_inches='tight')