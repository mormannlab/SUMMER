import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import sys
import os
import seaborn as sns

import config_decoding as config
print(f"Fontsize Labels Bar: {config.capsize_err}")

import config_colors as config_colors
from config_plot_params import *

from matplotlib import font_manager as fm
# set global font to be Helvetica
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
helvetica_path = os.path.join(current_dir, 'fonts', 'Helvetica-Bold.ttf')
font_prop = fm.FontProperties(fname=helvetica_path)
print("Loaded font name:", font_prop.get_name())
fm.fontManager.addfont(helvetica_path)
rc('font', family=font_prop.get_name())

def plot_decoding(values, values_err, metric, labels, save_path, ymin=0, ymax=1.0):
    
    # COLORS FULL
    #color_character = config.character_color
    color_character = config_colors.colors[0]
    #color_transition = config.transitions_color
    color_transition = config_colors.colors[3]
    #color_location = config.location_color
    color_location = config_colors.colors[4]
    colors_full = [color_character,color_character,color_character,color_character,color_transition,color_transition,color_location,color_location]

    x = np.arange(len(labels))  # the label locations

    # Define index
    if metric=="Cohen's Kappa":
        idx=0
    elif metric=="F1 Score":
        idx=1
    elif metric=="PR-AUC":
        idx=2
    elif metric=="AUROC":
        idx=3
    
    fig, ax = plt.subplots(figsize=(config.figure_x, config.figure_y))
    width=0.7 #the width of the bars
    fontsize = config.fontsize_labels_bar
    capsize = config.capsize_err

    for i in range(len(labels)):
        bar1 = ax.bar(x[i], values[i][idx], yerr=values_err[i][idx], width=width, color=colors_full[i], capsize=capsize)
        
        if i in [4,5]:
            plt.bar_label(container=bar1, labels=[np.round(values[i][idx],2)], padding=5, fontsize=config.fontsize_labels_bar)
        elif i in [0]:
            plt.bar_label(container=bar1, labels=[np.round(values[i][idx],2)], padding=20, fontsize=config.fontsize_labels_bar)
        else:
            plt.bar_label(container=bar1, labels=[np.round(values[i][idx],2)], padding=5, fontsize=config.fontsize_labels_bar)

    # X AXIS
    #ax.set_xlabel("Label", labelpad=config.y_labelpad, fontsize=config.x_fontsize_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, weight='bold')
    ax.tick_params(axis='x', labelsize=config.x_fontsize_tick_labels, length=config.tick_length, width=config.tick_width)
    # Different padding for ticks
    i=0
    for tick in ax.get_xaxis().get_major_ticks():
        if i%2==0:
            tick.set_pad(12.)
        else:
            tick.set_pad(45.)
        i+=1

    # YAXIS
    y = np.arange(0,1.3,0.2)  # the label locations
    print(f"y locations: {y}")
    ax.set_yticks(y)
    yaxis=f"Decoding Performance \n [{metric}]"
    plt.yticks(fontsize=config.y_fontsize_tick_labels)
    ax.set_ylabel(yaxis, fontsize=config.y_fontsize_label, labelpad=config.y_labelpad)
    ax.tick_params(axis='y', length=config.tick_length, width=config.tick_width, labelsize=config.y_fontsize_tick_labels)
    ax.set_ylim(ymin, ymax)

    # GENERAL
    #ax.tick_params(axis='both', color='black', labelsize=config.x_fontsize_tick_labels, length=config.x_tick_length, width=config.x_tick_width)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=config.linewidth_spines)

    # COLOR X LABELS
    print(f"Number of x ticks: {len(plt.gca().get_xticklabels())}")
    for i in range(len(x)):
        plt.gca().get_xticklabels()[i].set_color(colors_full[i])

    # TRIM AXES
    #sns.despine(top=True, right=True, trim=True, ax=ax)

    """# LEGEND (Option 1)
    pa1 = Patch(facecolor=config.character_color, edgecolor='black')
    pa2 = Patch(facecolor=config.transitions_color, edgecolor='black')
    pa3 = Patch(facecolor=config.location_color, edgecolor='black')
    pa1_light = Patch(facecolor=config.character_color_light, edgecolor='black')
    pa2_light = Patch(facecolor=config.transitions_color_light, edgecolor='black')
    pa3_light = Patch(facecolor=config.location_color_light, edgecolor='black')

    plt.legend(
        [pa1, pa1_light, pa2, pa2_light, pa3, pa3_light],
        ['', '', '', '', 'Recurrent neural network', 'Logistic regression'],
        #['', 'All','','A','','H','','EC','','PHC'],
        #ncol=1, 
        handletextpad=0.5, 
        handlelength=1.0, 
        columnspacing=-0.5, #-0.5,
        #labelspacing=6,
        frameon=config.frameon_legend,
        ncols=3, 
        fontsize=config.fontsize_legend
    )"""

    plt.tight_layout()
    if metric=="Cohen's Kappa":
        name= 'kappa'
    elif metric=="F1 Score":
        name= 'f1'
    elif metric=="PR-AUC":
        name= 'pr_auc'
    elif metric=="AUROC":
        name= 'auroc'
    plt.savefig(fname=os.path.join(save_path, f"decoding.jpg"), bbox_inches='tight', dpi=600)
