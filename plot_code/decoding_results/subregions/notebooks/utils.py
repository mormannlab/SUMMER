import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns

import config_characters as config
import config_colors as config_colors
print(f"Fontsize Labels Bar: {config.capsize_err}")

def regional_differences(values, values_err, metric, buffer, label_names, title, save_path, save_name, legend=True, fig_x=None, fig_y=None, ymax=1, ymin=0):

    # Define colors
    colors = [config_colors.colors[0], config_colors.color_by_region["A"], config_colors.color_by_region["H"], config_colors.color_by_region["EC"], config_colors.color_by_region["PHC"]]

    # Define x-axis
    labels=['All', 'A', 'H', 'EC', 'PHC']
    
    if fig_x is None and fig_y is None:
        fig, ax = plt.subplots(figsize=(config.figure_x, config.figure_y))
    else:
        fig, ax = plt.subplots(figsize=(fig_x, fig_y))
    
    # Characters
    width=0.15
    extra=0.02
    x = np.arange(len(label_names))  # the label locations
    shift=np.array([-2*width-2*extra, -1*width-extra, 0, 1*width+extra, 2*width+2*extra])

    for i in range(len(label_names)):
        rects1=plt.bar(shift+x[i], values, yerr=values_err[i], width=width, color=colors, capsize=config.capsize_err)
        # Label manually the regions for the first label (similar color pattern for all others)
        if i==0:
            for k in range(len(rects1.patches)):
                rects1.patches[k].set_label(labels[k])
        # Annotate the bars with values
        plt.bar_label(container=rects1, labels=np.round(values,2), padding=5, fontsize=config.fontsize_labels_bar)

    # If metric AUROC, plot baseline=0.5
    if metric=="AUROC":
        plt.hlines(y=0.5, xmin=-0.5, xmax=len(label_names)-0.5, color='black', label='Chance Level')
    
    # X AXIS
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, weight='bold', color=config_colors.colors[0])
    ax.tick_params(axis='x', labelsize=config.x_fontsize_tick_labels, length=config.tick_length, width=config.tick_width)

    # Y AXIS
    yaxis = f"Decoding Performance \n [{metric}]"
    ax.set_ylabel(yaxis, fontsize=config.y_fontsize_label, labelpad=config.y_labelpad)
    ax.tick_params(axis='y', color='black', labelsize=config.y_fontsize_tick_labels, length=config.tick_length, width=config.tick_width)

    # GENERAL
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=config.linewidth_spines)
    plt.tight_layout()

    # LEGEND
    if legend:
        ax.legend(
            fontsize=config.fontsize_legend, 
            ncols=5, 
            frameon=config.frameon_legend)

    plt.ylim(ymin,ymax)

    if metric=="Cohen's Kappa":
        name= 'kappa'
    elif metric=="F1 Score":
        name= 'f1'
    elif metric=="PR-AUC":
        name= 'pr_auc'
    elif metric=="AUROC":
        name= 'auroc'
    
    #os.makedirs(os.path.join(save_path, f"BUFFER{buffer}"), exist_ok=True)
    plt.savefig(fname=os.path.join(save_path, f"{save_name}.jpg"), dpi=600)
