import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from config_plot_params import *

# Parameters for the spike shape density plot
amp_min = -150
amp_max = 150
steps = 150

DENSITY_BINS = np.linspace(amp_min, amp_max, steps)
xticks = np.linspace(0, 19, 5)
ticklabels = [-1000, -500, 0, 500, 1000]

yticks = np.linspace(0, 150, 3)
yticklabels = [amp_min, 0, amp_max]

ycoord1 = -0.3
ycoord2 = 0.3
def spike_amp_plot_asset(ax, amps, cmap, invert=False, DENSITY_BINS=DENSITY_BINS, yicks=yticks, ytick_labels=yticklabels):
    """
    Generate the density plot for the spike shape of a given unit. 
    """
    if invert: 
        c = "white"
    else:
        c = "black"
    data = np.array([np.histogram(row, bins=DENSITY_BINS)[0]
                                    for row in amps.T])
    
    ax.imshow(data.T, aspect='auto', origin='lower',cmap=sns.color_palette(cmap, as_cmap=True))
    
    plt.draw()

    ax.set_xticks([0,63])
    ax.set_xticklabels([0, 2], fontsize=ticklabelsize-5, )
    for label in ax.get_xticklabels():
        label.set_y(.1)

    # ax.set_yticks(np.linspace(0, 150, 3))
    # ax.set_yticklabels(np.linspace(-150,150,3, dtype=int), color=c, fontsize=ticklabelsize-5)
    ax.set_yticks(yicks)
    ax.set_yticklabels(ytick_labels, color=c, fontsize=ticklabelsize-5)
    for label in ax.get_yticklabels():
        label.set_y(.1)
        label.set_x(.1)

    ax.set_xlabel('ms', color=c, fontsize=labelsize)
    ax.xaxis.set_label_coords(0.5,-0.1)
    
    ax.set_ylabel(r'$\mu V$', fontsize=labelsize, color=c, rotation=0, loc="center")
    ax.yaxis.set_label_coords(-0.4,0.4)
    
    if invert:
        ax.tick_params(axis='both', which='major', width=ticksize-5)
    else:
        ax.tick_params(axis='both', which='major', width=0)

    aspect = np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()) #force square ratio
    ax.set_aspect(aspect)

    return ax

def raster_plot_asset(ax, event_spikes, subsample=False, raster_linelength=2, y_label=True, invert=False):
    """
    Generate the unit's PSTH for the label onsets during the movie.
    """
    
    if invert: 
        c = "white"
        line_c = "lightgrey"
        line_width = 4
        raster_lw = 2
    else:
        c = "black"
        line_c = "grey"
        line_width = 4
        raster_lw = 2

    if subsample:
        event_spikes = random.sample(event_spikes, k=subsample)

    ax.eventplot(event_spikes, linelengths=raster_linelength, linewidths=raster_lw, color=c)
    ylims = ax.get_ylim()

    adjuster = (ylims[1] - ylims[0]) * 0.05

    ax.vlines(0, ylims[0]+adjuster, ylims[1]-adjuster, color=line_c, linewidth=line_width)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.tick_params(bottom = False)
    ax.patch.set_alpha(0)

    if y_label:
        ax.set_ylabel("Label onsets\nduring movie", rotation=0, fontsize=labelsize, color=c)
        ax.yaxis.set_label_coords(ycoord1, 0.45)
    else: 
        ax.set_ylabel("")
        
    return ax

def firingrate_plot_asset(ax, ax_raster, act_z, bins, 
                          l_c, 
                          x_label=True, y_label=True, invert=False):
    """
    Plot the unit's firing rate surrounding the label onsets as a lineplot.
    """

    if invert: 
        c = "white"
        line_c = "lightgrey"
        line_width = 5
        l_c = "#DBF3E1"
    else:
        c = "black"
        line_c = "grey"
        line_width = 2

    df = df_formatter_bestUnits(act_z, bins)
    lp = sns.lineplot(data=df, x="bin", y="vals", ax=ax, color=l_c, linewidth=line_width,)

    ylims = ax.get_ylim()
    ax.vlines(0, round(ylims[0],1),round(ylims[1],1), color=line_c, linewidth=4, zorder=1)
    ax.set_yticks([round(ylims[0],1), round(ylims[1],1)])
    ax.set_yticklabels([round(ylims[0],1), round(ylims[1],1)], color=c, fontsize=ticklabelsize)
    ax.set_xticks(ticklabels)
    ax.set_xticklabels(ticklabels, color=c, fontsize=ticklabelsize)
    plt.setp(ax_raster.get_xticklabels(), visible=False) # make tick labels for raster plot invisible

    if y_label:
        ax.set_ylabel("Firing rate\n[spikes / bin]", rotation=0, color=c, fontsize=labelsize)
        ax.yaxis.set_label_coords(ycoord1, 0.35)
        #ax.set_xlabel("")
    
    if x_label:
        ax.set_xlabel('Time [ms]', color=c, fontsize=labelsize)
        #ax.xaxis.set_label_coords(0.5, -0.5)
       # ax.set_ylabel("")
    
    if x_label is False:
        ax.set_xlabel("")
    if y_label is False:
        ax.set_ylabel("")
        
    return ax

def df_formatter_bestUnits(act_z, bins):
    bin_id = []
    vals = []

    for trial in act_z:
        for ind, val in enumerate(trial):
            bin_id.append(bins[ind])
            vals.append(val)
            
    return pd.DataFrame({"bin": bin_id, "vals": vals})