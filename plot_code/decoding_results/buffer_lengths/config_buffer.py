"""
File to save all configurations for the plots
"""

import seaborn as sns

# COLORS
character_color = sns.color_palette("rocket", n_colors=1)[0]
transitions_color = sns.color_palette("mako", n_colors=1)[0]
location_color = sns.color_palette("BrBG_r", n_colors=8)[6]
#extra_color = sns.color_palette("Purples", n_colors=10)[-2]
extra_color = sns.color_palette("bone", n_colors=5)[2]
#grey_color = sns.color_palette("bone", n_colors=5)[2]
#grey_color = sns.color_palette("bone", n_colors=8)[6]
grey_color = sns.color_palette("binary", n_colors=7)[1]

# COLOR PALETTES
character_palette = sns.color_palette("rocket", n_colors=9)[5:]
transitions_palette = sns.color_palette("mako", n_colors=4)
location_palette = sns.color_palette("BrBG", n_colors=4)
extra_palette = sns.color_palette("BuPu", n_colors=4)
grey_palette = sns.color_palette("bone_r", n_colors=4)

# SIZE FIGURE
figure_x = 11.5
figure_y = 7.2

# PLOTS - BARS
capsize_err = 8
fontsize_labels_bar = 20

# GENERAL AXIS
labelpad = 20

# X AXIS
x_fontsize_label = 32
x_fontsize_tick_labels = 26

# Y AXIS
y_fontsize_label = 32
y_fontsize_tick_labels = 26

# GENERAL
fontsize_legend = 26
frameon_legend=False
linewidth_spines = 4
tick_length = 10
tick_width = 4
