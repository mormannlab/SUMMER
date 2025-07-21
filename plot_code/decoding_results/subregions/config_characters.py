"""
File to save all configurations for the plots
"""

import seaborn as sns

# COLORS
character_color = sns.color_palette("rocket", n_colors=1)[0]
transitions_color = sns.color_palette("mako", n_colors=1)[0]
location_color = sns.color_palette("BrBG_r", n_colors=8)[6]
extra_color = sns.color_palette("Purples", n_colors=10)[-2]
grey_color = sns.color_palette("Greys", n_colors=1)[0]

# COLORS ADAPTED
character_color = sns.color_palette("rocket", n_colors=9)[2]
character_dark1 = sns.color_palette("rocket", n_colors=9)[4]
character_dark2 = sns.color_palette("rocket", n_colors=9)[5]
character_light1 = sns.color_palette("rocket", n_colors=9)[6]
character_light2 = sns.color_palette("rocket", n_colors=9)[7]

# SIZE FIGURE
figure_x = 14
figure_y = 8

# PLOTS - BARS
capsize_err = 6
fontsize_labels_bar = 22

# X AXIS
x_fontsize_label = 30
x_fontsize_tick_labels = 38

# Y AXIS
y_labelpad = 10
y_fontsize_label = 38
y_fontsize_tick_labels = 24

# GENERAL
fontsize_legend = 26
frameon_legend = False
linewidth_spines = 5
tick_length = 12
tick_width = 3
