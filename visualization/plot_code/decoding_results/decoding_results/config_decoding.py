"""
File to save all configurations for the plots
"""

import seaborn as sns

# COLORS
character_color = sns.color_palette("rocket", n_colors=1)[0]
transitions_color = sns.color_palette("mako", n_colors=1)[0]
location_color = sns.color_palette("BrBG_r", n_colors=8)[6]
extra_color = sns.color_palette("BuPu", n_colors=1)[0]
grey_color = sns.color_palette("Greys", n_colors=1)[0]

# COLOR PALETTES
character_palette = sns.color_palette("rocket", n_colors=9)[5:]
transitions_palette = sns.color_palette("mako", n_colors=9)[5:]
location_palette = sns.color_palette("BrBG_r", n_colors=9)[5:]
extra_palette = sns.color_palette("BuPu", n_colors=6)[1:]
grey_palette = sns.color_palette("bone_r", n_colors=4)

# COLOR LIGHT
character_color_light = sns.light_palette(character_color, n_colors=5)[1]
transitions_color_light = sns.light_palette(transitions_color, n_colors=5)[1]
location_color_light = sns.light_palette(location_color, n_colors=5)[1]

# SIZE FIGURE
figure_x = 30
figure_y = 9

# PLOTS - BARS
capsize_err = 8
fontsize_labels_bar = 32

# X AXIS
x_fontsize_label = 32
x_fontsize_tick_labels = 34

# Y AXIS
y_labelpad = 20
y_fontsize_label = 44
y_fontsize_tick_labels = 36

# GENERAL
fontsize_legend = 26
frameon_legend=False
linewidth_spines = 6
tick_length = 16
tick_width = 4
