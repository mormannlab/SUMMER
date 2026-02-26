#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameters for plot creation, e.g. height/width ratios, fig sizes, etc.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', )))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))

from matplotlib import font_manager as fm
from matplotlib import rc

axwidth= 3
tickwidth = 3
ticksize = 10
ticklabelsize= 12
small_ticklabelsize = 10
labelsize = 15
titlesize = 20

rc('axes', linewidth=axwidth)
rc('xtick.major', width=tickwidth, size=ticksize)
rc('xtick', labelsize=ticklabelsize)        
rc('ytick.major', width=tickwidth, size=ticksize)
rc('ytick', labelsize=ticklabelsize)

# set global font to be Helvetica
current_dir = os.path.dirname(__file__)
helvetica_path = os.path.join(current_dir, 'fonts', 'Helvetica.ttf')
font_prop = fm.FontProperties(fname=helvetica_path)
print("Loaded font name:", font_prop.get_name())
fm.fontManager.addfont(helvetica_path)
rc('font', family=font_prop.get_name())