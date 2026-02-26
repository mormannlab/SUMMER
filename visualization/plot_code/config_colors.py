#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Color specification for plots.

https://coolors.co/9c2338-f69d5a-f3f3a6-9bd4a1-486198


"""
import seaborn as sns

colors = [
"#9c2338", "#f69d5a",  "#f3f3a6", "#9bd4a1", "#486198",
]

color_by_region = {
    "A": '#E53E24', 
    "H": '#54aead', 
    "EC": "#D8D806", 
    "PHC": '#60A02C', 
    "Other": '#FCAF4A',
}

cmap = sns.color_palette("Spectral", as_cmap=True)