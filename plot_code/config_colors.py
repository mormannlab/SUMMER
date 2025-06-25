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
    "PIC": '#fdbf6f',
    "A": '#E53E24', 
    "H": '#54aead', 
    "EC": '#FFFF33', 
    "PHC": '#bfe5a0', 
    "FF": '#0A2463',
    "LG": '#3B5738',
    "PRC": '#6E1A0D',
}

cmap = sns.color_palette("Spectral", as_cmap=True)