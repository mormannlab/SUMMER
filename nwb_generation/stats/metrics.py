#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality metrics for assessing spike sorting quality.

Unit type (interneuron vs pyramidal neurons) calculated in cell_type.py.

Author: Alana Darcher, Uniklinikum Bonn
Date: 25 Feb 2025
License: MIT License
"""

import numpy as np
from scipy.stats import chi2


def calc_cv2(times):
    """Calculate the modified coefficient of variation given a set of spike times. 

    Taken from:
    Holt, G. R., Softky, W. R., Koch, C., & Douglas, R. J. (1996). 
    Comparison of discharge variability in vitro and in vivo in cat visual cortex neurons. 
    Journal of neurophysiology, 75(5), 1806–1814. https://doi.org/10.1152/jn.1996.75.5.1806

    Args:
        times (array_like): sorted list of spike times from a single neuron

    Returns:
        float: modified CV2 value of spike train
    """
    isi_list = np.diff(times)

    num = np.abs(isi_list[1:] - isi_list[:-1])
    den = isi_list[1:] + isi_list[:-1]

    return 2 * np.mean(num / den)

def count_isi_violations(spike_times, t_r,):
    """
    Count the number of inter-spike intervals that violation 
    the biological refractory period. 

    Args:
        spike_times (array-like): vector of spike times, in milliseconds
        t_r (int): biological refractory period, in milliseconds (e.g. 3 ms)

    Returns:
        num_violations: int, total number of contaminated inter-spike intervals
    """
    unit_isi = np.diff(spike_times)
    num_violations = sum(unit_isi <= t_r)
    return num_violations

def calc_isi_contamination(r, N, T, t_r, t_c):
    """
    Calculate rate of contamination for a unit.
    Outputs the ratio of ISI violations in the spike times. 

    Note: not used for the dataset --- overestimates the isi violation
    rate with longer recordings.

    Based on:
    Hill, D. N., Mehta, S. B. & Kleinfeld, D. (2011)
    Quality Metrics to Accompany Spike Sorting of Extracellular Signals. 
    J. Neurosci. 31, 8699–8705.
    https://doi.org/10.1523/JNEUROSCI.0971-11.2011

    Args:
        r (int): total number of contaminated inter-spike intervals
        N (int): number of spike events in a unit, in milliseconds
        T (int): duration of the experiment, in milliseconds
        t_r (int): biological refractory period, in milliseconds (e.g. 3 ms)
        t_c (int): censored period, in milliseconds (1/f * 1000, where f is the system sampling rate in Hz)

    Returns:
        p: float, expected rate of contaminated refractory periods in the unit
    """
    p = (r * T) / (2*(t_r - t_c)*N**2)
    
    if np.isnan(p):
        p = 0
    elif p > 1:
        p = 1

    return  p 

def poisson_conf_interval(k, conf_level=0.95):
    """
    https://www.statsdirect.com/help/rates/poisson_rate_ci.htm

    ULM, K. 
    SIMPLE METHOD TO CALCULATE THE CONFIDENCE INTERVAL OF A STANDARDIZED MORTALITY RATIO (SMR). 
    American Journal of Epidemiology 131, 373–375 (1990).

    Args:
        k (int): number of events
        conf_level (float, optional): 1 - alpha. Defaults to 0.95.

    Returns:
        lower: float, lower bound of the CI
        uppoer: float, upper bound of the CI
    """
    alpha = 1 - conf_level
    lower = 0.5 * chi2.ppf(alpha / 2, 2 * k) if k > 0 else 0.0
    upper = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (k + 1))
    return lower, upper

def calculate_snr(waveform, channel_mad):
    """
    Calculate the signal-to-noise ratio between the max amplitude of the mean spike waveform
    for a given unit and the background noise of the channel (median absolute deviation).

    Note: median absolute deviation is calculated via MATLAB offline and expected to be imported and fed as an input.

    Buccino, A. P., Hurwitz, C. L., Garca, S., Magland, J., Siegle, J. H., Hurwitz R., & Hennig, M. H. (2020).
    SpikeInterface, a unified framework for spike sorting
    eLife, 9:e61834.
    https://doi.org/10.7554/eLife.61834

    Args:
        waveform (array-like): average waveform of a unit, taken across all unit spikes
        channel_mad (float): median absolute deviation of the channel from which the unit was sorted

    Returns:
        peak_snr: float, signal-to-noise ratio for the peak amp of a given unit
    """
    if np.isnan(channel_mad):
            peak_snr = np.nan
    else:
        peak_snr = np.max(waveform) / channel_mad
    return peak_snr

