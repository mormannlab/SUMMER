"""
calc_responsive_units.py

Module for calculating responsive units in neural spike data, including
functions for extracting spike times around events, binning spike trains,
computing baseline and stimulus activity, and applying a ZETA test to each
unit.

Author: Alana Darcher
Date: 2025-09-16

Dependencies:
    numpy
    pandas
    zetapy
    tqdm

Usage:
    Import and use the provided functions to process spike/event data and
    assess responsiveness of neural units.

"""

import numpy as np
import pandas as pd
import tqdm
from zetapy import zetatest

num_permutations = 1000

EMPTY = np.array([])

def times_by_event(spiketimes, eventtimes, pre, post):
    """
    Find spiketimes surrounding a given event time as specified by 
    pre and post, for all included event times. 
    
    Returns list of spikes occurring pre-msec before and post-msec after 
    each event, for each event. Spikes are re-scaled such that their magnitude
    indicates distance from the event, and sign indicates if the spike occurred 
    pre/post stimulus. 
    
    inputs:
    spiketimes: array-like, spike times for one unit. 
    eventtimes: array-like, all event times from a screening 
        - each entry corresponds to one stimulus presentation 
    pre: int, period prior to event from which to collect spikes (milliseconds)
    post: int, period post event from which to collect spikes (milliseconds)

    outputs:
    ret: array, list of spiketimes corresponding 
    """
    ret = []

    for i_event, etime in enumerate(eventtimes):
        idx = (spiketimes >= etime + pre) & (spiketimes <= etime + post)
        if not idx.any():
            ret.append(EMPTY)
        else:
            ret.append(spiketimes[idx] - etime)

    return ret

def times_to_histogram(timelist, bins):
    """
    convert a list of spiketrains to a histogram.
    Note: bins must contain the rightmost edge, too.
    
    creates an array of size (length timelist, nr. of bins)
    - row-wise/axis 0: event
    - column-wise/axis 1: binned spike times 
    
    """
    ret = np.zeros((len(timelist), len(bins) - 1), int)
    for i_time, time in enumerate(timelist):
        ret[i_time, :] = np.histogram(time, bins)[0]

    return ret

def zscore(data, base=None):
    """
    z-score a vector of activity using its own mean + std

    Args:
        data (_type_): _description_
        base (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    act_z = np.zeros((data.shape))
    
    if base is not None:
        u = np.mean(base)
        sd = np.mean(base)
    for i, row in enumerate(data):
        
        if base is None:
            u = np.mean(row)
            sd = np.std(row)
        if not row.any():
            continue
        else:
            act_z[i,:] = (row - u) / sd
    
    return act_z

def df_formatter(reg_pvals, reg_act, alpha=0.001):
    """
    format activity and pvalues into a pandas dataframe format. 

    Args:
        reg_pvals (list): list of pvalues
        reg_act (list): list of activity vectors
    """

    bin_id = []
    val = []
    pval_cat = []

    for i, pval in enumerate(reg_pvals):
        act = reg_act[i, :]
        if pval > alpha: 
            cat = "ns"
        elif pval <= alpha:
            cat = "*"

        for k, bin_ in enumerate(act):
            bin_id.append(k)
            val.append(bin_)
            pval_cat.append(cat)

    lineplot_df = pd.DataFrame({"bin": bin_id, "FR": val, "sig": pval_cat})
    return lineplot_df

def identify_responses(df_unit_data, df_annotation, pat_subset, 
                       baseline_time=1000, stimulus_time=1000, 
                       key=1, alpha=0.001):
    
    total = len(df_unit_data)
    print("running zeta..")
    # iterate over the units and calculate the response statistic
    bins = np.arange(baseline_time*-1, stimulus_time + 1, 100)
    reg_act = np.zeros((total, len(bins)-1))
    reg_pvals = np.ones(total)

    ind = 0
    for patient_id in tqdm.tqdm(pat_subset):


        print(patient_id)
        
        # restrict the annotations to that of the patient
        df_annotations_patient = df_annotation[df_annotation["patient_id"]==patient_id]
        starts = df_annotations_patient["start_time"].to_numpy()
        stops = df_annotations_patient["stop_time"].to_numpy()
        values = df_annotations_patient["value"].to_numpy()

        starts = starts[values==key]
        stops = stops[values==key]
        # process the annotations to remove overlapping labels in the baseline 

        stim_onset_times = []
        for i in range(1, len(starts)):
            on = starts[i]   
            prev_off = stops[i-1]
            baseline_start = on - baseline_time

            if baseline_start > prev_off:
                stim_onset_times.append(on)

        onsets = stim_onset_times

        df_patient_data = df_unit_data[df_unit_data["patient_id"]==patient_id]

        for i, row in df_patient_data.iterrows():
            unit_spikes = row.spike_times
        
            event_spikes = times_by_event(unit_spikes, onsets, baseline_time*-1, stimulus_time)
            # bin
            bins = np.arange(baseline_time*-1, stimulus_time + 1, 100)
            binned_act = times_to_histogram(event_spikes, bins)
            
            # stats
            s = np.array(unit_spikes) / 1000
            e = np.array(onsets) / 1000
            
            z = zetatest(s, e, 1, intResampNum=num_permutations)
            pval = z[0]
            # norm -- currently norming to whole "trial", not baseline
            act_z = zscore(binned_act)

            # reduce to 2D vector
            act_u = np.mean(act_z, axis=0)
            reg_act[ind, :] = act_u
            reg_pvals[ind] = pval
            ind += 1

    sort_inds = np.argsort(reg_pvals)

    ct_sig_units = reg_pvals < alpha

    start_ns = np.where(reg_pvals[sort_inds] > alpha)[0][0]
    end_ns = np.where(reg_pvals[sort_inds] > alpha)[0][-1]

    return reg_act, reg_pvals, sort_inds, ct_sig_units, start_ns, end_ns

def grab_unit_response(unit_rank=None, patient_id=None, unit_id=None, 
                      df=None, df_unit_data=None, df_annotation=None, 
                      key=1, baseline_time=1000, stimulus_time=1000):
    """
    """
    df_reindexed = df.reset_index()
    
    if (patient_id is None) and (unit_id is None):
        
        patient_id = int(df_reindexed.iloc[unit_rank]["patient_id"])
        unit_id = int(df_reindexed.iloc[unit_rank]["unit_id"])

        print(patient_id, unit_id, unit_rank)
    
    elif unit_rank is None:
        unit_rank = df_reindexed.query(f"patient_id == {patient_id} and unit_id == {unit_id}").index[0]
        print(patient_id, unit_id, unit_rank)
    else:
        raise ValueError("Either provide unit_rank or patient_id and unit_id")

    specific_unit_data = df_unit_data[(df_unit_data["patient_id"] == patient_id) & (df_unit_data["unit_id"] == unit_id)]
    spike_times = specific_unit_data["spike_times"].iloc[0]
    waveform_mean = specific_unit_data["waveform_mean"].iloc[0]
    waveform_sem = specific_unit_data["waveform_sem"].iloc[0]

    df_annotations_patient = df_annotation[df_annotation["patient_id"]==patient_id]
    starts = df_annotations_patient["start_time"].to_numpy()
    stops = df_annotations_patient["stop_time"].to_numpy()
    values = df_annotations_patient["value"].to_numpy()

    starts = starts[values==key]
    stops = stops[values==key]
    values = values[values==key]

    stim_onset_times = []
    for i in range(1, len(starts)):
        on = starts[i]   
        prev_off = stops[i-1]
        baseline_start = on - baseline_time

        if baseline_start > prev_off:
            stim_onset_times.append(on)

    onsets = stim_onset_times

    event_spikes = times_by_event(spike_times, onsets, baseline_time*-1, stimulus_time)

    bins = np.arange(baseline_time*-1, stimulus_time + 1, 100)
    binned_act = times_to_histogram(event_spikes, bins)

    # norm -- currently norming to whole "trial", not baseline
    act_z = zscore(binned_act)
    return act_z, bins, binned_act, spike_times, event_spikes, waveform_mean, waveform_sem, onsets

def flatten(xss):
    return [x for xs in xss for x in xs]