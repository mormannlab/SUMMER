"""
calc_responsive_units.py

Module for calculating responsive units in neural spike data, including
functions for extracting spike times around events, binning spike trains,
computing baseline and stimulus activity, and performing statistical tests
such as Wilcoxon signed-rank. 

Same set of functions as used in decoding paper. 

Author: Alana Darcher
Date: 2025-09-16

Dependencies:
    numpy
    pandas
    scipy

Usage:
    Import and use the provided functions to process spike/event data and
    assess responsiveness of neural units.

"""

import warnings
import numpy as np
import pandas as pd
import tqdm
from scipy.stats import wilcoxon


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



def interleaved_bins_fmormann(spiketimes, eventtimes):
    """
    produce interleaved bins for activity from one stimulus
    """
    bins_0 = np.linspace(0, 1000, 11)
    bins_1 = np.linspace(50, 950, 10)

    timelist = times_by_event(spiketimes, eventtimes, 0, 1000)

    hist_0 = times_to_histogram(timelist, bins_0)
    hist_1 = times_to_histogram(timelist, bins_1)

    interleaved = np.zeros((len(eventtimes), len(bins_0) + len(bins_1) - 2),
                           int)

    interleaved[:, ::2] = hist_0   # places each value of hist_0 as every other value of interleaved 
    interleaved[:, 1::2] = hist_1

    return interleaved, bins_0, bins_1


#### Baseline functions
def baseline_signed_rank(spikes, times, start=-500, stop=0):
    """
    Written for performing signed-rank search over mini-screening results.
    Args:
        spikes:
        times:
        start:
        stop:

    Returns:

    """
    ## format baseline data - note: doing this differently than for existing scr stats, since paired

    # to start with -- pulling the activity from the 500 msec preceeding stim onset
    # and getting the average activity for each trial during the baseline period
    # In the end, each bin slice in the stimulus period is being compared to
    # the same set of values, for a given stimulus/session

    bins = np.arange(start, stop + 1, 100)
    base_hists = []
    for i, image_times in enumerate(times):
        eventtimes = times_by_event(spikes, image_times, start, stop)
        hist = np.array(times_to_histogram(eventtimes, bins))
        base_hists.append(hist)

    base_dist = np.array(base_hists).reshape(50, 5)
    base_hz = np.sum(base_dist, axis=1) * 2  # taking sum of spikes during a trial, converting to Hz

    return base_hz, base_dist

def baseline_fmormann(hist, bins, start=-500, stop=0):
    """
    calculate the baseline firing rate by taking the mean of defined bins

    inputs:
        hist: array, histogrammed/binned data produced by times_to_histogram
        bins: array, bin edgs, also produced by times_to_histogram 
        start: optional int, time prior to event from which to calculate baseline
        stop: optional int, end time for calculating baseline period

    """
    relevant_bins = (bins >= start) & (bins < stop)
    distribution = hist[:, relevant_bins[:-1]]

    return distribution.mean(1), distribution

def pool_stimulus_wrapper(spikes, times, start=0, stop=1000):
    """
    format stimulus-period activity for pooling across stim images
    Written for performing signed-rank search over mini-screening results.
    """

    # interleave the activity for each stim image separately
    stim_hists = []
    for i, image_times in enumerate(times):
        interleaved, bins_0, bins_1 = interleaved_bins_fmormann(spikes, image_times)
        # convert to Hz
        duration_bin = (bins_0[1] - bins_0[0])
        interleaved = interleaved * 1000 / duration_bin

        stim_hists.append(interleaved)

    stim_hists = np.array(stim_hists).reshape(10*len(stim_hists),len(stim_hists[0][0]))
    return stim_hists



def pvalue_binwise_signedrank(spiketimes, eventtimes, tmin_baseline, tmax_baseline, restriction="increase", debug=False):
    """
    compare each bin of the response period to a baseline bin
    """
    duration_baseline = tmax_baseline - tmin_baseline

    interleaved, bins_0, bins_1 = interleaved_bins_fmormann(spiketimes,
                                                            eventtimes)
    
    n_bins = interleaved.shape[1]
    duration_bin = (bins_0[1] - bins_0[0])
    interleaved = interleaved*1000/duration_bin
    
    timelist = times_by_event(spiketimes, eventtimes, tmin_baseline, tmax_baseline)
    baseline_hist = times_to_histogram(timelist, [tmin_baseline, tmax_baseline])
    
    # here we have to re-normalize
    baseline_hist = 1000*baseline_hist.ravel()/duration_baseline

    atrials = interleaved.any(1).sum()
    pvals = np.ones(n_bins)
    directions = np.zeros(n_bins, dtype=bool)
    
    if debug == True:
        print("interleaved: ", interleaved)
        print("bin0: ", bins_0)
        print("bin1: ", bins_1)
        print("active trials: ", atrials)


    if atrials > interleaved.shape[0]/3:
        baseline_sum = baseline_hist.sum()
        for i in range(n_bins):
            try:
                #print("Interleaved up to bin {}: {}".format(i, interleaved[:, i]))
                #print("Baseline:                 {}".format(baseline_hist))
                if (baseline_hist - interleaved[:, i]).any():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _, pval = wilcoxon(interleaved[:, i], baseline_hist)
                if not (baseline_hist - interleaved[:, i]).any():
                    pval = -1
                
                if restriction == "increase":
                    directions[i] = interleaved[:, i].sum() >= baseline_sum
                elif restriction == "decrease":
                    directions[i] = interleaved[:, i].sum() < baseline_sum
                else:
                    directions = np.ones(n_bins, dtype=bool)

            except ValueError as error:
                raise error

            pvals[i] = pval

        pvals[~directions] = 1

        pvals.sort()

        pvals *= n_bins/np.arange(1, n_bins + 1)
    abs_pval = np.abs(pvals)
    return atrials, abs_pval.min()

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
                       key=1, restriction="increase", alpha=0.001):
    
    total = len(df_unit_data)

    # iterate over the units and calculate the response statistic
    bins = np.arange(baseline_time*-1, stimulus_time + 1, 100)
    reg_act = np.zeros((total, len(bins)-1))
    reg_pvals = np.ones(total)

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
            unit_id = row.unit_id
            unit_spikes = row.spike_times
        
            event_spikes = times_by_event(unit_spikes, onsets, baseline_time*-1, stimulus_time)

            # bin
            bins = np.arange(baseline_time*-1, stimulus_time + 1, 100)
            binned_act = times_to_histogram(event_spikes, bins)

            # stats
            #baseline_rate, distribution = baseline_fmormann(binned_act, bins, start=baseline_time*-1)
            #atrials, pval = pvalue_fmormann(unit_spikes, onsets, distribution)
            atrials, pval = pvalue_binwise_signedrank(spiketimes=unit_spikes, eventtimes=onsets, 
                                                      tmin_baseline=-1*baseline_time, tmax_baseline=0, restriction=restriction, debug=False)

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
    waveform = specific_unit_data["waveforms"].iloc[0]

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
    return act_z, bins, binned_act, spike_times, event_spikes, waveform, onsets