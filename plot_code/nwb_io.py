#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for parsing NWB-formatted data.
Copied from dhv_decoding/preprocessing/nwb_io.py

Author: Alana Darcher, Uniklinikum Bonn
Date: 15 May 2025
"""
import os
import sys
import pickle
from pathlib import Path

path_location = os.getcwd()
path_base = Path(*Path(path_location).parts[:-2])
sys.path.append(path_base)

from tqdm import tqdm
import numpy as np
import pandas as pd

from pynwb import NWBHDF5IO

REGION_ALTERNATIVE_NAMES = {"PHC": ["PHC", "APH", "PPH", "MPH"],
                            "A": ["A"],
                            "EC": ["EC"],
                            "PIC": ["PIC"],
                            "H": ["H", "AH", "MH", "PH"]} 

#####
# Collection within patients
#####
def get_patient_spiking_activity(data_dir, patient_id, exclude_waveforms=False):
    """
    Extracts all spiking activity from a given patient's NWB file.

    Args:
        data_dir (str or Path): Directory where the NWB files are stored.
        patient_id (int): Patient ID to extract data from.
        exclude_waveforms (bool, optional): whether to include waveforms from return. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing all spike activity information 
    """

    path = Path(data_dir , f"sub{int(patient_id)}.nwb")
    io = NWBHDF5IO(path, mode="r")
    nwbfile = io.read()
    df_units = nwbfile.units.to_dataframe()
    df_units["unit_id"] = np.arange(0, len(df_units))
    df_units["patient_id"] = [int(patient_id)] * len(df_units)

    if exclude_waveforms:
        df_units.drop(columns="waveforms", inplace=True)
    
    io.close()
    
    return df_units

def get_patient_unit(df_units, unit_id, exclude_waveforms=True, ):
    """
    Extracts spike times (and optionally waveforms) for a specific unit of a recorded session.

    Args:
        df_units (pandas.DataFrame): output of get_patient_spiking_activity, contains all spike times and meta info
        unit_id (int): Unit index number to pull activity for. 
        exclude_waveforms (bool, optional): whether to exclude waveforms from return. Defaults to True.

    Returns:
        np.ndarray: Array of spike times.
        np.ndarray, optional: Array of waveforms (if exclude_waveforms is False).
    """
    
    restricted_df = df_units[df_units["unit_id"] == unit_id]
    spike_times = restricted_df["spike_times"].to_numpy()[0]
    
    if exclude_waveforms is False:
        waveforms = restricted_df["waveforms"].to_numpy()[0]
        return np.array(spike_times), np.array(waveforms)
    else:
        return spike_times

def get_patient_aligned_annotations(data_dir, patient_id, ):

    path = Path(data_dir, f"sub{int(patient_id)}.nwb")
    io = NWBHDF5IO(path, mode="r")
    nwbfile = io.read()

    df_a = nwbfile.stimulus["annotations_patient_aligned"].to_dataframe()
    df_a["patient_id"] = [int(patient_id)] * len(df_a)

    return df_a

def get_aligned_annotation(data_dir, patient_id, label_name, key):

    df_a = get_patient_aligned_annotations(data_dir, patient_id)
    df_label = df_a[df_a["label_name"] == label_name]
    starts = np.array(df_label[df_label["value"] == key]["start_time"])
    stops = np.array(df_label[df_label["value"] == key]["stop_time"])

    return starts, stops

def load_patient(data_dir, patient_id,):
    path = Path(data_dir, f"sub{int(patient_id)}.nwb")
    io = NWBHDF5IO(path, mode="r")
    nwbfile = io.read()
    return nwbfile

#####
# Collection across patients
#####

def collect_all_spike_data_in_region(data_dir, region, region_name_dict=REGION_ALTERNATIVE_NAMES, drop_waveforms=True):
    """
    Collect unitwise data from NWB files for a given region.

    Args:
        data_dir (pathlib object, string-like): path to the parent directory containing nwb files
        region (string): region abbreviation (e.g., 'PHC' for parahippocampal cortex)
        region_name_dict (dict, optional): Defaults to REGION_ALTERNATIVE_NAMES.

    Returns:
        pandas DataFrames: DataFrame containing collected unit data
    """
    i = 0 

    for path in tqdm(list(data_dir.glob("*.nwb"))):
        print(path)
        if path.is_dir():
            continue 

        patient_id = int(path.name.split(".")[0][3:])
        print(f"  {patient_id}")
        io = NWBHDF5IO(path, mode="r")
        nwbfile = io.read()
        df_units = nwbfile.units.to_dataframe()
        df_units["unit_id"] = np.arange(0, len(df_units))
        df_ = df_units[df_units["brain_region"].isin(region_name_dict[region])].copy() 
        df_["patient_id"] = np.array([int(patient_id)] * len(df_), dtype=np.int8)
        
        if drop_waveforms:
            df_.drop(columns="waveforms", inplace=True)

        if i == 0:
            df_region_restricted_units = df_.copy()
        else:
            df_region_restricted_units = pd.concat([df_region_restricted_units, df_], ignore_index=True)
        
        io.close()
        i += 1
        
    return df_region_restricted_units

def collect_psth_data(data_dir, region, label, region_name_dict=REGION_ALTERNATIVE_NAMES, drop_waveforms=True):
    """
    Collect unitwise data from NWB files for a given region, along with the annotation 
    information for a given label. Formatted for use in follow-up single-unit analyses. 

    Args:
        data_dir (pathlib object, string-like): path to the parent directory containing nwb files
        region (string): region abbreviation (e.g., 'PHC' for parahippocampal cortex)
        label (string): annotation name, e.g. "tom" or "camera-cuts"
        region_name_dict (dict, optional): Defaults to REGION_ALTERNATIVE_NAMES.

    Returns:
        pandas DataFrames: DataFrame containing collected unit data, DataFrame containing collected annotation information
    """
    i = 0
    
    for path in tqdm(list(data_dir.glob("*.nwb"))):
        print(path)
        if path.is_dir():
            continue 

        patient_id = int(path.name.split(".")[0][3:])
        print(f"  {patient_id}")
        io = NWBHDF5IO(path, mode="r")
        nwbfile = io.read()
        df_units = nwbfile.units.to_dataframe()
        df_units["unit_id"] = np.arange(0, len(df_units))
        df_ = df_units[df_units["brain_region"].isin(region_name_dict[region])].copy() # TODO expand the function call to just accept a list of units, potentially more flexible for indexing beyond this pipeline.
        df_["patient_id"] = np.array([int(patient_id)] * len(df_), dtype=np.int8)
        
        if drop_waveforms:
            df_.drop(columns="waveforms", inplace=True)

        if i == 0:
            df_region_restricted_units = df_.copy()
        else:
            df_region_restricted_units = pd.concat([df_region_restricted_units, df_], ignore_index=True)

        df_a = nwbfile.stimulus["annotations_patient_aligned"].to_dataframe()
        df_a = df_a[df_a["label_name"] == label]
        assert len(df_a) > 0, f"No entries for label '{label}' found."
        df_a["patient_id"] = [int(patient_id)] * len(df_a)

        if i == 0:
            df_annotation = df_a.copy()
        else:
            df_annotation = pd.concat([df_annotation, df_a], ignore_index=True)

        io.close()
        i += 1
        
    return df_region_restricted_units, df_annotation

def parse_annotation_df(df_annotation, patient_id, key,):
    """
    Filters the annotation DataFrame by patient and returns the 
    corresponding start and stop times for each annotation occurrence.

    Args:
        df_annotation (pandas DataFrame): annotation information returned by function, collect_data
        patient_id (int): patient id number of interest
        key (int): indicated whether to take onsets of a labeled entity (1) or offsets (0)

    Returns:
        numpy arrays: start times of label of interest, stop times of label of interest
    """
    df_patient = df_annotation[df_annotation['patient_id'] == patient_id].reset_index()
    starts = np.array(df_patient[df_patient["value"] == key]["start_time"])
    stops = np.array(df_patient[df_patient["value"] == key]["stop_time"])
    return starts, stops

def save_regional_unit_info(save_dir, df_region_restricted_units, region):
    """
    Saves regional unit information to a serialized file.

    Args:
        save_dir (str or Path): Directory where the unit information file will be saved.
        df_region_restricted_units (pd.DataFrame): DataFrame containing unit information. Must contain 'unit_id' and 'patient_id' columns.
        region (str): The brain region associated with the units.

    Returns:
        None
    """
    path_save = Path(save_dir, "unit_look_up")
    path_save.mkdir(exist_ok=True)

    unit_ids = df_region_restricted_units["unit_id"]
    patient_ids = df_region_restricted_units["patient_id"]

    region_index = list(zip(patient_ids, unit_ids))
    filehandler = open(path_save / f"units_in_{region}.obj","wb")
    pickle.dump(region_index, filehandler)
    filehandler.close()

def open_regional_unit_info(save_dir, region):
    """
    Loads regional unit information from a serialized file.

    Args:
        save_dir (str or Path): Directory where the unit information file is located.
        region (str): The brain region associated with the units.

    Returns:
        list[tuple]: A list of tuples containing patient IDs and unit IDs.
    """
    path_save = Path(save_dir, "unit_look_up", f"units_in_{region}.obj")
    file = open(path_save,'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file