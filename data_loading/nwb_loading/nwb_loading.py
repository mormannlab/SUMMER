from pathlib import Path
import numpy as np
from pynwb import NWBHDF5IO
import sys

from .binning import *

data_dir = "/storage/user/gerkenf/NWB_DATA/finalized_localizations"

def load_nwb(patient, data_dir_override=None):
    """
    Load NWB data for a given patient.

    Parameters:
        patient (str): The patient ID.
        data_dir_override (str, optional): Directory containing NWB files. If None, uses module default data_dir.

    Returns:
        tuple: A tuple containing the loaded NWB file and the path to the NWB file.
    """
    base = Path(data_dir_override) if data_dir_override is not None else Path(data_dir)
    nwb_file = base / f"sub{patient}.nwb"
    
    if not nwb_file.exists():
        raise FileNotFoundError(f"NWB file for patient {patient} not found at {nwb_file}")
    
    io = NWBHDF5IO(nwb_file, mode="r")
    nwb_data = io.read()
    
    return nwb_data, nwb_file

def get_movie_edges(nwb_data, bin_length):
    """
    Get movie edges from the NWB data for the requested bin length (ms).
    Uses 40ms edges as base; for larger bin_length subsamples so each bin spans
    bin_length (e.g. 80ms -> every 2nd 40ms edge). Stored 100ms edges are used
    when bin_length is 100.

    Parameters:
        nwb_data: Loaded NWB object.
        bin_length (float): Bin length in ms (e.g. 40, 80, 100).

    Returns:
        np.ndarray: Movie edges for histogram binning.
    """
    df_nwb = nwb_data.stimulus['movie_binning_info'].to_dataframe()
    edges_40 = df_nwb.loc[df_nwb['bin_length'] == 40]['edges'].values[0]
    if isinstance(edges_40, np.ndarray):
        pass
    else:
        edges_40 = np.asarray(edges_40)

    if bin_length < 40:
        # Subdivision is handled in binning.py
        movie_edges = edges_40
    elif bin_length == 40:
        movie_edges = edges_40
    elif bin_length == 100:
        movie_edges = df_nwb.loc[df_nwb['bin_length'] == 100]['edges'].values[0]
        if not isinstance(movie_edges, np.ndarray):
            movie_edges = np.asarray(movie_edges)
    else:
        # 80, 120, etc.: subsample 40ms edges so each bin spans bin_length ms
        # 80ms -> step 2, 120ms -> step 3
        if bin_length % 40 != 0:
            raise ValueError(
                f"bin_length must be a multiple of 40 (ms); got {bin_length}. "
                "Supported: 40, 80, 100, 120, ..."
            )
        step = int(bin_length // 40)
        movie_edges = edges_40[::step]
    print(f"Movie edges: {movie_edges.shape} (bin_length={bin_length} ms)")
    return movie_edges

def get_patient_rec(nwb_data):
    """
    Get patient recording information from the NWB data.
    
    Parameters:
        df_nwb (DataFrame): DataFrame containing NWB data.
        
    Returns:
        list: List of patient recording information.
    """
    df_nwb = nwb_data.stimulus['cleaned_watchlogs'].to_dataframe()
    patient_rec = df_nwb['neural_recording_time'].values
    
    return patient_rec

def get_patient_pts(nwb_data):
    """
    Get patient points from the NWB data.
    
    Parameters:
        df_nwb (DataFrame): DataFrame containing NWB data.
        
    Returns:
        list: List of patient points.
    """
    df_nwb = nwb_data.stimulus['cleaned_watchlogs'].to_dataframe()
    patient_pts = df_nwb['pts'].values
    
    return patient_pts

def get_binned_spikes_nwb(patient, units, bin_length, data_dir_override=None):
    """
    Get relevant data for binning spikes.

    Parameters:
        patient (str): The patient ID.
        units (list): List of unit IDs to bin spikes for.
        bin_length (float): Bin length in ms.
        data_dir_override (str, optional): Directory containing NWB files. If None, uses module default.

    Returns:
        np.ndarray: Binned spike data, shape (n_units, n_bins).
    """
    # Load nwb data file
    nwb_data, nwb_file = load_nwb(patient, data_dir_override=data_dir_override)

    # get movie edges
    movie_edges = get_movie_edges(nwb_data, bin_length)

    # get patient rec
    patient_rec = get_patient_rec(nwb_data)

    # get patient pts
    patient_pts = get_patient_pts(nwb_data)

    # get spike times
    binned_spikes_patient = bin_spikes_for_patient_with_movie_edges(
        nwb_data = nwb_data,
        patient_id=patient,
        units=units,    
        session_nr=1, 
        bin_length=bin_length,
        movie_edges=movie_edges,
        patient_rec=patient_rec,
        patient_pts=patient_pts
    )
    
    return binned_spikes_patient

def get_unit_ids_for_patient_nwb(patient, brain_regions=None, data_dir_override=None):
    # Load nwb data file
    nwb_data, nwb_file = load_nwb(patient, data_dir_override=data_dir_override)
    df_units = nwb_data.units.to_dataframe()

    # load units for patient
    if brain_regions is None:
        unit_ids = list(range(len(df_units)))
        unit_ids = np.asarray(unit_ids)
    else:
        subregions={'A': ['A'], 'H': ['AH', 'MH', 'PH'], 'EC': ['EC'], 'PHC': ['PHC'], 'PIC': ['PIC']}
        res=[]
        print(f"Brain Region: {brain_regions}")
        for brain_region in brain_regions:
            brain_region_neurons=[]
            abbrev = subregions[brain_region]
            print(f"Abbrev to load: {abbrev}")
            for abb in abbrev:
                subregion_neurons = df_units[df_units['brain_region'] == brain_region].index.tolist()
                brain_region_neurons += subregion_neurons
            res += brain_region_neurons
            print(f"Patient {patient}: {brain_region_neurons}")
        unit_ids = np.asarray(res)

    return unit_ids
