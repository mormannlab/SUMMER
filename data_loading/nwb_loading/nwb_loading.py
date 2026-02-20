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
    Reads edges directly from movie_binning_info for that bin_length.

    Parameters:
        nwb_data: Loaded NWB object.
        bin_length (float): Bin length in ms (e.g. 40, 80, 200, 480, 1000).

    Returns:
        np.ndarray: Movie edges for histogram binning.
    """
    df_nwb = nwb_data.processing["movie_binning"].data_interfaces["movie_binning_info"].to_dataframe()
    matches = df_nwb.loc[df_nwb["bin_length"] == bin_length]
    if len(matches) == 0:
        available = df_nwb["bin_length"].unique().tolist()
        raise ValueError(
            f"bin_length {bin_length} not found in movie_binning_info. "
            f"Available: {available}"
        )
    movie_edges = matches["edges"].values[0]
    if not isinstance(movie_edges, np.ndarray):
        movie_edges = np.asarray(movie_edges)
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

# Bin lengths for which frame numbers exist in movie_binning (one frame per bin)
MOVIE_BINNING_BIN_LENGTHS = (40, 80, 200, 480, 1000)


def _frame_string_to_number(frame_str):
    """Extract 1-based frame number from string like 'frame_0001.png' -> 1."""
    # Format: something_<digits>.png -> int(digits)
    part = frame_str.split("_")[1][:-4]
    return int(part)


def _get_movie_frames_for_bin_length(nwb_data, bin_length):
    """
    Get the array of frame identifiers for the given bin_length from movie_binning.
    Returns (frame_numbers, n_bins) where frame_numbers are 1-based, or (None, None) if not available.
    """
    if "movie_binning" not in nwb_data.processing:
        return None, None
    try:
        iface = nwb_data.processing["movie_binning"].data_interfaces["movie_binning_info"]
    except KeyError:
        return None, None
    df = iface.to_dataframe()
    # Prefer exact match; bin_length in file may be int (40, 80, ...)
    matches = df.loc[df["bin_length"] == bin_length]
    if len(matches) == 0:
        return None, None
    frames = matches["frames"].values[0]
    if hasattr(frames, "__len__") and not isinstance(frames, (str, bytes)):
        frames = np.asarray(frames).ravel()
    else:
        frames = np.asarray([frames])
    # Parse to 1-based frame numbers
    frame_numbers = np.array([_frame_string_to_number(str(f)) for f in frames], dtype=np.int64)
    return frame_numbers, len(frame_numbers)


def get_annotation_labels_nwb(patient, label_name, bin_length, data_dir_override=None):
    """
    Load annotation indicator function for a label and align to bin indices.

    When movie_binning provides frame numbers for this bin_length (40, 80, 200, 480, 1000 ms),
    each bin is mapped to its corresponding frame; the indicator is indexed by frame number
    (frames are 1-based, so indicator index = frame_number - 1). This yields one label per bin
    with the same length as the binned spike vector. If frame mapping is not available,
    falls back to proportional resampling.

    Parameters:
        patient (str): The patient ID.
        label_name (str): Label to load (e.g. "summer", "alison").
        bin_length (float): Bin length in ms (must match the one used for binned spikes).
        data_dir_override (str, optional): Directory containing NWB files.

    Returns:
        tuple: (labels, n_classes) where labels is np.ndarray of shape (n_bins,) with
               integer class indices in [0, n_classes-1], and n_classes is the number
               of unique classes.
    """
    nwb_data, _ = load_nwb(patient, data_dir_override=data_dir_override)
    movie_edges = get_movie_edges(nwb_data, bin_length)
    n_bins = len(movie_edges) - 1

    if "movie_annotations_indicator_functions" not in nwb_data.processing:
        raise KeyError(
            "NWB file has no processing['movie_annotations_indicator_functions']; "
            "cannot load annotation labels."
        )
    iface = nwb_data.processing["movie_annotations_indicator_functions"].data_interfaces[
        "movie_annotations_indicator_functions"
    ]
    df = iface.to_dataframe()
    rows = df[df["label_name"] == label_name]
    if len(rows) == 0:
        raise ValueError(
            f"Label '{label_name}' not found in movie_annotations_indicator_functions. "
            f"Available: {df['label_name'].unique().tolist()}"
        )
    indicator = rows["indicator_function"].values[0]
    if hasattr(indicator, "__len__") and not isinstance(indicator, (str, bytes)):
        indicator = np.asarray(indicator).ravel()
    else:
        indicator = np.asarray([indicator])

    # Prefer frame-based indexing when movie_binning has frames for this bin_length
    frame_numbers, n_frames = _get_movie_frames_for_bin_length(nwb_data, bin_length)
    if frame_numbers is not None and n_frames == n_bins:
        # Indicator is indexed by frame (1-based) -> use index = frame_number - 1
        # Clip to valid range in case of off-by-one or extra frames
        ind_max = len(indicator) - 1
        indices_0based = np.clip(frame_numbers - 1, 0, ind_max)
        labels_raw = indicator[indices_0based]
    else:
        # Fallback: proportional resampling so output length matches n_bins
        n_src = len(indicator)
        idx = np.clip(
            np.round(np.linspace(0, n_src - 1, n_bins)).astype(int), 0, n_src - 1
        )
        labels_raw = indicator[idx]

    # Map to integer class indices 0, 1, ...
    unique_vals = np.unique(labels_raw)
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    labels = np.array([val_to_idx[v] for v in labels_raw], dtype=np.int64)
    n_classes = len(unique_vals)
    return labels, n_classes


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
