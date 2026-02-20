from pathlib import Path
import numpy as np
from pynwb import NWBHDF5IO
import sys

data_dir = "/data"

patient = 29
nwb_file = Path(data_dir) / f"sub{patient}.nwb"
    
if not nwb_file.exists():
    raise FileNotFoundError(f"NWB file for patient {patient} not found at {nwb_file}")

io = NWBHDF5IO(nwb_file, mode="r")
nwb_data = io.read()
print(f"NWB data: {nwb_data}")

# extract labels from nwb_data
label = nwb_data.stimulus["annotations_base"].to_dataframe()
print(f"Labels: {label}")

movie_annotations_nwb = nwb_data.processing["movie_annotations_indicator_functions"].data_interfaces["movie_annotations_indicator_functions"].to_dataframe()
# extract only the one for summer
summer_annotations = movie_annotations_nwb[movie_annotations_nwb["label_name"] == "summer"]
print(f"Summer annotations: {summer_annotations}")
print(f"Type of summer annotations: {type(summer_annotations)}")
# extract indicator function as numpy
summer_indicator_function = summer_annotations["indicator_function"].values[0]
print(f"Summer indicator function: {summer_indicator_function}")
print(f"Length of summer indicator function: {len(summer_indicator_function)}")

# get movie edges
# Accessing the "movie_binning" processing module and extracting its DataInterface/DataFrame
movie_binning_df = nwb_data.processing["movie_binning"].data_interfaces["movie_binning_info"].to_dataframe()
print(f"Movie edges: {movie_binning_df}")
print(f"Length of movie edges: {len(movie_binning_df)}")
# extract the frames column as numpy vector for the bin length 80 ms
movie_frames = movie_binning_df.loc[movie_binning_df["bin_length"] == 80]["frames"].values[0]
print(f"Movie frames: {movie_frames}")
print(f"Type of movie frames: {movie_frames[:10]}")
print(f"Length of movie frames: {len(movie_frames)}")

# extract the frame number as int from the string
movie_frames_int = [int(frame.split("_")[1][:-4]) for frame in movie_frames]
print(f"Movie frames int: {movie_frames_int[:10]}")

