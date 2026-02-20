from pathlib import Path
import numpy as np
from pynwb import NWBHDF5IO
import sys

data_dir = "/data"

patient = 14
nwb_file = Path(data_dir) / f"sub{patient}.nwb"
    
if not nwb_file.exists():
    raise FileNotFoundError(f"NWB file for patient {patient} not found at {nwb_file}")

io = NWBHDF5IO(nwb_file, mode="r")
nwb_data = io.read()
print(f"NWB data: {nwb_data}")

# extract labels from nwb_data
label = nwb_data.stimulus["annotations_base"].to_dataframe()
movie_times = nwb_data.stimulus["base_movie_frame_times"]
print(f"Labels: {label}")
print(f"Movie times: {movie_times}")

# extract value from label dataframe where label_name is "summer"
summer_labels = label[label["label_name"] == "summer"]
print(f"Summer labels: {summer_labels}")
# extract only column value as numpy array
summer_labels = summer_labels["value"].values
print(f"Summer labels: {summer_labels}")