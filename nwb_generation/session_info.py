## experiment-wide information for NWB file generation

import numpy as np


session_description_base_text = "Single unit activity and associated stimulus labels for movie viewing session of participant "
lab = "Mormann Lab at Universitätsklinikum Bonn (University Hospital Bonn)"
institution = "Universitätsklinikum Bonn (University Hospital Bonn)"
experiment_description = "Single-neuron data recorded from Behnke-Fried microwires implanted in the human medial temporal lobe during the German-language presentation of a full-length movie (500 Days of Summer). This file also includes over 50 annotations of movie content."
keywords = ["cognitive neuroscience", "single units", "human single neuron", "human electrophysiology", 
            "amygdala", "parahippocampal cortex", "entorhinal cortex", "hippocampus", 
            "movie", "500 Days of Summer", "summer", "open source", "NWB"]
related_publications = "https://doi.org/10.7554/eLife.106758.1"


patient_subset = np.arange(1, 30) # use the index of this to get sub ids 

movie_analysis_pts = [36.72, 4763.12]

patient_nwb_id = {
    # intentionally empty -- data has been anonymized
}

patient_ages = {
    # intentionally empty -- see manuscript or data descriptor for age information
}

patient_sex = {
    # intentionally empty -- see manuscript or data descriptor for gender information
}

region_restriction = {
    "AH": "H",
    "MH": "H",
    "PH": "H",
    "A": "A",
    "EC": "EC",
    "PHC": "PHC",
    "PIC": "PIC", 
    "PRC": "PRC",
    "APH": "PHC",
    "MPH": "PHC",
    "PPH": "PHC",
}

region_exclusion = ["TT", "Ta", "Tb", "T","WM"] # T* - Tiefeelectroken; WM - white matter. 

region_full_names = {
    "AH": "anterior hippocampus",
    "MH": "medial hippocampus",
    "PH": "posterior hippocampus",
    "A": "amygdala",
    "EC": "entorhinal cortex",
    "PHC": "parahippocampal cortex",
    "PIC": "piriform cortex",
    "PRC": "perirhinal cortex",
    "APH": "parahippocampal cortex",
    "MPH": "parahippocampal cortex",
    "PPH": "parahippocampal cortex",
    "WM": "white matter",
    "FF": "fusiform gyrus",
    "LG": "lingual gyrus",
    "I": "insula",
}

hemisphere_full_names = {
    "L": "left",
    "R": "right"
}

dataset_labels = [
          'alison', 
          'autumn', 
          'beach',
          'bus', 
          'cafe', 
          'camera-cuts', 
          'car',
          'days-of-summer', 
          'douche', 
          'elevator',
          'family-home', 
          'gallery',
          'ikea', 
          'indoor-setting',
          'karaoke-bar', 
          'mckenzie', 
          'mckenzie-faces', 
          'millie',
          'millie-faces', 
          'office',  
          'other-cafe', 
          'park', 
          'paul',
          'paul-faces', 
          'persons', 
          'punch-bar', 
          'rachel', 
          'rachel-faces',
          'record-store', 
          'restaurant', 
          'rhoda', 
          'scenes', 
          'secretary',
          'soccer-field', 
          'street', 
          'summer', 
          'summer-apartment',
          'summer-body-sequence', 
          'summer-child-bedroom', 
          'summer-faces',
          'summer-presence', 
          'summer-speaking', 
          'the-graduate', 
          'theater',
          'tom', 
          'tom-apartment', 
          'tom-child-bedroom', 
          'tom-faces',
          'tom-speaking', 
          'train', 
          'vance', 
          'vance-faces', 
          'wedding-venue']