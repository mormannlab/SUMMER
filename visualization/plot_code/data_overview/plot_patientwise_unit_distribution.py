#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for generating a simple overview of the region/unit information in the dataset. 
Formatting classes for plotting stacked barchart. 

Taken from dhv_decoding.

Author: Alana Darcher
Date: 15 April 2025
"""
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd 
from tqdm import tqdm

from pynwb import NWBHDF5IO

target_regions = ["A", "AH", "MH", "PH", "M", "EC", "PHC", 'PHC', 'APH', 'MPH', 'PPH', "Other"]

replacement_regions = {
    "AH": "H",
    "MH": "H",
    "M": "H",
    "PH": "H", 
    "APH": "PHC",
    "MPH": "PHC", 
    "PPH": "PHC"
}

final_regions = ["A", "EC", "H", "PHC", "Other"]

def create_data_overview(data_dir):
    """
    Create a pandas DataFrame collection of the brain region and hemisphere for 
    every unit in the complete dataset. 

    Runs through every units table from each NWB file in the data directory and 
    collects the region information. useful for indexing data without having all
    info in memory.

    Returns:
        pandas.DataFrame: row-wise info for each unit in the complete dataset, across patients
    """
    i = 0 

    for path in tqdm(list(data_dir.glob("*.nwb"))):
        print(path)
        if path.is_dir():
            continue 

        patient_id = int(path.name.split(".")[0][3:])

        print(patient_id)

        io = NWBHDF5IO(path, mode="r")
        nwbfile = io.read()
        df_units = nwbfile.units.to_dataframe()
        df_ = pd.DataFrame({
        "patient_id": [patient_id] * len(df_units), 
        "brain_region": df_units["brain_region"],
        "hemisphere": [item[0] for item in df_units["hemisphere"]]
    })

        if i == 0:
            df_patient_overview = df_.copy()
        else:
            df_patient_overview = pd.concat([df_patient_overview, df_], ignore_index=True)
        io.close()
        i += 1
    return df_patient_overview


class UnitRegionDataCollectorNWB:
    """
    Collect the region information for each unit from the pandas DataFrame overview of NWB-formatted datasets.
    Formatted for input to UnitRegionDataProcessor. 
    """
    def __init__(self, df):
        self.df = df
        self.all_units = self.collect_unit_ids()
        self.nm_units = self.get_total_nm_units_per_patient()
        self.all_regions = self.collect_regions()
        
    def collect_unit_ids(self):
        return [np.arange(0, len(self.df[self.df["patient_id"] == patient]))
                for patient in set(self.df["patient_id"])]

    def collect_regions(self):
        return [np.array(self.df[self.df["patient_id"] == patient]["brain_region"])
             for patient in set(self.df["patient_id"])]
    
    def get_total_nm_units_per_patient(self):
        return [len(l) for l in self.all_units]


class UnitRegionDataProcessor:
    def __init__(self, data_collector):
        self.data_collector = data_collector
    
    def filter_by_region(self, target_regions):
        
        all_units = self.data_collector.all_units
        nm_units = self.data_collector.nm_units
        all_regions = self.data_collector.all_regions

        
        filtered_units = [
            [unit for unit, region in zip(pat_units, pat_regions) if region in target_regions]
            for pat_units, pat_regions in zip(all_units, all_regions)
        ]
                
        filtered_regions = [
            [region for region in pat_regions if region in target_regions]
            for pat_regions in all_regions
        ]
        
        nm_units = [len(u) for u in filtered_units]
        
        return filtered_units, nm_units, filtered_regions
    
    def consolidate_subregions(self, replacement_regions, region_list):
        """
        Replace more-specific subregion names (e.g."MH") with the
        more-general subregion name (e.g., "H").
        """
        consolidated_regions = [
            [replacement_regions.get(item, item) for item in pat_regions]
            for pat_regions in region_list
        ]
        return consolidated_regions
    
def format_regions_for_barplot(regions, regions_list):
    """
    Reformats regions into the form needed by matplotlib.bar
    in order to create a stacked bar plot for regions/patient.

    End indices for each list match "patients" inputted to
    get_unit_region_data()
    """
    
    data = {
        region: [Counter(pat_regions).get(region,0) for pat_regions in regions]
        for region in regions_list
    }
    
    return data