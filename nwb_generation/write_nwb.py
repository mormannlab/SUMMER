#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for structuring and saving the file-system-based movies data into the NWB format.

Included as an example -- will not run without access to original server with clinical data. 


Author: Alana Darcher, Uniklinikum Bonn
Date: 8 April 2025
License: MIT License
Dependencies: numpy, pandas, pynwb, mat73, locals
"""
import logging

import shutil 
from collections import Counter
from pathlib import Path
from datetime import datetime 
from dateutil import tz
from uuid import uuid4
import re

import numpy as np
import pandas as pd
from scipy.stats import sem

from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject, ProcessingModule
from pynwb.misc import Units, TimeSeries
from pynwb.core import DynamicTable, VectorData
from pynwb.epoch import TimeIntervals

import mat73

from session_infos import *
from stats.cell_type import SpikeWidth
from stats.metrics import calc_cv2, calculate_snr, count_isi_violations
from nwb_generation.utils.data_io import *
from utils.process_labels import *
from static_info import *

class SessionInfo:
    """
    Manages session info for a given patient.
    """
    def __init__(self, subject_dir):
        self.subject_dir = subject_dir
        self.session_info = self._load_session_info()
        self.session_start_time = self._get_session_start_time()

    def _load_session_info(self):
        session_info_path = self.subject_dir / "session_info.npy"
        return np.load(session_info_path, allow_pickle=True).item()
    
    def _get_session_start_time(self):
        session_time = f"{self.session_info['date']}"
        session_start_time = datetime.strptime(session_time, '%Y-%m-%d',)

        # For privacy reasons, we only include the year. All other time values are set to a common value.
        session_start_time = datetime.strptime(f"{session_start_time.year}-1-1_12:00:00", '%Y-%m-%d_%H:%M:%S')
        return session_start_time.replace(tzinfo=tz.gettz('Europe/Berlin'))

class SpikeInfo: 
    def __init__(self, subject_dir):
        self.subject_dir = subject_dir
        self.names_cscs = self._load_cscs()
        self.cscs_with_units = self._filter_cscs()

    def _load_cscs(self):
        """
        Loads and sorts the names of the unit data files from the refractored spiking_data directory.
        Uses natural sorting/human readable sorting. 
        Filenames have the following pattern: CSC6_SUA1.npy
        """
        filepaths_cscs = (self.subject_dir / "spiking_data").glob(f"CSC*")
        names_cscs = [p.name for p in filepaths_cscs]
        return sorted(names_cscs, key=natural_keys)

    def _filter_cscs(self):
        """
        Grab the channel numbers from the filenames and return all unique numbers. 
        Because the CSC ids are pulled from the spike files, by definition this list will only
        include channels with valid units. 
        """
        return np.unique([int(re.findall(r'-?\d+', f)[0]) for f in self.names_cscs])

class WatchlogInfo:
    def __init__(self, subject_dir, pat_id):
        self.subject_dir = subject_dir
        self.pat_id = pat_id
        
        self.clean_wl = self._load_clean_wl()
        self.clean_rec = self._load_clean_rec()
        
        self.raw_wl = self._load_raw_wl()
        self.raw_rec = self._load_raw_rec()

        self.run_check()

    def _load_clean_wl(self):
        path = Path(self.subject_dir, "cleaned_watchlog", f"{self.pat_id}_pts.npy")
        return np.load(path, allow_pickle=True)

    def _load_clean_rec(self):
        path = Path(self.subject_dir, "cleaned_watchlog", f"{self.pat_id}_rec.npy")
        return np.load(path, allow_pickle=True)
    
    def _load_raw_wl(self):
        path = Path(self.subject_dir, "watchlogs", f"raw_{self.pat_id}_pts.npy")
        return np.load(path, allow_pickle=True)
    
    def _load_raw_rec(self):
        path = Path(self.subject_dir, "watchlogs", f"raw_{self.pat_id}_rec.npy")
        return np.load(path, allow_pickle=True) / 1000 # convert to milliseconds
    
    def run_check(self):
        if self.clean_rec[0] != self.raw_rec[0]:
            print(f"cleaned onset {self.clean_rec[0]}")
            print(f"raw onset     {self.raw_rec[0]}")
        assert self.clean_rec[0] >= self.raw_rec[0]

class MovieBinningData:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.bin_lengths = [40, 100, 200, 500, 1000] # multiples of the frame rate (0.04s)
        self.actual_bin_length = {"40": 40, "100": 80, "200": 200, "500": 480, "1000": 1000} # hack to get around filenames using other numbers
        self.df = self.build_df()
    
    def build_df(self):
        bin_info_dicts = []

        for bl in self.bin_lengths:
            edges = np.load(self.data_dir.parent / "movie_edges" / f"edges_movie_{bl}.npy", allow_pickle=True)
            frames = np.load(self.data_dir.parent / "movie_edges" / f"relevant_frames_{bl}.npy", allow_pickle=True)

            assert len(edges) == len(frames) + 1

            bin_info_dicts.append(
                {"bin_length": self.actual_bin_length[str(bl)], 
                "edges": edges,
                "frames": frames}
            )

        return pd.DataFrame(bin_info_dicts)

class MovieData:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.movie_pts = self._load_movie_pts()
    
    def _load_movie_pts(self):
        """PTS used for analysis, excludes production credits and end credits."""
        return np.load(self.data_dir.parent / "movie_edges" / f"pts_movie.npy", allow_pickle=True)

class LocalizationData:
    def __init__(self, subject_dir):
        self.subject_dir = subject_dir
        self.patient_id = int(self.subject_dir.parent.name)
        self.processed_localizations = self._load_processed_localizations()
        self.processed_localizations = self._rename_hippocampal_electrodes(self.processed_localizations)

        self.df = self.build_df()
        self.df = self._rename_hippocampal_electrodes(self.df)

        self.df_mtl_only = self.build_df(restrict=True)
        self.df_mtl_only = self._rename_hippocampal_electrodes(self.df_mtl_only)

        self.df_electrode_groups = self.build_df_electrode_groups(manual_localizations=True)
        self.df_electrode_groups = self._rename_hippocampal_electrodes(self.df_electrode_groups)

    def _load_legui_csv(self):
        localizations = pd.read_csv(list((self.subject_dir / "localizations").glob("*MNI*"))[0])
        return localizations
    
    def _load_processed_localizations(self):
        loaded_df = pd.read_csv(self.subject_dir / "localizations" / f"{self.patient_id}_finalized_localizations.csv")
        bundle_ids = [entry[-1] for entry in loaded_df["channel_name_postop"]]
        loaded_df["bundle_index"] = bundle_ids
        return loaded_df
    
    def _rename_hippocampal_electrodes(self, df):
        """Correct for the outdated/original naming scheme of middle hippocampal electrodes (MH).

        Replaces MH with PH (posterior hippocampus) for implantation schemes with only two hippocampal electrodes.
        Ignores schemes with three hippocampal electrodes. Uses the region_pre_review (original clincial assignment)
        to decide if there were two or three hippocampal electrodes.

        Args:
            df (pd.DataFrame): dataframe containing electrode information

        Returns:
            df: dataframe with corrected hippocampal region abbreviation
        """
        regions_in_dataframe = np.unique(self.processed_localizations["region_pre_review"])

        if "AH" and "MH" in regions_in_dataframe:    
            if "PH" not in regions_in_dataframe:
                df.replace({"brain_region":"MH"}, {"brain_region":"PH"}, inplace=True)
        return df

    def build_df(self, restrict=False):
        """
        Creates the DataFrame used for populating the NWB electrodes and electrode_groups.
        Averages across the MNI coordinates for the micros (within coordinate) and sets the average for the micro locations.

        Args:
            restrict (bool, optional): Restricts to just originally-labeled MTL neurons when true. Defaults to False.

        Returns:
            pandas dataframe: DataFrame containing localization information
        """
        if restrict:
            region_set = set(zip(self.processed_localizations["region_pre_review"], self.processed_localizations["hemisphere"]))
            region_set = {item for item in region_set if any(item[0].startswith(prefix) for prefix in region_restriction.keys())}
        else:
            region_set = set(zip(self.processed_localizations["region_pre_review"], self.processed_localizations["hemisphere"]))

        for p, pair in enumerate(region_set):
            
            # The matching and averaging is done for using the pre-review locations as these are unique within each patient.
            # After localization, it's possible to have "duplicated" regions. 
            region_matches = self.processed_localizations["region_pre_review"] == pair[0]
            hemisphere_matches = self.processed_localizations["hemisphere"] == pair[1]
            matching_rows = self.processed_localizations[region_matches & hemisphere_matches].copy()
            assert len(matching_rows) == 8, f"{len(matching_rows)}"

            matching_rows["mni_x_micro"]  = [np.mean(matching_rows["mni_x_micro"])] * 8
            matching_rows["mni_y_micro"]  = [np.mean(matching_rows["mni_y_micro"])] * 8
            matching_rows["mni_z_micro"]  = [np.mean(matching_rows["mni_z_micro"])] * 8

            if p == 0:
                df = matching_rows.copy()
            else: 
                df = pd.concat([df, matching_rows], ignore_index=True)
        df = df.sort_values(by='csc_nr')
        return df
    
    def handle_duplicate_electrode_groups(self, df):
        region_hemisphere_combinations = list((zip(df["brain_region"], df["hemisphere"])))
        region_hemisphere_set = set(region_hemisphere_combinations)

        if len(region_hemisphere_combinations) == len(region_hemisphere_set):
            dummy_column = [""] * len(df)
            df["unique_identifier_tags_for_duplicated_locations"] = dummy_column
            return df
        else:
            print("Handling duplicated localization groups.")
            duplicate_region_dict = Counter(region_hemisphere_combinations)
            unique_region_identifier_tags = []

            temp_duplicate_list = []
            temp_duplicate_list_2 = []

            for entry in region_hemisphere_combinations:
                if duplicate_region_dict[entry] == 2:
                    if entry not in temp_duplicate_list:
                        unique_region_identifier_tags.append("a")
                        temp_duplicate_list.append(entry)
                    else: 
                        unique_region_identifier_tags.append("b")
                elif duplicate_region_dict[entry] == 3:
                    if entry not in temp_duplicate_list:
                        unique_region_identifier_tags.append("a")
                        temp_duplicate_list.append(entry)
                    elif entry not in temp_duplicate_list_2:
                        unique_region_identifier_tags.append("b")
                        temp_duplicate_list_2.append(entry)
                    else: 
                        unique_region_identifier_tags.append("c")

                elif duplicate_region_dict[entry] == 1:
                    unique_region_identifier_tags.append('')

            assert len(unique_region_identifier_tags) == len(df["brain_region"])
            df["unique_identifier_tags_for_duplicated_locations"] = unique_region_identifier_tags
            return df

    def build_df_electrode_groups(self, manual_localizations=True):
        """
        Create the DataFrame used for populating the NWB electrode groups entry.
        """
        if manual_localizations:
            tmp = []
            for row in self.processed_localizations.itertuples():
                name = row.channel_name_postop
                region = row.region_pre_review
                hemisphere = row.hemisphere

                if int(name[-1]) == 1:
                    print(f"  including {hemisphere}{region} in electrode_groups.")        
                    tmp.append(row)
            df = pd.DataFrame(tmp)
            df = self.handle_duplicate_electrode_groups(df)
        
        else:
            tmp = []
            for row in self.processed_localizations.itertuples():
                name = row.channel_name_postop
                region = row.region_pre_review 
                
                if region not in region_restriction.keys():
                    print(f"  excluding {region}")
                    continue
                
                if int(name[-1]) == 1:
                    tmp.append(row)

            df = pd.DataFrame(tmp)
        return df
    
class ChannelData:
    def __init__(self, subject_dir, patient_id):
        self.subject_dir = subject_dir
        self.patient_id = patient_id
        self.micro_channels = self._load_channel_info()

    def _load_channel_info(self):
        micro_channels = pd.read_csv(self.subject_dir / "ChannelNames.txt", delimiter="\t", header=None, names=['Filename',])
        micro_channels["csc_nr"] = micro_channels.index + 1
        return micro_channels

class MetricsData:
    def __init__(self, subject_dir):
        self.subject_dir = subject_dir
        self.median_abs_deviation_file = self._load_mad_file()
        self.mad = self.median_abs_deviation_file["MAD_channels"]

        self.iso_distances = self._load_isolation_distances()
        self.iso_channels = self._grab_cscs_with_units()

    def _load_mad_file(self):
        mad_all_channels = mat73.loadmat(self.subject_dir / 'metrics' / 'median_abs_deviations.mat')
        if sum(np.isnan(mad_all_channels["MAD_channels"])) != 0:
            print(f"Unassigned value for MAD for a channel. {np.where(np.isnan(mad_all_channels['MAD_channels']))}")
        return mad_all_channels 
    
    def _load_isolation_distances(self):
        return np.load(self.subject_dir / "metrics" / "isolation_distances.npy", allow_pickle=True).item()
    
    def _grab_cscs_with_units(self):
        return np.unique([int(name.split("_")[0][3:]) for name in self.iso_distances["filenames"]])

class BaseLabelsData:
    def __init__(self, subject_dir):
        self.subject_dir = subject_dir
        self.base_labels = self._load_base_labels()
        self.df = self.build_df()

    def _load_base_labels(self):
        base_labels = Path(self.subject_dir, "base_labels.npy")
        base_labels = np.load(base_labels, allow_pickle=True).item()
        df_base_labels = pd.DataFrame(base_labels)
        return df_base_labels
    
    def build_df(self):
        for i, label in enumerate(self.base_labels.itertuples()):
            name = label.names
            values = label.values
            starts = label.starts
            stops = label.stops

            if name in dataset_labels:
                pass
            else:
                print(f"   {name} not included in base labels.")
                continue

            assert len(values) == len(starts) == len(stops), "Number of label entries is inconsistent."

            entry_index = np.arange(1, len(values)+1) # using 1-indexing for compatibility with matlab

            for e, entry in enumerate(entry_index):
                d_ = {
                    "names": [name.lower()],
                    "entry_index": [entry],
                    "values": [values[e]],
                    "starts": [starts[e]], 
                    "stops": [stops[e]]
                }
                df = pd.DataFrame(d_,)

                if i == 0 and e == 0:
                    df_base_labels_db_format = df.copy()                     
                else:
                    df_base_labels_db_format = pd.concat([df_base_labels_db_format, df])

        return df_base_labels_db_format

class AlignedLabelsData:
    def __init__(self, subject_dir, rec):
        self.subject_dir = subject_dir
        self.rec = rec
        self.aligned_labels = self._load_aligned_labels()
        self.df = self.build_df()

    def _load_aligned_labels(self):
        aligned_labels = Path(self.subject_dir, "aligned_labels.npy")
        aligned_labels = np.load(aligned_labels, allow_pickle=True).item()
        df_aligned_labels = pd.DataFrame(aligned_labels)

        return df_aligned_labels

    def build_df(self):
        for i, label in enumerate(self.aligned_labels.itertuples()):
            name = label.names
            values = label.values
            starts = label.starts / 1000 - self.rec[0]
            stops = label.stops / 1000 - self.rec[0]

            if name in dataset_labels:
                pass
            else:
                print(f"   {name} not included in aligned labels.")
                continue

            assert len(values) == len(starts) == len(stops), "Number of label entries is inconsistent."
            assert len(values) > 1, f"Only one entry for label {label}."
            entry_index = np.arange(1, len(values)+1) # using 1-indexing for compatibility with matlab

            for e, entry in enumerate(entry_index):
                d_ = {
                    "names": [name.lower()],
                    "entry_index": [entry],
                    "values": [values[e]],
                    "starts": [starts[e]], 
                    "stops": [stops[e]]
                }
                df = pd.DataFrame(d_,)
                
                if i == 0 and e == 0:
                    df_aligned_labels_db_format = df.copy()
                else:
                    df_aligned_labels_db_format = pd.concat([df_aligned_labels_db_format, df])

        return df_aligned_labels_db_format

##
# Workhorse Class
##

class PatientNWB:
    def __init__(self, patient_id, sub_id, data_dir, save_dir,):
        self.pat_id = patient_id
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.sub_id = sub_id
        self.subject_dir = Path(self.data_dir, str(self.pat_id), "session_1")

        self.logger = self._init_logger()

        self.sw = SpikeWidth()

        self.session_info = SessionInfo(self.subject_dir)
        self.spike_info = SpikeInfo(self.subject_dir)
        self.watchlog_info = WatchlogInfo(self.subject_dir, self.pat_id)
        
        self.movie_binning_data = MovieBinningData(self.data_dir)
        self.movie_pts = MovieData(self.data_dir).movie_pts

        self.localizations_data = LocalizationData(self.subject_dir)
        self.channel_data = ChannelData(self.subject_dir, self.pat_id)

        self.metrics = MetricsData(self.subject_dir)
        self.median_abs_deviations = self.metrics.mad
        assert len(self.median_abs_deviations) == len(self.localizations_data.processed_localizations["csc_nr"]), "Discrepancy in number of channels in MAD results."

        self.isolation_distances = self.metrics.iso_distances
        self.df_isolation_distances = pd.DataFrame(self.isolation_distances)
        assert np.all(self.metrics.iso_channels == self.spike_info.cscs_with_units), f"Mismatch in the number of channels with units. Isodist: {np.unique(self.df_isolation_distances['filenames'])}, SpikeInfos: {self.spike_info.cscs_with_units}"

        self.base_labels_data = BaseLabelsData(self.subject_dir)
        self.aligned_labels_data = AlignedLabelsData(self.subject_dir, self.watchlog_info.raw_rec)

        self.sr = 32768. # sampling rate from the NLX amplifier
        self.frame_rate = 0.04 # frame rate of the movie file
        self.len_movie = 5029.68 
        self.fps = 25
        self.num_frames = int(self.len_movie / self.frame_rate)+1 
        self.pts_full_movie = [round((x * self.frame_rate), 2) for x in range(0, self.num_frames)] # includes production and end credits.

        self.t_r = 3 # refractory period, ms
        self.t_c = (1 / self.sr) * 1000 # censored time period 

        # init nwbfile components
        self.nwbfile = self._init_nwbfile()
        self.device = self._define_device()
        self._init_electrodes()
        self._init_units()
        self._init_unit_columns()
        self.annotations_patient_aligned = self._init_aligned_annotations()
        self.annotations_base = self._init_base_annotations()
        self._init_processing_module()

    #### initialization functions ####

    def _init_logger(self,):
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M")
        logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.DEBUG,
        filename=f"logs/write_nwb_{timestamp}.log")
        logger = logging.getLogger(__name__)
        logger.info(f"patient {self.pat_id}, subject id: {self.sub_id}")
        logger.info(f"Data location: {self.data_dir}")
        logger.info(f"Save location: {self.save_dir}")
        logger.info(f"Subject Dir: {self.subject_dir}")
        return logger

    def _init_nwbfile(self, ):
        # note: session_description includes patient-identifying information
        # and is excluded from the codebase. 
        nwbfile = NWBFile(
            session_description=f"{session_description_base_text}{self.sub_id}",
            identifier=str(uuid4()),
            session_start_time=self.session_info.session_start_time,
            session_id=f"sub{self.sub_id}",
            lab=lab, 
            institution=institution,
            experiment_description=experiment_description,
            keywords=keywords,
            related_publications=related_publications
        )
        self.logger.info("NWBfile created.")
        return nwbfile 

    def _define_device(self, ):
        device = self.nwbfile.create_device(
                    name="NeuraLynx ATLAS", 
                    description="Acquisition amplifier", 
                    manufacturer="NeuraLynx"
                )
        self.logger.info("Device defined.")
        return device

    def _init_electrodes(self,):
        self.nwbfile.add_electrode_column(name="hemisphere",                description="Hemisphere in which electrode was implanted")
        self.nwbfile.add_electrode_column(name="brain_region",              description="Brain region in which electrode was implanted")
        self.nwbfile.add_electrode_column(name="bundle_index",              description="Microwire channel number within the implanted bundle (unordered)")
        self.nwbfile.add_electrode_column(name="csc_nr",                    description="Continuously sampling channel (CSC) id assigned to the microwire channel")
        self.logger.info("Electrode table created.")

    def _init_units(self):
        self.nwbfile.units = Units(
        name="units", 
        waveform_rate=self.sr,
        waveform_unit="microvolts", 
        description=f"Spike times (milliseconds), waveforms (uV), and associated spike sorting metrics for {self.sub_id}'s movie session."
        )
        self.logger.info("Unit table created.")

    def _init_unit_columns(self):
        self.nwbfile.add_unit_column(name="unit_id",        description="unique integer id for each unit within a session")
        self.nwbfile.add_unit_column(name="csc_nr",         description="id for the electrode channel from which the unit was sorted, links to electrodes table")
        self.nwbfile.add_unit_column(name="brain_region",   description="brain region from which unit was recorded.")
        self.nwbfile.add_unit_column(name="hemisphere",     description="hemisphere from which unit was recorded")
        self.nwbfile.add_unit_column(name="is_single_unit", description="indicates if a given unit is a putative single neuron or putative multi-unit activity")
        self.nwbfile.add_unit_column(name="peak_SNR",       description="signal-to-noise ratio calculated via max. amp of the mean spike waveform")
        self.nwbfile.add_unit_column(name="isi_violations", description="percentage of refractory-period violations in the spike train")
        self.nwbfile.add_unit_column(name="cv2",            description="degree of variation in the inter-spike intervals")
        self.nwbfile.add_unit_column(name="iso_dist",       description="isolation distance of cluster from other spike events on the same channel")
        self.nwbfile.add_unit_column(name="cell_type",      description="indicates if a given unit is a putative pyramidal cell or interneuron")
        self.nwbfile.add_unit_column(name="waveform_sem",   description="standard error of the mean waveform across all spikes for the unit")

        self.logger.info("Custom unit columns added.")

    def _init_aligned_annotations(self):
        annotations_patient_aligned = TimeIntervals(
                name="annotations_patient_aligned",
                                description=("Annotation data for the labeled movie features, aligned to the watchlog of a given participant (handles pauses in movie playback). Can be non-monotonic due to patient-led skips in playbak. Time given in milliseconds. "
                            "Contents: starts: onset of a labeled segment, stops: offset of the labeled segment, values: whether the feature was present in the labeled segment (1) or not (0)")
                        )
        
        annotations_patient_aligned.add_column(
                name="label_name", 
                description="label name"
            )

        annotations_patient_aligned.add_column(
                name="entry_index", 
                description="index for the occurrence of the label within the movie"
            )

        annotations_patient_aligned.add_column(
                name="value", 
                description="indicates if time span corresponds to labeled entity being present (=1) or absent (=0)"
            )
        
        self.logger.info("Patient aligned annotations table created.")
        return annotations_patient_aligned

    def _init_base_annotations(self):
        annotations_base = TimeIntervals(
            name="annotations_base",
            description=("Annotation data for the labeled movie features, aligned to directly to the movie, given in presentation time stamps (seconds)."
                        "Contents: starts: onset of a labeled segment, stops: offset of the labeled segment, values: whether the feature was present in the labeled segment (1) or not (0)")
                        )

        annotations_base.add_column(
            name="label_name", 
            description="label name"
        )

        annotations_base.add_column(
            name="entry_index", 
            description="index for the occurrence of the label within the movie"
        )

        annotations_base.add_column(
            name="value", 
            description="indicates if time span corresponds to labeled entity being present (=1) or absent (=0)"
        )
        self.logger.info("Base annotations table created.")
        return annotations_base
    
    def _init_processing_module(self):
        mod = ProcessingModule(name="misc", description="movie annotation and frame information processed for machine learning applications")
        self.nwbfile.add_processing_module(mod)
        self.logger.info("Processing table added.")

    ##### population functions #####

    def populate_subject(self, ):
        subject = Subject(
        subject_id=str(self.sub_id),
        age=f"P{patient_ages[self.pat_id]}Y",
        sex=patient_sex[self.pat_id],
        species="Homo sapiens",
        description="BF-implanted epilepsy patient"
        )
        self.nwbfile.subject = subject
        self.logger.info(f"Subject info populated: {subject}")

    def populate_movie_binning_data(self):
        """
        Adds movie binning data to the NWB file using an explicit DynamicTable with ragged columns for 
        edges and frames. Populates from MovieBinningData.

        Note: differs from other stimulus info and tables as this is not pre-defined.
        """
        df = self.movie_binning_data.df
        n = len(df)

        # create the table
        table = DynamicTable(
            name="movie_binning_info",
            description=(
                "Edges and corresponding frames indices (given as filenames) for binning the data using the frame onsets as edges."
                "Each row corresponds to one bin_length."
            ),
            id=np.arange(n, dtype=np.int64),
            columns=[]
        )

        # add column: bin_length
        table.add_column(
            name="bin_length",
            description="Length of the bin in milliseconds.",
            data=np.array(df["bin_length"].values, dtype=np.int64), 
        )

        edges = [np.asarray(x, dtype=np.float64) for x in df["edges"].tolist()]
        table.add_column(
            name="edges",
            description="Bin edges aligned to frame onsets.",
            data=edges,
            index=True
        )

        frames = [list(map(str, x)) for x in df["frames"].tolist()]
        table.add_column(
            name="frames",
            description="Frames indices corresponding to the bin edges.",
            data=frames,
            index=True
        )

        mod = self.nwbfile.processing["misc"]
        mod.add(table)

        self.logger.info("Movie binning info populated.")


    def populate_movie_annotations(self):
        """
        Creates and adds a table to the NWB file containing the indicator functions for
        each labeled feature.
        """
        df = self.base_labels_data.df

        table = DynamicTable(
            name="movie_annotations_indicator_functions",
            description=(
                "Indicator functions (0=feature absent, 1=feature present) for all labels, spanning the entire film."
                "Each value in the function corresponds to a movie frame, monotonically increasing."
            )
        )

        table.add_column(
            name="label_name",
            description="Label name (string)."
        )

        table.add_column(
            name="indicator_function",
            description=f"Indicates presence/absence of labeled feature for all {self.num_frames} frames in the movie.",
            index=True
        )

        for label, sub in df.groupby("names"):
            print(f"  {label}")
            vec = make_label_from_start_stop_times(sub["values"].to_numpy(), sub["starts"].to_numpy(), sub["stops"].to_numpy(), self.pts_full_movie, )
            table.add_row(label_name=str(label).lower(), indicator_function=vec)
            print(sum(vec))
       
        mod = self.nwbfile.processing["misc"]
        mod.add(table)

        self.logger.info("Movie anntation table populated.")

    def populate_movie_pts(self, ):
        """
        Adds movie PTS (presentation timestamps) data to the NWB file from dataframe built by MovieData.
        """
        movie_pts_nwb = TimeSeries(name="movie_frame_times_analysis", 
           description="onset times for each frame in the movie for the portion used in analysis (excludes production credits and the end credits), given in the presentation time stamps (seconds, referenced to movie start)",
           unit="seconds",
           rate=self.frame_rate,
           data=self.movie_pts)
        self.nwbfile.add_stimulus(stimulus=movie_pts_nwb)

        movie_pts_nwb = TimeSeries(name="movie_frame_times_base", 
           description="onset times for each frame in the movie for the entire movie, given in the presentation time stamps (seconds, referenced to movie start)",
           unit="seconds",
           rate=self.frame_rate,
           data=self.pts_full_movie)
        self.nwbfile.add_stimulus(stimulus=movie_pts_nwb)

        self.logger.info("Movie PTS info populated (frames analyzed and complete set).")

    def populate_electrode_groups(self):
        """
        Populates the NWB file electrode groups table by adding all macros in the recording. 
        Uses the dataframe built by LocalizationData.df_electrode_groups.
        Name includes post-localization brain region and a character tag in case of duplicated regions. 
        """
        self.logger.info("Populating Electrode Groups table:")

        for macro in self.localizations_data.df_electrode_groups.itertuples():
            
            self.nwbfile.create_electrode_group(
                name=f"{macro.hemisphere}{macro.brain_region}{macro.unique_identifier_tags_for_duplicated_locations}",
                description=f"depth electrode {macro.region_pre_review}{macro.hemisphere} (macro contact), {macro.brain_region}{macro.hemisphere} manual review",
                device=self.device,
                location=f"{hemisphere_full_names[macro.hemisphere]} {region_full_names[macro.brain_region]}"
            )

            self.logger.info(f"    manual location:{macro.hemisphere}{macro.brain_region}  / pre-review: {macro.region_pre_review} added.")

    def populate_micro_channels(self):
        """
        Populates the NWB file electrode table by iterating through all micro channels with valid units. 
        Only includes channels with valid units AND from regions of interest (e.g. excluding temporal FCD electrodes).
        Process:
            - processed_localizations (LocalizationData) is filtered to include only channels with valid units
            - for each channel, the averaged micro location is taken from LocalizationData.df
        """
        self.logger.info("Populating Electrode table: ")
        
        for row in self.localizations_data.processed_localizations[self.localizations_data.processed_localizations["csc_nr"].isin(self.spike_info.cscs_with_units)].itertuples():
        
            if row.region_pre_review in region_exclusion:
                self.logger.info(f"    excluding {row.region_pre_review}")
                continue
            
            # manual localizations version
            original_channel_name = row.channel_name_postop 
            original_channel_name_first_microwire = f"{original_channel_name[:-1]}1" # using this as a unique identifier
            unique_identifier_tags_for_duplicated_locations = self.localizations_data.df_electrode_groups[self.localizations_data.df_electrode_groups["channel_name_postop"] == original_channel_name_first_microwire]["unique_identifier_tags_for_duplicated_locations"].iloc[0]

            # take averaged micro locations
            x = self.localizations_data.df[self.localizations_data.df["channel_name_postop"] == original_channel_name]["mni_x_micro"].iloc[0]
            y = self.localizations_data.df[self.localizations_data.df["channel_name_postop"] == original_channel_name]["mni_y_micro"].iloc[0]
            z = self.localizations_data.df[self.localizations_data.df["channel_name_postop"] == original_channel_name]["mni_z_micro"].iloc[0]

            electrode_group = self.nwbfile.electrode_groups[f"{row.hemisphere}{row.brain_region}{unique_identifier_tags_for_duplicated_locations}"]
            self.nwbfile.add_electrode(
                group=electrode_group,
                location=f"{hemisphere_full_names[row.hemisphere]} {region_full_names[row.brain_region]}",
                hemisphere=row.hemisphere,
                brain_region=row.brain_region,
                bundle_index=row.bundle_index,
                csc_nr=row.csc_nr,
                x=x,
                y=y,
                z=z
            )
            self.logger.info(f"    {row.hemisphere}{row.brain_region}{row.bundle_index} added.")

    def restrict_spiketimes_to_movie(self, unit_path):
        unit = np.load(unit_path, allow_pickle=True).item()
        times_raw = unit["times"] # in milliseconds
        amps_raw = unit["amps"]
        
        # restrict to just the movie-related activity
        left_idx = np.searchsorted(times_raw, self.watchlog_info.raw_rec[0], side='left')
        right_idx = np.searchsorted(times_raw, self.watchlog_info.raw_rec[-1], side='right')
        times_filtered = times_raw[left_idx:right_idx] - self.watchlog_info.raw_rec[0] ## reindex times to the start of the movie, instead of amplifier time
        amps_filtered = amps_raw[left_idx:right_idx]
        
        # spikes, but not during the movie period. 
        if times_raw.any() == 1 and times_filtered.any() == 0:
            pass
        
        return times_filtered, amps_filtered
    
    def restrict_spiketimes_to_analyzed_movie(self, times_filtered, amps_filtered):
        """
        Optionally restrict the spiketimes to just those 
        included in the portion of the movie analyzed.

        Args:
            times_filtered (np.ndarray): spike times, standardized to the start of the movie
            amps_filtered (np.ndarray): amplitudes of each spike in times_filtered
        """
        pts_analysis_start, pts_analysis_stop = movie_analysis_pts
        left_idx = np.searchsorted(times_filtered, pts_analysis_start, side='left')
        right_idx = np.searchsorted(times_filtered, pts_analysis_stop, side='right')
        times_filtered_analysis = times_filtered[left_idx:right_idx]
        amps_filtered_analysis = amps_filtered[left_idx:right_idx]

        assert np.min(times_filtered_analysis) >= pts_analysis_start, f"Something went wrong with the spike time filtering. Min spike time is f{np.min(times_filtered_analysis)}, shouldn't be less than f{pts_analysis_start}."
        assert np.max(times_filtered_analysis) <= pts_analysis_stop, f"Something went wrong with the spike time filtering. Max spike time is f{np.max(times_filtered_analysis)}, shouldn't be more than f{pts_analysis_stop}."

        return times_filtered_analysis, amps_filtered_analysis
        
    
    def get_unit_type(self, name, waveform):
        unit_class_info = name.split("_")[1]
        if unit_class_info[0] == "M":
            single_unit = 0
            cell_type = "multi-unit"
        else:
            single_unit = 1

            if waveform[19] > 0:
                spike_width, cell_type = self.sw.calculate_spike_width(waveform)
            else: 
                cell_type = "negative-peak"

        unit_index = int(unit_class_info[-1])
        return single_unit, cell_type, unit_index

    def isi_contamination_wrapper(self, times):
        """
        Wraps the ISI contamination functions. 

        Args:
            times (np.array): list of spike times for a unit in milliseconds

        Returns:
            float: percentage of violations
        """
        r = count_isi_violations(times, self.t_r)
        if len(times) <= 1:
            p = 0
        else:
            p = r / (len(times) - 1) # denomniator is number of ISIs
        return p * 100

    def grab_isolation_distance(self, filename_csc):
        print(filename_csc)
        fname = filename_csc.split(".")[0]
        matches = self.df_isolation_distances.loc[self.df_isolation_distances["filenames"] == fname, "isolation_distances"]
        isolation_distance = float(matches.iloc[0]) if not matches.empty else np.nan
        return isolation_distance

    def populate_units(self):
        self.logger.info(f"Populating unit table:")

        nm_units_included = 0
        nm_units_excluded = 0
        all_regions = []
        unit_id = 0

        for filename_csc in self.spike_info.names_cscs:
            unit_path = self.subject_dir / "spiking_data" / filename_csc
            csc_nr = int(re.findall(r'-?\d+', filename_csc)[0])
            name = filename_csc.split(".")[0]

            brain_region = self.localizations_data.processed_localizations[self.localizations_data.processed_localizations["csc_nr"] == csc_nr]["brain_region"] 
            if not brain_region.empty:
                brain_region = brain_region.iloc[0]
            else:
                AssertionError

            hemisphere = self.localizations_data.processed_localizations[self.localizations_data.processed_localizations["csc_nr"] == csc_nr]["hemisphere"]
            if not hemisphere.empty:
                hemisphere = hemisphere.iloc[0]
            else:
                AssertionError

            # ############# REMOVE BEFORE RELEASE
            region_pre_review = self.localizations_data.processed_localizations[self.localizations_data.processed_localizations["csc_nr"] == csc_nr]["region_pre_review"].iloc[0]
            # #####################################

            if region_pre_review in region_exclusion:
                self.logger.info(f"    excluding {name}, {hemisphere}{brain_region}, original region {region_pre_review}")
                nm_units_excluded += 1
                continue
            else:
                self.logger.info(f"    including {name}, {hemisphere}{brain_region}. original region {region_pre_review}")
                nm_units_included += 1
                all_regions.append(brain_region)

            times, amps = self.restrict_spiketimes_to_movie(unit_path)

            waveform_mean = np.mean(amps, axis=0)
            waveform_sem = sem(amps, axis=0)
            assert len(waveform_mean) == len(waveform_sem), "Waveform mean and sem have different lengths."

            single_unit, cell_type, _ = self.get_unit_type(name, waveform_mean)
            
            # metrics
            channel_mad = self.median_abs_deviations[csc_nr-1]
            peak_snr = calculate_snr(waveform_mean, channel_mad)
            isi_below = self.isi_contamination_wrapper(times)
            cv2 = calc_cv2(times)
            iso_dist = self.grab_isolation_distance(filename_csc)

            self.nwbfile.add_unit(spike_times=times, unit_id=unit_id, csc_nr=csc_nr, brain_region=brain_region, hemisphere=hemisphere, is_single_unit=bool(single_unit), 
                       peak_SNR=peak_snr, isi_violations=isi_below, cv2=cv2, iso_dist=iso_dist, cell_type=cell_type, 
                       waveform_mean=waveform_mean, waveform_sem=waveform_sem,
                       )
            unit_id += 1
            
        self.logger.info(f"Number of units included for patient {self.pat_id}: {nm_units_included}")
        self.logger.info(f"Number of units excluded for patient {self.pat_id}: {nm_units_excluded}")
        self.logger.info(f"Regions kept: {Counter(all_regions)}")
        

    def populate_aligned_annotations(self):
        for row in self.aligned_labels_data.df.itertuples():
                       
            self.annotations_patient_aligned.add_row(start_time=row.starts, stop_time=row.stops, 
                                                     label_name=row.names, entry_index=row.entry_index, value=row.values)

        self.nwbfile.add_stimulus(stimulus=self.annotations_patient_aligned)
        self.logger.info(f"Populated patient aligned annotations.")

    def populate_base_annotations(self):
        for row in self.base_labels_data.df.itertuples():

            self.annotations_base.add_row(start_time=row.starts, stop_time=row.stops, 
                                          label_name=row.names, entry_index=row.entry_index, value=int(row.values))

        self.nwbfile.add_stimulus(stimulus=self.annotations_base)
        self.logger.info(f"Populated base annotations.")

    def populate_clean_watchlogs(self):
        pts_column = VectorData(
            name="pts",
            data=self.watchlog_info.clean_wl,
            description="presentation timestamps (pts, secoonds) for each frame in the movie shown to the patient during the experiment",
        )

        neural_rectime_column = VectorData(
            name="neural_recording_time",
            data=self.watchlog_info.clean_rec - self.watchlog_info.raw_rec[0],
            description="neural recording system timestamp (milliseconds) for each frame in the movie shown to the patient during the experiment, corresponds to the pts",
        )

        watchlogs = DynamicTable(
            name="cleaned_watchlogs",
            description=("Log of movie presentation to participant. Processed: paused portions unified, inconsistencies in frame rate corrected. "
                        "Lookup table containing: frame times (presentation time stamps, seconds) and the corresponding timestamp from the neural recording system (milliseconds)."),
            
            colnames=[
                "pts",
                "neural_recording_time", 
                    ],
            columns=[
                pts_column,  
                neural_rectime_column,
            ],
        )
        self.nwbfile.add_stimulus(stimulus=watchlogs)
        self.logger.info(f"Populated cleaned watchlogs.")

    def populate_raw_watchlogs(self):
        pts_column = VectorData(
            name="pts",
            data=self.watchlog_info.raw_wl,
            description="presentation timestamps (pts, seconds) for each frame in the movie shown to the patient during the experiment (in seconds, movie play time)",
        )

        neural_rectime_column = VectorData(
            name="neural_recording_time",
            data=self.watchlog_info.raw_rec - self.watchlog_info.raw_rec[0],
            description="neural recording system timestamp (milliseconds) for each frame in the movie shown to the patient during the experiment, corresponds to the pts",
        )

        watchlogs = DynamicTable(
            name="raw_watchlogs",
            description=("Log of movie presentation to participant. Raw watchlog file, for data record purposes. "
                        "Lookup table containing: frame times (presentation time stamps, seconds) and the corresponding timestamp from the neural recording system (milliseconds)."),
            colnames=[
                "pts",
                "neural_recording_time", 
                    ],
            columns=[
                pts_column,  
                neural_rectime_column,
            ],
        )
        self.nwbfile.add_stimulus(stimulus=watchlogs)
        self.logger.info(f"Populated raw watchlogs")
    
    def save_nwbfile(self, save_path=None):
        if save_path is None:
            save_path = Path(self.subject_dir, f"sub{self.sub_id}.nwb")

        io = NWBHDF5IO(save_path, mode="w")
        io.write(self.nwbfile)
        io.close()

        complete_dataset_path = Path(self.save_dir, f"sub{self.sub_id}.nwb")
        shutil.copy(save_path, complete_dataset_path)
        self.logger.info(f"NWB file saved! \n")

def generate_nwb(patient_id, sub_id, data_dir, save_dir):
    patNWB = PatientNWB(patient_id=patient_id, sub_id=sub_id, data_dir=data_dir, save_dir=save_dir)
    patNWB.populate_subject()
    patNWB.populate_movie_binning_data()
    patNWB.populate_movie_annotations()
    patNWB.populate_movie_pts()
    patNWB.populate_electrode_groups()
    patNWB.populate_micro_channels()
    patNWB.populate_units()
    patNWB.populate_aligned_annotations()
    patNWB.populate_base_annotations()
    patNWB.populate_clean_watchlogs()
    patNWB.populate_raw_watchlogs()
    patNWB.save_nwbfile()

if __name__ == "__main__":
    data_dir = Path("/media/al/refractored_data/patient_data")
    save_dir = Path("/media/al/movies_dataset_nwb")

    patient_id = 1 #
    sub_id = 1
    generate_nwb(patient_id=patient_id, sub_id=sub_id, data_dir=data_dir, save_dir=save_dir)