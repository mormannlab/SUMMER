"""
PyTorch Dataset and Lightning DataModule for DHV NWB data.
Supports single or multiple patients; multi-patient data is concatenated along the units dimension.
Optional buffer/fold split for train/val/test with temporal separation.
"""

from pathlib import Path
import logging
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

# Buffer size (seconds) -> number of repeated parts in the session (for split layout)
BUFFER_NUMBER_REPEATED_PARTS = {
    5: 59, 10: 36, 15: 26, 20: 21, 25: 17, 30: 14, 35: 12, 40: 11, 45: 10, 50: 9, 55: 8,
}
VALID_BUFFERS = list(BUFFER_NUMBER_REPEATED_PARTS.keys())
VALID_FOLDS = [1, 2, 3, 4, 5]

from data_loading.nwb_loading.nwb_loading import (
    get_binned_spikes_nwb,
    get_unit_ids_for_patient_nwb,
)

# Default data directory (devcontainer link to data folder)
DEFAULT_NWB_DIR = "/data"
DEFAULT_PATIENT_ID = "14"


def _patient_and_dir_from_path(nwb_path):
    """Extract patient id and data directory from path like .../NWB_DATA/sub14.nwb."""
    path = Path(nwb_path)
    data_dir_override = str(path.parent)
    # stem is e.g. "sub14"
    patient_id = path.stem.replace("sub", "") if path.stem.lower().startswith("sub") else path.stem
    return patient_id, data_dir_override


def _normalize_patient_ids(patient_ids):
    """Ensure patient IDs are strings (e.g. 14 -> '14')."""
    if patient_ids is None:
        return None
    return [str(pid) for pid in patient_ids]


class DHVDataset(Dataset):
    """
    Dataset of binned spike counts for next-bin prediction.
    Each sample is (current_bin, next_bin) with shapes (n_units,) and (n_units,).
    Supports one or multiple patients; multiple patients are concatenated along the units axis.
    """

    def __init__(
        self,
        nwb_path=None,
        data_dir=None,
        patient_id=None,
        patient_ids=None,
        bin_length=40,
        brain_regions=None,
    ):
        """
        Args:
            nwb_path: Full path to a single NWB file (e.g. .../sub14.nwb). If given, overrides data_dir and patient_id.
            data_dir: Directory containing sub{patient_id}.nwb. Used if nwb_path is None.
            patient_id: Single patient ID string (e.g. "14"). Used if nwb_path is None and patient_ids is None.
            patient_ids: List of patient ID strings (e.g. ["14", "20"]) to load and concatenate along units. Overrides patient_id when set.
            bin_length: Bin length in ms for spike binning.
            brain_regions: Optional list of brain regions to restrict units; None = all units.
        """
        if nwb_path is not None:
            patient_id, data_dir = _patient_and_dir_from_path(nwb_path)
            patient_ids = [patient_id]
        else:
            data_dir = data_dir or DEFAULT_NWB_DIR
            patient_ids = _normalize_patient_ids(patient_ids)
            if patient_ids is None:
                patient_id = patient_id or DEFAULT_PATIENT_ID
                patient_ids = [str(patient_id)]

        self.data_dir_override = data_dir
        self.patient_ids = patient_ids
        self.bin_length = bin_length
        self.brain_regions = brain_regions

        # Load and bin each patient, then concatenate along units (axis 0)
        list_binned = []
        n_bins_ref = None
        for pid in self.patient_ids:
            units = get_unit_ids_for_patient_nwb(
                pid,
                brain_regions=brain_regions,
                data_dir_override=self.data_dir_override,
            )
            binned = get_binned_spikes_nwb(
                pid,
                units=units,
                bin_length=bin_length,
                data_dir_override=self.data_dir_override,
            )
            # (n_units_i, n_bins)
            if n_bins_ref is not None and binned.shape[1] != n_bins_ref:
                raise ValueError(
                    f"Patient {pid} has {binned.shape[1]} bins, but previous patient(s) have {n_bins_ref}. "
                    "All patients must have the same number of bins (same session/bin_length)."
                )
            n_bins_ref = binned.shape[1]
            list_binned.append(binned)

        self.binned_spikes = np.concatenate(list_binned, axis=0)
        print(f"Shape of binned spikes (patients {self.patient_ids}): {self.binned_spikes.shape}\n")

        # binned_spikes: (n_units_total, n_bins), float-like counts
        self.n_units = self.binned_spikes.shape[0]
        self.n_bins = self.binned_spikes.shape[1]

    def __len__(self):
        # One sample per consecutive pair (current, next)
        return max(0, self.n_bins - 1)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.binned_spikes[:, idx].astype("float32"))
        y = torch.from_numpy(self.binned_spikes[:, idx + 1].astype("float32"))
        return x, y


class DHVDataModule(pl.LightningDataModule):
    """Lightning DataModule for DHV NWB data (single or multiple patients).
    Supports either a simple fraction-based train/val split or a buffer/fold-based
    train/val/test split with temporal separation (fold 1–5, buffer 5–55 s in steps of 5).
    """

    def __init__(
        self,
        nwb_path=None,
        data_dir=None,
        patient_id=None,
        patient_ids=None,
        bin_length=40,
        brain_regions=None,
        batch_size=32,
        num_workers=0,
        val_fraction=0.1,
        buffer=None,
        fold=None,
        sequence=(3, 1),
    ):
        """
        Args:
            buffer: If set, use buffer/fold split. Must be in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55] (seconds).
            fold: When using buffer split, which fold is validation (1–5). Must be set together with buffer.
            sequence: (past_sec, future_sec) used to compute add_buffer in ms; default (3, 1).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["nwb_path", "data_dir", "patient_id", "patient_ids", "brain_regions"])
        self.nwb_path = nwb_path
        self.data_dir = data_dir or DEFAULT_NWB_DIR
        self.patient_id = patient_id
        self.patient_ids = _normalize_patient_ids(patient_ids)
        self.bin_length = bin_length
        self.brain_regions = brain_regions
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.buffer = buffer
        self.fold = fold
        self.sequence = tuple(sequence)
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        if buffer is not None:
            if buffer not in VALID_BUFFERS:
                raise ValueError(f"buffer must be one of {VALID_BUFFERS}, got {buffer}")
            if fold is None or fold not in VALID_FOLDS:
                raise ValueError(f"fold must be in {VALID_FOLDS} when buffer is set, got {fold}")
        if fold is not None and buffer is None:
            raise ValueError("fold requires buffer to be set")

    def _prepare_split_indices(self):
        """Compute train/val/test indices from buffer and fold. Call after self.dataset is set."""
        n = len(self.dataset)
        number_repeated_parts = BUFFER_NUMBER_REPEATED_PARTS[self.buffer]
        add_buffer_ms = (self.sequence[0] + self.sequence[1]) * 1000
        buffer_length_ms = self.buffer * 1000 + add_buffer_ms
        base = int(n / number_repeated_parts)
        buffer_bins = math.ceil(buffer_length_ms / self.bin_length)
        fold_len = int((base - 5 * buffer_bins) / 5)
        step = 1

        logger.info(
            "Data split: buffer=%s s, sequence=%s, n=%s, base=%s, buffer_bins=%s, fold_len=%s, fold=%s",
            self.buffer,
            self.sequence,
            n,
            base,
            buffer_bins,
            fold_len,
            self.fold,
        )

        indices_train = []
        indices_val = []
        indices_test = []

        for k in range(number_repeated_parts):
            i = k * base
            fold1 = list(range(i, i + fold_len, step))
            buffer1 = list(range(i + fold_len, i + fold_len + buffer_bins, step))
            fold2 = list(range(i + fold_len + buffer_bins, i + 2 * fold_len + buffer_bins, step))
            buffer2 = list(range(i + 2 * fold_len + buffer_bins, i + 2 * fold_len + 2 * buffer_bins, step))
            fold3 = list(range(i + 2 * fold_len + 2 * buffer_bins, i + 3 * fold_len + 2 * buffer_bins, step))
            buffer3 = list(range(i + 3 * fold_len + 2 * buffer_bins, i + 3 * fold_len + 3 * buffer_bins, step))
            fold4 = list(range(i + 3 * fold_len + 3 * buffer_bins, i + 4 * fold_len + 3 * buffer_bins, step))
            buffer4 = list(range(i + 4 * fold_len + 3 * buffer_bins, i + 4 * fold_len + 4 * buffer_bins, step))
            fold5 = list(range(i + 4 * fold_len + 4 * buffer_bins, i + 5 * fold_len + 4 * buffer_bins, step))
            buffer5 = list(range(i + 5 * fold_len + 4 * buffer_bins, i + 5 * fold_len + 5 * buffer_bins, step))

            fold_int = int(self.fold)
            if fold_int == 1:
                temp_test, temp_val = fold1, fold2
                temp_train = fold3 + buffer3 + fold4 + buffer4 + fold5
            elif fold_int == 2:
                temp_test, temp_val = fold2, fold3
                temp_train = fold1 + fold4 + buffer4 + fold5 + buffer5
            elif fold_int == 3:
                temp_test, temp_val = fold3, fold4
                temp_train = fold1 + buffer1 + fold2 + fold5 + buffer5
            elif fold_int == 4:
                temp_test, temp_val = fold4, fold5
                temp_train = fold1 + buffer1 + fold2 + buffer2 + fold3
            else:  # 5
                temp_test, temp_val = fold5, fold1
                temp_train = fold2 + buffer2 + fold3 + buffer3 + fold4

            indices_train.extend(temp_train)
            indices_val.extend(temp_val)
            indices_test.extend(temp_test)

        return indices_train, indices_val, indices_test

    def setup(self, stage=None):
        self.dataset = DHVDataset(
            nwb_path=self.nwb_path,
            data_dir=self.data_dir,
            patient_id=self.patient_id,
            patient_ids=self.patient_ids,
            bin_length=self.bin_length,
            brain_regions=self.brain_regions,
        )
        if self.buffer is not None and self.fold is not None:
            indices_train, indices_val, indices_test = self._prepare_split_indices()
            self.train_dataset = Subset(self.dataset, indices_train)
            self.val_dataset = Subset(self.dataset, indices_val)
            self.test_dataset = Subset(self.dataset, indices_test)
        else:
            n = len(self.dataset)
            n_val = max(1, int(n * self.val_fraction))
            n_train = n - n_val
            self.train_dataset = Subset(self.dataset, range(0, n_train))
            self.val_dataset = Subset(self.dataset, range(n_train, n))
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def n_units(self):
        if self.dataset is None:
            return None
        return self.dataset.n_units
