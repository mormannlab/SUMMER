"""
Training script for DHV NWB data using PyTorch Lightning.
Uses patient 14 NWB file and a small two-layer linear model.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path when running script directly (e.g. python data_loading/train.py)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from data_loading.dataloader import DHVDataModule
from data_loading.model import LinearNextBin


# Default data directory (devcontainer link to data folder)
DEFAULT_DATA_DIR = "/data"
DEFAULT_NWB_PATH = None  # unused when patient_ids/data_dir are used

def train(
    nwb_path=None,
    data_dir=None,
    patient_ids=None,
    bin_length=40,
    batch_size=32,
    max_epochs=20,
    hidden_size=64,
    lr=1e-3,
    val_fraction=0.1,
    num_workers=0,
    output_dir=None,
    devices=None,
    buffer=None,
    fold=None,
    sequence=(3, 1),
):
    output_dir = output_dir or "./lightning_logs"
    data_dir = data_dir or DEFAULT_DATA_DIR
    sequence = tuple(sequence)

    common_dm_kw = dict(
        bin_length=bin_length,
        batch_size=batch_size,
        num_workers=num_workers,
        val_fraction=val_fraction,
        buffer=buffer,
        fold=fold,
        sequence=sequence,
    )

    if patient_ids is not None and len(patient_ids) > 0:
        datamodule = DHVDataModule(
            data_dir=data_dir,
            patient_ids=patient_ids,
            **common_dm_kw,
        )
    else:
        nwb_path = nwb_path or str(Path(data_dir) / "sub14.nwb")
        datamodule = DHVDataModule(
            nwb_path=nwb_path,
            data_dir=data_dir,
            **common_dm_kw,
        )
    datamodule.setup()
    n_units = datamodule.n_units
    if n_units is None:
        raise RuntimeError("DataModule did not set n_units; check NWB path and data loading.")

    # summarize the split of train, val and test, how many samples are in each?
    print(f"Train: {len(datamodule.train_dataset)} samples")
    print(f"Val: {len(datamodule.val_dataset)} samples")
    print(f"Test: {len(datamodule.test_dataset)} samples")
    # specify information about split loaded
    print(f"Split information: buffer={buffer}, fold={fold}, sequence={sequence}")

    model = LinearNextBin(n_units=n_units, hidden_size=hidden_size, lr=lr)
    logger = CSVLogger(save_dir=output_dir, name="dhv")
    trainer_kw = dict(max_epochs=max_epochs, logger=logger, enable_progress_bar=True)
    if devices is not None:
        trainer_kw["devices"] = devices
        trainer_kw["accelerator"] = "gpu" if devices > 0 else "auto"
    trainer = pl.Trainer(**trainer_kw)
    trainer.fit(model, datamodule=datamodule)
    if datamodule.test_dataloader() is not None:
        trainer.test(model, datamodule=datamodule)
    return trainer, model


def main():
    parser = argparse.ArgumentParser(description="Train LinearNextBin on DHV NWB data (single or multiple patients).")
    parser.add_argument("--nwb_path", type=str, default=None, help="Path to a single NWB file (e.g. sub14.nwb). Ignored if --patient_ids is set.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing sub{id}.nwb files (default: /data).")
    parser.add_argument("--patient_ids", type=str, nargs="*", default=None, metavar="ID", help="Patient IDs to load and concatenate (e.g. 14 20). Uses data_dir. Default: single patient from nwb_path or sub14.")
    parser.add_argument("--bin_length", type=float, default=40, help="Bin length in ms")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of data for validation (ignored if --buffer/--fold are set).")
    parser.add_argument("--buffer", type=int, default=None, metavar="SEC", help="Buffer size in seconds for train/val/test split. One of 5,10,15,20,25,30,35,40,45,50,55. Requires --fold.")
    parser.add_argument("--fold", type=int, default=None, choices=[1, 2, 3, 4, 5], help="Which fold is validation (1-5). Requires --buffer. Enables test set.")
    parser.add_argument("--sequence", type=float, nargs=2, default=[3.0, 1.0], metavar=("PAST", "FUTURE"), help="Sequence (past_sec, future_sec) for buffer split; default 3 1.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./lightning_logs")
    parser.add_argument("--devices", type=int, default=None, help="Number of GPUs to use (default: all visible). Use 1 for single GPU.")
    args = parser.parse_args()

    # Optional: allow comma-separated IDs (e.g. --patient_ids 14,20)
    patient_ids = args.patient_ids
    if patient_ids is not None and len(patient_ids) == 1 and "," in patient_ids[0]:
        patient_ids = [x.strip() for x in patient_ids[0].split(",")]

    train(
        nwb_path=args.nwb_path,
        data_dir=args.data_dir,
        patient_ids=patient_ids,
        bin_length=args.bin_length,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        hidden_size=args.hidden_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        devices=args.devices,
        buffer=args.buffer,
        fold=args.fold,
        sequence=tuple(args.sequence),
    )


if __name__ == "__main__":
    main()
