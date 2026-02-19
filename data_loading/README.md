# DHV data loading and training

PyTorch Lightning training pipeline for DHV NWB binned spike data: next-bin prediction with a small feedforward model. Supports single or multiple patients and optional buffer/fold-based train–val–test splits.

---

<p align="center">
  <img src="fig/header.png" width="500" alt="Data overview: stimulus frames and neural raster" />
</p>

*To generate the header image, place `frames.png` and `raster.png` in `data_loading/` and run:*  
`python data_loading/combine_header_images.py`

---

## Folder structure

```
data_loading/
├── __init__.py
├── README.md
├── requirements-train.txt       # pip dependencies for training
├── train.py                     # training script (CLI entrypoint)
├── model.py                     # LinearNextBin Lightning module
├── dataloader.py                # DHVDataset, DHVDataModule (single/multi-patient, buffer/fold splits)
├── combine_header_images.py     # stacks frames.png + raster.png → fig/header.png
├── test.py
├── fig/                         # generated figure (header.png)
└── nwb_loading/                 # NWB loading and binning
    ├── __init__.py
    ├── nwb_loading.py           # load NWB, get movie edges, get binned spikes
    └── binning.py               # spike binning with movie edges, pause handling
```

---

## Getting started

### Base environment

This pipeline was set up using **`nvcr.io/nvidia/pytorch:26.01-py3`** as the base image. It provides:

- **Python 3** with a standard scientific stack
- **PyTorch** with CUDA support for GPU training
- **NVIDIA libraries** (CUDA, cuDNN, NCCL) suitable for current NVIDIA GPUs

To run the code elsewhere, use an environment with compatible Python 3, PyTorch (and CUDA drivers if using GPUs), then install the project dependencies below.

---

### Install dependencies

In addition to the base image, install the Python dependencies for training:

```bash
pip install -r data_loading/requirements-train.txt
```

---

## Data location

- **Default data directory**: **`/data`** (inside the container).
- NWB files are expected as `sub{id}.nwb` (e.g. `/data/sub14.nwb`, `/data/sub20.nwb`).
- Override with `--data_dir` if your data lives elsewhere.

---

## Command-line arguments

| Feature              | CLI argument     | Options / notes                                      |
|----------------------|------------------|------------------------------------------------------|
| Data directory       | `--data_dir`     | Path to folder with `sub{id}.nwb` (default: `/data`) |
| Single NWB file      | `--nwb_path`     | Path to one NWB file (ignored if `--patient_ids` set)|
| Patient IDs          | `--patient_ids`  | Space- or comma-separated, e.g. `14 20` or `14,20`   |
| Bin length           | `--bin_length`   | ms, e.g. 40, 80, 100 (default: 40)                    |
| Batch size           | `--batch_size`   | Default: 32                                          |
| Max epochs           | `--max_epochs`   | Default: 20                                          |
| Hidden size          | `--hidden_size`  | Default: 64                                           |
| Learning rate        | `--lr`           | Default: 1e-3                                        |
| Val fraction         | `--val_fraction` | 0–1; used only when `--buffer`/`--fold` not set (default: 0.1) |
| Buffer (split)       | `--buffer`       | Seconds: 5,10,15,20,25,30,35,40,45,50,55. Requires `--fold`. |
| Fold (split)         | `--fold`         | 1–5; which fold is val (and test from same layout). Requires `--buffer`. |
| Sequence (split)     | `--sequence`     | Two floats: past_sec future_sec (default: `3 1`)     |
| DataLoader workers   | `--num_workers`  | Default: 0                                           |
| Output directory     | `--output_dir`   | Default: `./lightning_logs`                           |
| GPUs                 | `--devices`      | Number of GPUs (e.g. 1); default: all visible         |

---

## Output

- Logs and checkpoints are written under **`./lightning_logs`** by default (override with `--output_dir`).
- Subfolder structure: `lightning_logs/dhv/version_*/` (CSV logs, checkpoints if configured).

---

## How to run

**Single or multiple patients (fraction-based val split):**

```bash
python data_loading/train.py --patient_ids 14 20 --max_epochs 10
```

**Buffer/fold split (train / val / test with temporal separation):**

```bash
python data_loading/train.py --patient_ids 14 20 --buffer 20 --fold 3 --sequence 5 2 --max_epochs 10
```

**Single GPU:**

```bash
python data_loading/train.py --patient_ids 14 20 --devices 1 --max_epochs 10
```

---

## Data splitting (buffer / fold / sequence)

When you set **`--buffer`** and **`--fold`**, the pipeline uses a fixed train/val/test split with temporal gaps instead of a random fraction.

- **Buffer** (`--buffer`): Size of the temporal buffer in **seconds** (5, 10, 15, …, 55). The session is split into repeated “blocks”; each block is subdivided into 5 segments (folds) separated by buffers. Larger buffer → fewer blocks, more data per fold.
- **Fold** (`--fold`): Which of the 5 segments is used as **validation** (1–5). One segment is also reserved as **test**. The rest are used for training. Changing the fold rotates which part of the timeline is val/test.
- **Sequence** (`--sequence`): Pair **(past_sec, future_sec)** used to compute an extra time offset (in ms) added to the buffer length for binning. Default `3 1` (3 s past, 1 s future).

This avoids leakage between train/val/test by keeping gaps between segments and gives a deterministic split for reproducibility.
