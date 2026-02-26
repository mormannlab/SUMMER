# SUMMER data loading and training

PyTorch Lightning training pipeline for SUMMER NWB binned spike data. Supports two tasks:

1. **Next-bin prediction** — predict the next bin’s spike counts from the current bin (regression).
2. **Frame-label prediction** — predict an annotation label for each bin from binned spikes (classification).

Single or multiple patients are supported, with optional buffer/fold-based train–val–test splits.

---

<p align="center">
  <img src="../visualization/header_img/data_overview.png" width="500" alt="Data overview: stimulus frames and neural raster" />
</p>


---

## Folder structure

```
ML_framework/
├── __init__.py
├── README.md
├── requirements-train.txt       # pip dependencies for training
├── train.py                     # training script (CLI entrypoint)
├── model.py                     # LinearNextBin (regression), LinearLabel (classification)
├── dataloader.py                # SUMMERDataset, SUMMERDatasetLabels, SUMMERDataModule, SUMMERDataModuleLabels
    └── nwb_loading/                 # NWB loading and binning
        ├── __init__.py
        ├── nwb_loading.py           # load NWB, movie edges, binned spikes, annotation labels
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
pip install -r ML_framework/requirements-train.txt
```

---

## Data location

- **Default data directory**: **`/data`** (inside the container).
- NWB files are expected as `sub{id}.nwb` (e.g. `/data/sub14.nwb`, `/data/sub20.nwb`).
- Override with `--data_dir` if your data lives elsewhere.

---

## Tasks

### 1. Next-bin prediction (`--task next_bin`, default)

- **Input:** binned spike counts for the current bin (one vector per bin).
- **Target:** binned spike counts for the **next** bin.
- **Model:** `LinearNextBin` — two linear layers, MSE loss.
- **Use case:** predict future neural activity from the current time bin.

### 2. Frame-label prediction (`--task label`)

- **Input:** binned spike counts for one bin (one vector per bin).
- **Target:** annotation label for that bin (e.g. presence of a scene or character).
- **Model:** `LinearLabel` — two linear layers, output size = number of classes, CrossEntropy loss; logs accuracy.
- **Use case:** decode stimulus annotations (e.g. “summer”, “alison”) from neural activity.
- **Labels** come from the NWB `movie_annotations_indicator_functions` processing module. They are aligned to bins using frame numbers from `movie_binning_info` when available (bin lengths 40, 80, 200, 480, 1000 ms), so label length matches the binned spike vector.

---

## Command-line arguments

| Feature              | CLI argument     | Options / notes                                      |
|----------------------|------------------|------------------------------------------------------|
| **Task**             | `--task`         | `next_bin` (default) or `label`                      |
| **Label name**       | `--label_name`   | Annotation name when `--task label` (e.g. `summer`). Default: `summer` |
| Data directory       | `--data_dir`     | Path to folder with `sub{id}.nwb` (default: `/data`) |
| Single NWB file      | `--nwb_path`     | Path to one NWB file (ignored if `--patient_ids` set)|
| Patient IDs          | `--patient_ids`  | Space- or comma-separated, e.g. `14 20` or `14,20`   |
| Bin length           | `--bin_length`   | ms, e.g. 40, 80 (default: 80). For labels, frame mapping exists for 40, 80, 200, 480, 1000. |
| Batch size           | `--batch_size`   | Default: 32                                          |
| Max epochs           | `--max_epochs`   | Default: 20                                          |
| Hidden size          | `--hidden_size`  | Default: 64                                           |
| Learning rate        | `--lr`           | Default: 1e-3                                        |
| Val fraction         | `--val_fraction` | 0–1; used only when `--buffer`/`--fold` not set (default: 0.1) |
| Buffer (split)       | `--buffer`       | Seconds: 4,8,12,16,20,24,28,32,36,40,44. Requires `--fold`. Default: 32. |
| Fold (split)         | `--fold`         | 1–5; which fold is val (and test from same layout). Requires `--buffer`. Default: 1. |
| Sequence (split)     | `--sequence`     | Two floats: past_sec future_sec (default: `3 1`)     |
| DataLoader workers   | `--num_workers`  | Default: 0                                           |
| Output directory     | `--output_dir`   | Default: `./lightning_logs`                           |
| GPUs                 | `--devices`      | Number of GPUs (e.g. 1); default: all visible         |

---

## Output

- Logs and checkpoints are written under **`./lightning_logs`** by default (override with `--output_dir`).
- **Next-bin:** `lightning_logs/dhv/version_*/` (CSV logs, checkpoints if configured).
- **Label:** `lightning_logs/dhv_label/version_*/` (train/val/test loss and accuracy).

---

## How to run

### Next-bin prediction (default)

**Single or multiple patients (fraction-based val split):**

```bash
python ML_framework/train.py --patient_ids 14 20 --max_epochs 10
```

**Buffer/fold split (train / val / test with temporal separation):**

```bash
python ML_framework/train.py --patient_ids 14 20 --buffer 20 --fold 3 --sequence 5 2 --max_epochs 10
```

### Frame-label prediction

**Default label `summer`, one or more patients:**

```bash
python ML_framework/train.py --task label --patient_ids 29 8 --max_epochs 10
```

**Another annotation (e.g. `alison`) with buffer/fold split:**

```bash
python ML_framework/train.py --task label --label_name alison --patient_ids 29 8 --buffer 10 --fold 1 --max_epochs 10
```

**Single GPU (either task):**

```bash
python ML_framework/train.py --patient_ids 14 20 --devices 1 --max_epochs 10
```

---

## Data splitting (buffer / fold / sequence)

When you set **`--buffer`** and **`--fold`**, the pipeline uses a fixed train/val/test split with temporal gaps instead of a random fraction.

- **Buffer** (`--buffer`): Size of the temporal buffer in **seconds** (4, 8, 12, …, 44). The session is split into repeated “blocks”; each block is subdivided into 5 segments (folds) separated by buffers. Larger buffer → fewer blocks, more data per fold.
- **Fold** (`--fold`): Which of the 5 segments is used as **validation** (1–5). One segment is also reserved as **test**. The rest are used for training. Changing the fold rotates which part of the timeline is val/test.
- **Sequence** (`--sequence`): Pair **(past_sec, future_sec)** used to compute an extra time offset (in ms) added to the buffer length for binning. Default `3 1` (3 s past, 1 s future).

This avoids leakage between train/val/test by keeping gaps between segments and gives a deterministic split for reproducibility.
