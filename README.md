# SUMMER — Single Unit activity during a Movie in the human Medial Temporal lobe via Electrophysiological Recordings

All code associated with the SUMMER dataset: NWB file generation, technical validation, movie stimulus synchronization, and result visualization. Implemented in Python and Matlab, with a conda environment file for reproducible setup.

A module for local database initialization is included for users who prefer a database-driven workflow, along with a tutorial for integrating it into existing pipelines. For the ML community, the repository provides a complete baseline training pipeline to go from download to model training with minimal configuration.

---

<p align="center">
  <img src="visualization/header_img/data_overview.png" width="500" alt="Data overview: stimulus frames and neural raster" />
</p>

---

## Folder structure

```
dhv_dev/
├── README.md
├── config_paths.py                  # shared path configuration
├── environment.yml                  # conda environment (with builds)
├── environment_no_builds.yml        # conda environment (no builds)
├── movie_wrapper/                   # wrappers for accessing the movie database
│   ├── README.md
│   └── wrapper.py
├── nwb_generation/                  # nwb file generators and associated stats code
│   ├── stats/                       # scripts for spike sorting metrics
│   ├── utils/                       # processing modules for nwb file creation
│   ├── README.md
│   ├── static_info.py
│   ├── write_nwb.py                 # classes for creating nwb files
│   ├── write_nwb.ipynb              # runner for building nwb files
├── visualization/                   # plot code and figure generation
│   ├── figure_generation/           # svgutils-based panel assembly
│   │   ├── figure_annotations_vis/
│   │   ├── figure_data_overview/
│   │   ├── figure_decoding_results/
│   │   ├── figure_responsive_units/
│   │   ├── figure_splits/
│   │   └── tasks.py                 # invoke tasks (e.g. convertpngpdf)
│   ├── header_img/                  # images used in the documentation
│   └── plot_code/                   # scripts and config for individual plots
│       ├── __init__.py
│       ├── config_colors.py
│       ├── config_plot_params.py
│       ├── nwb_io.py
│       ├── annotations_overview/
│       ├── data_overview/
│       ├── decoding_results/
│       ├── responsive_units/
│       ├── spike_sorting/
│       └── fonts/
└── ML_framework/                    # PyTorch Lightning training pipeline
    ├── README.md
    ├── requirements-train.txt
    ├── train.py                     # training script (CLI entrypoint)
    ├── model.py                     # LinearNextBin, LinearLabel models
    ├── dataloader.py                # datasets and data modules
    └── nwb_loading/                 # NWB loading and binning
        ├── nwb_loading.py
        └── binning.py
```

---

## NWB & Stats

The **`nwb_generation/`** folder contains utilities for building the NWB files from the original neural datasets. 

The **`stats/`** subdirectory includes the modules used to calculate the spike sorting metrics.
The `write_nwb.py` contains the classes used to build the various containers and elements of the NWB files. The `write_nwb.ipynb` combines these classes to build each file. 

Note: the nwb generation code is included for demonstartion purposes. 

---

## Movie wrapper

The **`movie_wrapper/`** folder contains utilities for synchronizing frame numbers across different versions of the movie stimulus.

The original movie has 125,743 frames. Because the DVD and HD releases differ in frame layout (chapter breaks, skipped frames), `wrapper.py` provides two functions to map any original frame number to its equivalent in each version:

- `wrapper_dvd(frame_number)` — converts an original frame number to the corresponding DVD frame number.
- `wrapper_hd(frame_number)` — converts an original frame number to the corresponding HD frame number.

For more details, see **[movie_wrapper/README.md](movie_wrapper/README.md)**.

---

## Plot code and figure generation

The **`visualization/`** folder is split into two parts:

### plot_code

Notebooks and scripts for creating individual analysis plots, organized by topic (data overview, spike sorting, responsive units, decoding results, annotations). Shared configuration files define the color palette and matplotlib style used across all plots.

### figure_generation

Notebooks for composing publication-ready figures from individual plot panels. Figures are saved as SVGs and can be converted to PDF/PNG using the provided [Invoke](https://www.pyinvoke.org/) tasks.

For more details, see **[visualization/README.md](visualization/README.md)**.

---

## ML framework

The **`ML_framework/`** folder folder provides a complete PyTorch Lightning training pipeline for SUMMER NWB binned spike data, supporting two tasks:

- **Next-bin prediction** — predict the next bin's spike counts from the current bin (regression).
- **Frame-label prediction** — predict an annotation label for each bin from binned spikes (classification).

Supports single or multiple patients and optional buffer/fold-based train–val–test splits. Default: `bin_length=80` ms, `buffer=40` s, `fold=1`.

All requirements and configuration needed to get started are included. A training run can be launched with a single command:

```bash
python ML_framework/train.py --patient_ids 14 20 --max_epochs 10
```

For a quick start guide, installation instructions, and full argument documentation, see **[ML_framework/README.md](ML_framework/README.md)**.

---

## Development setup

A pre-commit hook is configured to strip outputs from Jupyter notebooks before each commit, so that committed notebooks stay clean. To use it, install pre-commit and enable the hooks (once per clone):

```bash
pip install pre-commit
pre-commit install
```

After that, the hook runs automatically on `git commit`. If it strips outputs from any notebook, stage the updated files and commit again.
