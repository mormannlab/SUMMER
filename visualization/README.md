# Visualization

Notebooks and utilities for exploring and visualizing the SUMMER dataset and downstream analyses.

The folder is split into two main parts: `plot_code/` for exploratory analysis plots and `figure_generation/` for assembling publication-ready figures.

---

## Folder structure

```
visualization/
├── header_img/                 # images used in READMEs
├── plot_code/                  # exploratory analysis plots
│   ├── fonts/
│   ├── annotations_overview/
│   ├── data_overview/
│   ├── decoding_results/
│   │   ├── buffer_lengths/
│   │   ├── decoding_results/
│   │   └── subregions/
│   ├── responsive_units/
│   └── spike_sorting/
└── figure_generation/          # publication figure assembly
    ├── figure_annotations_vis/
    ├── figure_data_overview/
    ├── figure_decoding_results/
    ├── figure_responsive_units/
    └── figure_splits/
```

---

## `plot_code/`

Exploratory notebooks and supporting utilities, organized by analysis topic.

### Shared configuration

| File | Purpose |
|---|---|
| `config_colors.py` | Color palette and per-region colors (`A`, `H`, `EC`, `PHC`, `Other`) |
| `config_plot_params.py` | Global matplotlib settings (Helvetica font, axis/tick widths and label sizes) |
| `nwb_io.py` | Functions for loading spike times, waveforms, and annotations from NWB files |

### Analysis subfolders

**`annotations_overview/`** — Visualizes stimulus annotation timelines and label distributions across patients.

**`data_overview/`** — Summary plots of the dataset: recording coverage, unit counts, and patient-wise unit distribution across brain regions.

**`decoding_results/`** — Decoding performance plots, split into three views:
- `decoding_results/` — overall per-patient decoding performance
- `buffer_lengths/` — comparison across different temporal buffer lengths
- `subregions/` — performance broken down by brain subregion (A, H, EC, PHC)

**`responsive_units/`** — Single-unit response analysis: identifies stimulus-responsive neurons using cluster-permutation tests and plots their response functions.

**`spike_sorting/`** — Quality-control plots for the spike-sorting output.

---

## `figure_generation/`

Notebooks for assembling and exporting publication figures. Each subfolder corresponds to one figure and contains a `01_build_figure.ipynb` that loads analysis outputs, composes panels, and saves the figure as an SVG.

### `tasks.py`

[Invoke](https://www.pyinvoke.org/) tasks for converting saved SVGs into print-ready files:

```bash
# Convert a specific figure SVG → PDF → PNG (600 dpi, white background)
invoke convertpngpdf --fig decoding

# Available figure keys: overview, sorting, annotations, SU, decoding, annotations_vis, data_splits
```

Under the hood, `convertpngpdf` chains two steps:
1. `_convertsvg2pdf` — calls Inkscape to render each `.svg` in the figure's `fig/` folder to `.pdf`
2. `_convertpdf2png` — calls Inkscape to render each `.pdf` to `.png` at 600 dpi

---

## Notes

- `nwb_io.py` expects NWB files at `<data_dir>/sub{patient_id}.nwb` (default data directory: `/data`).
- `config_plot_params.py` loads a bundled Helvetica font from `plot_code/fonts/Helvetica.ttf`; ensure this file is present before importing.
- Figure SVGs are saved inside each figure folder under `fig/` (not tracked by git).
