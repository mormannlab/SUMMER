# Data

Default location for the dataset when downloaded from DANDI, following the Quick Start Guide on the main README. 

Run `refactor_datafiles.py` before using the repo notebooks (see below). 

---

## Folder structure

```
data/
├── README.md
└── refactor_datafiles.py                       # re-organize the DANDI-downloaded data into a flat hierarchy
```

---

## Retrieving the data and organizing:

The complete set-up is given in the Quick Start Guide, ~/SUMMER/README.md. 

Assuming you have a conda environment set up, the steps to retrieve and organize the data are: 

1. Download to this directory: `dandi download DANDI:001616 --output-dir data` 
    Optionally: download elsewhere, and set the path in `~/SUMMER/config_path.py`.

2. Flatten the data to a single directory: `python3 refactor_datafiles.py`

