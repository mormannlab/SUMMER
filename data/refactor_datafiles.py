# Flatten NWB files from <data-directory>/001616/sub-*/ into data/, then remove 001616.

import shutil
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_paths import NWB_data_dir

dataset_dir = NWB_data_dir / "001616"

for nwb_file in dataset_dir.rglob("*.nwb"):
    shutil.copy2(nwb_file, NWB_data_dir / nwb_file.name)

shutil.rmtree(dataset_dir)