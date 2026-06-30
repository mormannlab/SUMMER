#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path specification for NWB-formatted data, save paths, etc.
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
NWB_data_dir = PROJECT_ROOT / "data" # modify path to data here