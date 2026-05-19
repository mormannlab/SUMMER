import subprocess
from pathlib import Path

import numpy as np


def wrapper_dvd(frame_number):
    """
    Wrapper function that converts the original movie frame number to the corresponding number of frame in the dvd version.
    """

    # the original version has 125 743 movie frames
    assert (0 <= frame_number <= 125743), "frame_number must be in range [0, 125743)"

    if frame_number <= 97211:
        new_frame_number = frame_number
    elif frame_number > 97211:
        onset = 108232
        diff = frame_number - 97212
        new_frame_number = onset + diff

    return new_frame_number


def inverse_wrapper_dvd(dvd_frame_number):
    """Invert wrapper_dvd: given a DVD frame number, return the paradigm frame number."""
    if dvd_frame_number <= 97211:
        return dvd_frame_number
    elif dvd_frame_number >= 108232:
        return 97212 + (dvd_frame_number - 108232)
    else:
        raise ValueError(
            f"DVD frame {dvd_frame_number} falls in the gap (97212–108231) "
            "that has no corresponding paradigm frame."
        )
