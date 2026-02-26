
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d


class SpikeWidth:
    def __init__(self, upsample_ratio=1000, sr=32768, cutoff=0.65):
        """Init the spike width analysis with data directory and parameters.

        Args:
            subject_dir (str): path to the subject direction with single unit info stored as a dictionary in npy format.
            upsample_ratio (int, optional): Upsampling ratio for interpolation. Defaults to 1000.
            sr (int, optional): Sampling rate of the recording system/amplifier. Defaults to 32768.
            cutoff (float, optional): Threshold in milliseconds for determining if a cell is an interneuron. Values less than given float are labeled interneuron.
        """
        self.upsample_ratio = upsample_ratio
        self.sr = sr
        self.cutoff = cutoff

    def calculate_spike_width(self, waveform):
        """Calculate the spike width for a single waveform. 
        Classify neuron type based on the width (shorter than 0.65 is interneuron).
        
        Note: this implementation will extrapolate values for query points outside the 
        original range of indices (i.e. for values between 63 and 64) to match TP Reber's 
        matlab implementation.
        
        Args:
            waveform (np.ndarray): Mean waveform of the given unit in microvolts

        Returns:
            tuple: duration of the spike width in milliseconds, and the corresponding cell type 
        """
        nsamples = len(waveform)

        # upsample the waveform for interpolation
        x = np.arange(nsamples)
        xp = np.arange(1/self.upsample_ratio, nsamples, 1/self.upsample_ratio)
        time_axis = np.arange(0, len(xp)) / (self.sr * self.upsample_ratio)

        spline_interpolator = interp1d(x, waveform, kind="cubic", fill_value="extrapolate")
        waveform_upsampled = spline_interpolator(xp)

        max_amp_index = np.argmax(waveform_upsampled) # around ~19000, exact value varies due to smoothing
        time_axis = time_axis - time_axis[max_amp_index]

        min_post_peak = np.min(waveform_upsampled[time_axis > 0])
        min_post_peak_index = np.where(waveform_upsampled==min_post_peak)[0][0]

        spike_width = time_axis[min_post_peak_index] - time_axis[max_amp_index] # ms
        
        cell_type = "interneuron" if spike_width*1000 < 0.65 else "pyramidal"

        return spike_width*1000, cell_type

