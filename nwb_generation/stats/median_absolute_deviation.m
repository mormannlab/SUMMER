function [mad_m] = median_absolute_deviation(data, scale_factor)
%MEDIAN_ABSOLUTE_DEVIATION Compute the median absolute deviation (MAD) of a signal.
%
%   mad_m = MEDIAN_ABSOLUTE_DEVIATION(data, scale_factor) calculates the
%   median absolute deviation of the input vector or matrix `data`.
%   NaN values are ignored in the computation. The resulting MAD is
%   scaled by the provided `scale_factor`, which can be used to make the
%   output comparable to other measures such as the standard deviation.
%
%   Based on:
%   Leys, C., Ley, C., Klein, O., Bernard, P. & Licata, L. (2013)
%   Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. 
%   Journal of Experimental Social Psychology 49, 764–766.
%   https://doi.org/10.1016/j.jesp.2013.03.013 
%
%   Inputs:
%       data         - A vector or matrix of numeric values, possibly containing NaNs.
%       scale_factor - A scalar value to scale the MAD (e.g., ~1.4826 to estimate std).
%
%   Output:
%       mad_m        - The scaled median absolute deviation of the input data.
%
%   Example:
%       data = [1, 2, 3, 4, 5, NaN];
%       mad = median_absolute_deviation(data, 1.4826);
%
%   See also NANMEDIAN, STD.

    m = nanmedian(data);
    disp(m);
    mad_diffs = abs(data - m);
    mad_m = nanmedian(mad_diffs) * scale_factor;

end
