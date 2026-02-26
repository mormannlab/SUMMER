function [catsamples catx sr chunks] = get_continuous_data(channel)
%% get_continuous_data
% Reads in continuous neural data from an `.ncs` file, handles discontinuous timestamps,
% and returns the data samples, timestamps, sampling rate, and chunk information.
%
% Syntax:
%   [catsamples, catx, sr, chunks] = get_continuous_data(channel)
%
% Inputs:
%   channel: Either the channel number (numeric) or the file name (string) of the `.ncs` file.
%
% Outputs:
%   catsamples: A vector of concatenated data samples from the specified channel.
%   catx: A vector of concatenated time-stamps corresponding to the data samples.
%   sr: The calculated sampling rate based on the timestamps in the file.
%   chunks: Information about the chunks of continuous data, including gaps between recordings.
%
% Description:
%   This function reads an `.ncs` file, retrieves the header information (sampling frequency, record size, scale factor),
%   and checks for discrepancies between header-specified and calculated sampling rates. It reads the timestamp data
%   from the file, computes the sampling rate, and handles any discontinuities in the timestamps. It returns a vector
%   of data samples, the associated timestamps, and the calculated sampling rate. The function also handles chunking
%   for continuous data and performs scaling of the sample values to microvolts.
%
% Note:
%   - The file should have a valid `.ncs` format with timestamped data.
%   - The `get_continuous_chunks` function is used to identify gaps in the timestamp data.
%   - The `upsample_nlx_ts` function is used for timestamp upsampling within chunks.
%
% Example:
%   [data, timestamps, sampling_rate, chunk_info] = get_continuous_data(1);
%
% See also: get_continuous_chunks, upsample_nlx_ts


sr_tolerance = 1; % how many Hz can the header info be off to what is calculated from the timestamps
verbose = true; 
%% handle input
if isnumeric(channel)
    filename = sprintf('CSC%d.ncs', channel);
else
    filename = channel;
end
if ~exist(filename, 'file')
    error(sprintf('file %s not found ', filename));
end
if verbose
    fprintf('reading %s - ', filename);
    tic
end

%% get some info from file-header
header=textread(filename,'%s',43);
sr_hdr = str2num(header{find(strcmp('-SamplingFrequency', header))+1});
recsize_hdr = str2num(header{find(strcmp('-RecordSize', header))+1});
scale_factor_hdr = str2num(header{find(strcmp('-ADBitVolts', header))+1});

%% get timestamps and some header infos in file
f=fopen(filename,'r','l');
fseek(f,16384,'bof'); % Skip Header
ts=fread(f,inf,'int64',(4+4+4+2*512));

td = floor(median(diff(ts)));
sr = 512*1e6/td; 
err = sr_hdr - sr;
if abs(err) > 0 & abs(err) <= sr_tolerance
    warning(sprintf(['header sr (%d Hz) doesn''t match to sr calulated ' ...
                     'from timestamps in file (%d Hz) but is within ' ...
                     'tolerance (+/- %d HZ)'], sr_hdr, sr,sr_tolerance));
elseif abs(err) > sr_tolerance
    error('header sr doesn''t match to timestamps in file');
end
recsize=512*2+8+4+4+4;  %1044
assert(recsize == recsize_hdr);

%% find gaps in recording (discontinuous timestamps) 
[chunks chunkheader]= get_continuous_chunks(ts, 1e6);

%% read in data
for r = 1:size(chunks,1)
    fseek(f,16384+recsize*(chunks(r,1)-1)+8+4+4+4,'bof'); % put pointer to the first record to be read
    samples{r} = fread(f,512*chunks(r,3),...
                     '512*int16=>int16',8+4+4+4);

    % convert sample-values to1 mV
    samples{r} = double(samples{r}) .* scale_factor_hdr * 1e6;
    % upsample timestamps in chunk
    uts{r} = upsample_nlx_ts(ts(chunks(r,1):chunks(r,2)));
end

%% concatenate output
catsamples = [];
catx = [];
for g = 1:length(samples)
    catsamples = horzcat(catsamples, samples{g}');
    catx = horzcat(catx, uts{g});
end

if verbose
    fprintf('took %.2f seconds \n', toc);
end

