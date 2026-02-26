%% collect_mad_values.m
% -------------------------------------------------------------------------
% Collects median absolute deviations (MAD) from bandpass-filtered neural data
% for each channel in a set of recording sessions.
%
% Main purpose of the script is to give method used to obtain the median absolute deviations
% used to evaluate the peak signal-to-noise ratio of the spike sorted data.
% Some patient-related meta-data is withheld from the dataset release for privacy purposes. 
%
% For each session:
%   - Loads channel information and movie timing metadata.
%   - Restricts continuous data to the movie viewing interval.
%   - Bandpass filters the data (default: 300–3000 Hz).
%   - Calculates the median absolute deviation (MAD) per channel, scaled
%     to be comparable to the standard deviation.
%   - Logs results to text files and saves results in a .mat file.
%
% NOTES:
%   - The raw bandpassed data is *not* saved.
%   - Script assumes access to patient-specific metadata (not included). 
%   - Specific handling is included for inconsistencies in channel metadata files.
%
% INPUTS:
%   - movie_meta_data.mat: must contain `patient_id`, `movie_onset`, and
%     `movie_offset` variables. Not included in the dataset.
%   - Channel information per session (e.g., 'channelnames.mat' or 'channels.mat').
%   - Continuous data retrievable via `get_continuous_data` function.
%
% PARAMETERS:
%   - scale_factor: 1.4826 (to make MAD comparable to standard deviation).
%   - bandpass_values: [300 3000] Hz.
%
% OUTPUTS:
%   - median_abs_deviations.mat per session.
%   - channelwise_mad_values.txt per session.
%   - bandpassMadLog_DATE.txt per session.
%
% REQUIREMENTS:
%   - MATLAB Signal Processing Toolbox (for `bandpass` function).
%   - Custom functions: `get_continuous_data`, `median_absolute_deviation`.
%
% AUTHOR: Alana Darcher
% DATE: 15 January 2025
% -------------------------------------------------------------------------

clear;
clc;

% not included in public release, contains patient data
load("movie_meta_data.mat");

scale_factor = 1.4826; % for calculation of MAD - makes comparable to std
bandpass_values = [300 3000]; % Hz
verbose = true; % Set to false to suppress frequent console output

session_list = [
	"sub1",
	"sub2",
	"sub3",
	"sub4",
	"sub5",
	"sub6",
	"sub7",
	"sub8",
	"sub9",
	"sub10",
	"sub11",
	"sub12",
	"sub13",
	"sub14",
	"sub15",
	"sub16",
	"sub17",
	"sub18",
	"sub19",
	"sub20",
	"sub21",
	"sub22",
	"sub23",
	"sub24",
	"sub25",
	"sub26",
	"sub27",
	"sub28",
	"sub29",
    ];

% for each session
for scr_idx=1:length(session_list)

    session_name = session_list{scr_idx};
    pat = str2double(session_name(1:3));
    indexer = patient_id == pat;

    mv_on = movie_onset(indexer);
    mv_off = movie_offset(indexer);

    path_screening = sprintf('/data/%s',session_name);
    
    if verbose
	disp(path_screening);
	cd(path_screening);
    end 
    
    try
    	load(fullfile(path_screening, 'channelnames.mat'));
        nchans = length(chnname);

        if sum(startsWith(chnname, "u")) > 0
            disp("channelnames.mat includes all channels, not just micros");
            left = startsWith(chnname, "L");
            right = startsWith(chnname, "R");
            nchans = sum(left) + sum(right);
        end

    catch 
        load(fullfile(path_screening, 'channels.mat'));
        nchans = length(channels);
        % Not implemented: 
        % checking for only micros in the channel name txt file. 
    end
    
    if nchans <= 0
        warning('No valid channels found for session %s. Skipping...', session_name);
        continue;
    end
    
    % workaround - at least one session's channelnames.mat contains 
    % more than just the micro channels.
    
    MAD_channels = NaN(nchans, 1);
    fprintf("   Number of channels: %d\n", nchans);

    % Initialize log files
    try
        if verbose
            fileID_mad = fopen("channelwise_mad_values.txt", "w");
            fprintf(fileID_mad, 'channelnr,mad_value\n');
            if fileID_mad == -1
                error("Can't open MAD file.");
            end

            date = datestr(datetime('now'), 'yyyy-mm-dd');
            logname = sprintf("bandpassMadLog_%s.txt", date);
            fileID_log = fopen(logname, 'w');
            if fileID_log == -1
                error("Can't open log file.");
            end
        end
    catch
        error('Failed to open log files.');
    end
    
    % for each channel in a session
    for c=1:nchans
    
        if verbose
            disp(c);
        end

        % bandpass 
        try
            [catsamples, catx, sr, chunks] = get_continuous_data(c);
            if verbose
	       disp("    Resizing to movie section..")
            end
                        
            time_interval = 1 / sr;
            sample_start = catx(1, 1);
            sample_end = catx(1, end);
            
            index_mv_on = fix(((mv_on - sample_start) / 1e6) * sr);
            index_mv_off = length(catx) - fix(((sample_end - mv_off) / 1e6) * sr);

            % quick check for plausiblity
            nm_movie_samples = index_mv_off - index_mv_on;
            duration_movie = (nm_movie_samples / sr) / 60;
            try 
                assert(duration_movie > 80, 'Movie duration is shorter than movie runtime.');
            catch
                warning('Number of samples is fewer than expected for the movie runtime.');
            end
      
            restricted_samples = catsamples(1, index_mv_on:index_mv_off);
            try 
                assert(length(restricted_samples) > 1, 'Something weird with the movie indexing.');
            catch
                warning('Something weird with the movie indexing.');
            end
      
            tic
            if verbose
                disp("    Bandpassing...");
            end
            bandpassed_samples = bandpass(restricted_samples,[bandpass_values(1), bandpass_values(2)],sr);
            toc
            fprintf("   channel %d  of session %s bandpassed\n", c, session_name);
            
            % calculate the median absolute deviation
            mad_m = median_absolute_deviation(bandpassed_samples, scale_factor);
            if verbose
                fprintf("   Median abs deviation: %.3f\n", mad_m);
            end

            MAD_channels(c, 1) = mad_m;
            fprintf(fileID_mad, '%d,%.5f\n', c, mad_m);

        catch ME
            disp(ME.message);
            fprintf(fileID_log, '%s channel %d did not run correctly. Error: %s\n', path_screening, c, ME.message);
            break
        end
    end
    
    % Check if there are any valid MAD values before saving
    if all(isnan(MAD_channels))
        warning('All MAD values are NaN for session %s. Not saving file.', session_name);
    else
        save_name = sprintf("%s/median_abs_deviations_testfornans.mat", path_screening);
        save(save_name, 'MAD_channels', '-v7.3');
    end

    fclose(fileID_mad);
    fclose(fileID_log);
end

disp("done");

