
import numpy as np
import warnings
import sys

def create_patient_edges(patient_pts, patient_rec, movie_edges):
    len_edges = len(movie_edges)
    # Create edges in patient specific rec time
    patient_edges = []
    for i in range(len_edges):
        index = np.argwhere(patient_pts == movie_edges[i])
        if len(index)>1:
            index=index[1][0]
        elif len(index)==0:
            raise ValueError(f"Movie Edge {movie_edges[i]} could not be found!")
        index = int(index.squeeze())
        rec = patient_rec[index]
        patient_edges.append(rec)
    patient_edges=np.asarray(patient_edges)

    return patient_edges

def identify_pause_bins(movie_edges, patient_edges, patient_pts, patient_rec):
    counter=0
    indices_affected_bins=[]
    for idx in range(len(patient_pts)-1):
        if patient_pts[idx]==patient_pts[idx+1]:
            counter+=1
            exclude_pts=patient_pts[idx]
            # Find corresponding bin: Side 'left' (pause lies at end of bin, not start) and then substract 1 to get the start_edge of the the affected bin
            affected_bin_idx = np.searchsorted(movie_edges, exclude_pts, side='left')-1
            # Save the index of the affected bin - make sure it is not the very first or very last bin
            if affected_bin_idx+1 < len(movie_edges) and affected_bin_idx>=0:
                indices_affected_bins.append(affected_bin_idx)
    
    # HANDLING THE OUTLIER BINS
    pause_bin = []
    # Go reverse since adding new indices has an influence on the elements in indices_affected_bins
    indices_affected_bins = np.sort(indices_affected_bins)[::-1]
    for i in indices_affected_bins:
        a=i
        # The start of the bin is already added as edge, we will now split up the pause bin into three different bins, hence adding 2 more edges
        # Find the rec times that mark the begin and end of the pause in patient_rec
        diff_rec = np.diff(patient_rec)
        start_idx = np.argwhere(patient_pts==movie_edges[a]).squeeze()
        stop_idx = np.argwhere(patient_pts==movie_edges[a+1]).squeeze(axis=1)
        # Pause bin has two times the same pts time
        # Per construction, it can either lie at the end of a bin or in between
        # If it lies at the end, the length of stop_idx is equal to 2
        if len(stop_idx)==2:
            # Insert first rec times in the array patient_edges
            index1 = stop_idx[0]
            rec1 = patient_rec[index1]
            pts1 = patient_pts[index1]
            #print(f"Rec time for start of the pause: {rec1}")
            patient_edges=np.insert(patient_edges, a+1, rec1)
            movie_edges=np.insert(movie_edges, a+1, pts1)
            # Total Rec time of the three bins that belong to the pause bin
            rec_length= np.diff(patient_edges)[a+1]
            # Save information in pause_bin_end
            pause_bin.append((a+1, 'end'))
        else:
            stop_idx = stop_idx.squeeze()
            # find start and stop times of pause
            diff_partial = np.diff(patient_pts[start_idx:stop_idx+1])
            pause_idx = np.argwhere(diff_partial == 0).squeeze()
            pause_start = start_idx + pause_idx
            pause_end = start_idx + pause_idx + 1
            # Corresponding rec time
            corr_rec_start = patient_rec[pause_start]
            corr_rec_end = patient_rec[pause_end]
            corr_pts_start = patient_pts[pause_start]
            corr_pts_end = patient_pts[pause_end]
            assert (corr_pts_start==corr_pts_start), f"The extracted start and end pts of the pause have to be identical - but they are {corr_pts_start} and {corr_pts_end}."
            # Add both start and stop rec times to patient edges
            patient_edges=np.insert(patient_edges, a+1, corr_rec_end)
            patient_edges=np.insert(patient_edges, a+1, corr_rec_start)
            movie_edges=np.insert(movie_edges, a+1, corr_pts_start)
            movie_edges=np.insert(movie_edges, a+1, corr_pts_start)
            # Append index to delete to pause_bin_middle - a+1 is the start index of the pause bin, after deleting the surrounding bins need to be merged
            pause_bin.append((a+1, 'middle'))

    # SUMMARY OF DETECTED BINS
    pause_bin.sort()
            
    # Check that the extracted patient_edges are monotonically increasing
    assert np.all(np.diff(patient_edges) >= 0), 'Patient Edges are not monotonically increasing!'

    # CHECK LENGTH OF EXTRACTED BINS
    diff = np.diff(patient_edges)
    diff = np.diff(patient_edges)
    sorted_diff = np.sort(diff)
    
    counter=-1
    indices_affected_bins=[]
    for idx in range(len(movie_edges)-1):
        if movie_edges[idx]==movie_edges[idx+1]:
            counter+=1
            affected_bin_idx=idx
            exclude_pts=movie_edges[idx]
            # Change the corresponding index in pause_bin
            pause_bin[counter]=(affected_bin_idx, pause_bin[counter][1])

    return movie_edges, patient_edges, pause_bin

def subdivide_bins(movie_edges, patient_edges, pause_bin, subdivide, small_bin_length):
    if len(pause_bin)>0:
        subdivide_bins = [idx for idx in list(range(len(patient_edges)-1)) if idx not in list(list(zip(*pause_bin))[0])]
        len_pause_bin = len(list(list(zip(*pause_bin))[0]))
    else:
        subdivide_bins = list(range(len(patient_edges)-1))
        len_pause_bin = 0
    assert(len(subdivide_bins)==len(patient_edges)-1-len_pause_bin)
    assert np.all(np.diff(patient_edges) >= 0), 'Patient Edges are not monotonically increasing!'
    # go reversed order so that the insertion process does not affect the for loop
    for idx in subdivide_bins[::-1]:
        start_rec = patient_edges[idx]
        end_rec = patient_edges[idx+1]
        assert (end_rec - start_rec < 60 and end_rec - start_rec>10), f"The bin that should be splitted has length not in [20,60], but has length {end_rec-start_rec} at index {idx}."
        start_pts = movie_edges[idx]
        end_pts = movie_edges[idx+1]
        
        #Choose num=subdivide+1 so that we split the interval in subdivide pieces
        new_rec = np.linspace(start=start_rec, stop=end_rec, num=subdivide+1).tolist()[1:-1]
        new_pts = np.linspace(start=start_pts, stop=end_pts, num=subdivide+1).tolist()[1:-1]
        patient_edges = np.insert(patient_edges, idx+1, new_rec)
        movie_edges = np.insert(movie_edges, idx+1, new_pts)
        
        assert (np.all(np.diff(patient_edges[idx:idx+subdivide+1])>=0)), f"PATIENT EDGES: The changed part is not monotonic. The diffs are: {np.diff(patient_edges[idx:idx+subdivide+1])} "
        assert (np.all(np.diff(movie_edges[idx:idx+subdivide+1])>=0)), f"MOVIE EDGES: The changed part is not monotonic. The diffs are: {np.diff(movie_edges[idx:idx+subdivide+1])} "
    assert np.all(np.diff(patient_edges) >= 0), 'Patient Edges are not monotonically increasing!'
    # UPDATE TIME VARIABLE TO NEW BIN LENGTH
    time = small_bin_length
    print(f'We splitted all normal bins to length of {time}!\n')

    # CHECK LENGTH OF EXTRACTED BINS
    diff = np.diff(patient_edges)
    diff = np.diff(patient_edges)
    sorted_diff = np.sort(diff)

    # UPDATE PAUSE BIN INFORMATION
    counter=-1
    for idx in range(len(movie_edges)-1):
        if movie_edges[idx]==movie_edges[idx+1]:
            counter+=1
            affected_bin_idx=idx
            exclude_pts=movie_edges[idx]
            # Change the corresponding index in pause_bin
            pause_bin[counter]=(affected_bin_idx, pause_bin[counter][1])

    return movie_edges, patient_edges, pause_bin

def correct_binned_spikes_for_pause_bins(res, pause_bin, patient_edges):
    pause_bin.sort(reverse=True)
    for element in pause_bin:
        if element[1]=='end':
            # (1) Pause Bin
            delete_idx=element[0]
            # Delete the pause bin
            res = np.delete(res, delete_idx, axis=1)
            # Rec Length of new bin 
            rec_length= np.diff(patient_edges)[delete_idx-1]
            rec_length= np.diff(patient_edges)[delete_idx+1]
        elif element[1]=='middle':
            # (1) Pause Bin
            delete_idx=element[0]
            # Delete the pause bin
            res = np.delete(res, delete_idx, axis=1)
            # Rec Length of new bin 
            rec_length= np.diff(patient_edges)[delete_idx-1]
            rec_length= np.diff(patient_edges)[delete_idx+1]
            # Merge the surrounding bins to one bin
            new_bin = np.add(res[:, delete_idx-1], res[:, delete_idx])
            # Delete one bin
            res = np.delete(res, delete_idx, axis=1)
            # Replace new bin with old on index_delete_bin-1
            res = np.delete(res, delete_idx-1, axis=1)
            res = np.insert(res, delete_idx-1, new_bin, axis=1)
            # Check that the right bin has been deleted
            # Rec Length of new bin 
            rec_length= np.diff(patient_edges)[delete_idx-1]+np.diff(patient_edges)[delete_idx+1]
    
    return res

def bin_spikes_with_movie_edges(bin_length, spikes, movie_edges, patient_rec, patient_pts, subdivide, small_bin_length):
    """
    This function bins spikes, which are represented as a list of time points
    :param bin_length: Bin Lenghts wich will be used for binning
    :param spike_times: a vector (np.array) of time points of spikes
    :param output_edges: defining whether the edges used to bin the spikes should be outputted (necessary for tracking timepoints that are annotated during binning) (boolean, default = False)
    
    :return array with binned spikes
    """

    nb_units = len(spikes)
    res=[]
    # Check whether patient_rec is monotonically increasing
    assert np.all(np.diff(patient_rec) >= 0), 'Patient Edges are not monotonically increasing!'
    
    # CREATE EDGE VECTOR CONTAINING REC TIMES FOR PATIENT
    patient_edges = create_patient_edges(patient_pts, patient_rec, movie_edges)

    # CHECK THE GENERAL FORM OF THE EXTRACTED EDGES FOR MONOTONY AND SHAPE
    assert len(patient_edges)==len(movie_edges), 'Number of patient edges must be similar to number of movie edges!'
    assert np.all(np.diff(patient_edges) >= 0), 'Patient Edges are not monotonically increasing!\n'

    # DEFINE THE EXPECTED LENGTH FOR THE CHOSEN BIN LENGTH
    rate=0.04*1000
    n = int(bin_length/rate)
    time = n * 40
    
    # CHECK LENGTH OF EXTRACTED BINS AND SAVE THE OUTLIERS
    diff = np.diff(patient_edges)
    sorted_diff = np.sort(diff)

    # IDEENTIFY THE PAUSE BINS - all information is stored in variable pause_bin and movie edges/patient_edges is changed accordingly
    movie_edges, patient_edges, pause_bin = identify_pause_bins(movie_edges, patient_edges, patient_pts, patient_rec)

    # IF SUBDIVIDE WE NEED TO SUBDIVIDE EACH NORMAL INTO SEVERAL BINS - WE EXCLUDE ALL THE PAUSE BINS FROM THIS PROCESS
    if subdivide is not None:
        movie_edges, patient_edges, pause_bin = subdivide_bins(movie_edges, patient_edges, pause_bin, subdivide, small_bin_length)
        
    # BIN SPIKES FOR THE EDGES IN PATIENT_EDGES
    for i in range(nb_units):
        spikes_per_unit = spikes[i]
        hist, bins = np.histogram(spikes_per_unit, patient_edges)
        res.append(hist)
    # Convert final list to numpy array
    res = np.array(res)

    # DELETE THE BINS THAT ARE SAVED IN LIST PAUSE_BIN
    res = correct_binned_spikes_for_pause_bins(res, pause_bin, patient_edges)

    # FIN - DOUBLECHECK EXPECTED LENGHT OF BINNED SPIKES
    extra_indices_pause=0
    for (pause_idx, pause_type) in pause_bin:
        if pause_type=='end':
            extra_indices_pause+=1
        if pause_type=='middle':
            extra_indices_pause+=2
    exp_length = len(movie_edges)-1-extra_indices_pause

    assert (exp_length==len(res[0])), f'The length of the binned spikes is not equal to the expected length. It is: {len(res[0])} instead of {exp_length}.'

    return res

# Final function to call to get binned spikes for specific patient, movie_session

def bin_spikes_for_patient_with_movie_edges(nwb_data, patient_id, units, session_nr, bin_length, movie_edges=None, patient_rec=None, patient_pts=None):

    # Print Bin Length
    print(f"Patient: {patient_id}")
    print(f"Session Nr: {session_nr}")
    print(f'Bin Length: {bin_length}')

    # Load spike times for patient from nwb file
    df_nwb = nwb_data.units.to_dataframe()
    spike_times = []
    for unit in units:
        spikes_unit = np.array(df_nwb.loc[unit]["spike_times"])
        spike_times.append(spikes_unit)

    # If bin length < 40ms, we need to interpolate between the rec times for each frame
    if bin_length<40:
        subdivide = 40//bin_length
        print(f'Between each two frames, we need to subdivide the corresponding rec time by {subdivide}.')
        small_bin_length=bin_length
        bin_length=40
    else:
        subdivide=None
        small_bin_length=None

    # Bin Spikes
    binned_spikes = bin_spikes_with_movie_edges(bin_length, spike_times, movie_edges, patient_rec, patient_pts, subdivide, small_bin_length)

    return binned_spikes