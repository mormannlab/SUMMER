import numpy as np

def get_index_nearest_timestamp_in_vector(vector, timestamp):
    """
    This function returns the index of the value closest to 'timestamp' in 'vector'
    :param vector: array
        vector which shall be searched for 'timestamp'
    :param timestamp: float
        timestamp, which shall be searched
    :return int index
    """
    return (np.abs(np.array(vector) - timestamp)).argmin()


def make_label_from_start_stop_times(values, start_times, stop_times, ref_vec, default_value=0):
    """
    This function takes a vector with tuples with start and stop times and converts it to the default label
    :param ref_vec: reference vector, e.g. either PTS of movie or neural recording time of patient
    :param values: vector with all values
    :param start_times: vector with all start_times of segments
    :param stop_times: vector with all stop times of segments
    :param default_value: default value of label, which shall be added to all gaps in start stop times
    :return: label (0 and 1 for the length of the movie)
    """
    if not (len(values) == len(start_times) == len(stop_times)):
        print("vectors values, starts and stops have to be the same length")
        return -1
    
    default_label = [default_value] * len(ref_vec)
    
    for i in range(len(values)):
        start_index_in_default_vec = get_index_nearest_timestamp_in_vector(np.array(ref_vec), start_times[i])
        end_index_in_default_vec = get_index_nearest_timestamp_in_vector(np.array(ref_vec), stop_times[i])

        default_label[start_index_in_default_vec:(end_index_in_default_vec+1)] = \
            [int(values[i])]*(end_index_in_default_vec - start_index_in_default_vec + 1)

    return default_label