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

def wrapper_hd(frame_number):
    """
    Wrapper function that converts the original movie frame number to the corresponding number of frame in the hd version.
    """

    # the original version has 125 743 movie frames
    assert (0 <= frame_number <= 125743), "frame_number must be in range [0, 125743)"

    if frame_number <= 27:
        new_frame_number = frame_number
    elif frame_number > 27 and frame_number <= 97211:
        onset = 50
        diff = frame_number - 28
        skips = [2959, 8984, 15026, 21051, 27083, 33125, 39150, 45175, 51217, 57259, 63284, 68414, 75341, 81383, 87408, 93440]
        if frame_number < 2937:
            new_frame_number = onset + diff
        elif frame_number >= 2937 and frame_number < 93403:
            for i in range(len(skips)-1):
                start = skips[i]-50+28-i
                end = skips[i+1]-50+28-i
                print(f"start: {start}")
                print(f"end: {end}\n")
                if frame_number >= start and frame_number < end:
                    # if we are in a skip, jump to the end of the skip
                    new_frame_number = onset + diff + i + 1
        elif frame_number >= 93403:
            new_frame_number = onset + diff + len(skips)
    elif frame_number > 97211 and frame_number < 125730:
            onset = 108272
            diff = frame_number - 97212
            skips = [111532,117574,123599,129641]
            if frame_number < 100472:
                new_frame_number = onset + diff
            elif frame_number >= 100472 and frame_number < 118578:
                for i in range(len(skips)-1):
                    start = skips[i]-108272+97212-i
                    end = skips[i+1]-108272+97212-i
                    print(f"start: {start}")
                    print(f"end: {end}\n")
                    if frame_number >= start and frame_number < end:
                        # if we are in a skip, jump to the end of the skip
                        new_frame_number = onset + diff + i + 1
            elif frame_number >= 118578:
                new_frame_number = onset + diff + len(skips)
    # HD doesn"t contain last part, return -1
    elif frame_number >= 125730:
        new_frame_number = -1
        
    return new_frame_number