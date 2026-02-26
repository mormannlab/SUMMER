## [17 feb 25 aid] misc data preprocessing helper functions

import re
from itertools import groupby

def atoi(text):
    return int(text) if text.isdigit() else text

def split_text(s):
    for k, g in groupby(s, str.isalpha):
        yield ''.join(g)

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]