"""
This file contains utility classes and functions for logging to stdout, stderr,
and to tensorboard.

This file is adopted from robomimic
https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/log_utils.py
"""
import sys
from tqdm import tqdm


class custom_tqdm(tqdm):
    """
    Small extension to tqdm to make a few changes from default behavior.
    By default tqdm writes to stderr. Instead, we change it to write
    to stdout.
    """
    def __init__(self, *args, **kwargs):
        assert "file" not in kwargs
        super(custom_tqdm, self).__init__(*args, file=sys.stdout, **kwargs)
