"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import logging

import numpy as np
from numpy.lib.format import open_memmap

from .registry import register_array_parser


@register_array_parser("numpy")
class NumpyArrayParser(object):
    """Wrapper class for numpy read/write operations."""

    def __init__(self):
        pass

    def read(self, path):
        """Read operation on a file.

        Argument
        --------
        path: str
            Path of a file from which to read.
        """
        logging.debug("Reading from %s using numpy format" % path)
        arr = np.load(path, mmap_mode="r")
        logging.debug("Done reading from %s" % path)
        return arr

    def write(self, path, arr):
        """Write operation to a file.

        Argument
        --------
        path: str
            Path of a file to write data.
        arr: numpy ndarray
            Numpy ndarray to write to a file.
        """
        logging.debug("Writing to %s using numpy format" % path)
        # np.save would load the entire memmap array up into CPU.  So we manually open
        # an empty npy file with memmap mode and manually flush it instead.
        new_arr = open_memmap(path, mode="w+", dtype=arr.dtype, shape=arr.shape)
        new_arr[:] = arr[:]
        logging.debug("Done writing to %s" % path)
