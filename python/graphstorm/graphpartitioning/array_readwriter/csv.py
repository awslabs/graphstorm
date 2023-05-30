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

import pandas as pd
import pyarrow
import pyarrow.csv

from .registry import register_array_parser


@register_array_parser("csv")
class CSVArrayParser(object):
    """Wrapper class for CSV read/write operations.

    Argument
    --------
    delimiter: str
        Delimiter character to use when writing to a file.
    """

    def __init__(self, delimiter=","):
        self.delimiter = delimiter

    def read(self, path):
        """Read operation on a file.

        Argument
        --------
        path: str
            Path of a file from which to read.
        """
        logging.debug(
            "Reading from %s using CSV format with configuration %s"
            % (path, self.__dict__)
        )
        # do not read the first line as header
        read_options = pyarrow.csv.ReadOptions(autogenerate_column_names=True)
        parse_options = pyarrow.csv.ParseOptions(delimiter=self.delimiter)
        arr = pyarrow.csv.read_csv(
            path, read_options=read_options, parse_options=parse_options
        )
        logging.debug("Done reading from %s" % path)
        return arr.to_pandas().to_numpy()

    def write(self, path, arr):
        """Write operation to a file.

        Argument
        --------
        path: str
            Path of a file to write data.
        arr: numpy ndarray
            Numpy ndarray to write to a file.
        """
        logging.debug(
            "Writing to %s using CSV format with configuration %s"
            % (path, self.__dict__)
        )
        write_options = pyarrow.csv.WriteOptions(
            include_header=False, delimiter=self.delimiter
        )
        arr = pyarrow.Table.from_pandas(pd.DataFrame(arr))
        pyarrow.csv.write_csv(arr, path, write_options=write_options)
        logging.debug("Done writing to %s" % path)
