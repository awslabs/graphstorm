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

    Generate example graph data using built-in datasets for node classifcation,
    node regression, edge classification and edge regression.
"""
from functools import partial
import glob
import json
import os

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import torch as th
import h5py

def read_index_json(data_file):
    """ Read the index from a JSON file.

    Each row is a JSON object that contains an index.

    Parameters
    ----------
    data_file : str
        The JSON file that contains the index.

    Returns
    -------
    Numpy array : the index array.
    """
    with open(data_file, 'r', encoding="utf8") as json_file:
        indices = []
        for line in json_file.readlines():
            indices.append(json.loads(line))
    return np.array(indices)

def write_index_json(data, data_file):
    """ Write the index to a json file.

    Parameters
    ----------
    data : Numpy array
        The index array
    data_file : str
        The data file where the indices are written to.
    """
    with open(data_file, 'w', encoding="utf8") as json_file:
        for index in data:
            json_file.write(json.dumps(int(index)) + "\n")

def read_data_json(data_file, data_fields):
    """ Read data from a JSON file.

    Each row of the JSON file represents a data record. Each JSON object
    is single-level dict without any nested dict structure.
    The function tries to extract a set of values from the data record.

    Parameters
    ----------
    data_file : str
        The JSON file that contains the data.
    data_fields : list of str
        The data fields that we will read data from a data record.

    Returns
    -------
    dict : map from data name to data
    """
    with open(data_file, 'r', encoding="utf8") as json_file:
        data_records = []
        for line in json_file.readlines():
            record = json.loads(line)
            data_records.append(record)

    data = {key: [] for key in data_fields}
    for record in data_records:
        for key in data_fields:
            assert key in record, \
                    f"The data field {key} does not exist in the record {record} of {data_file}."
            data[key].append(record[key])
    for key in data:
        data[key] = np.array(data[key])
    return data

def write_data_json(data, data_file):
    """ Write data to a json file.
    """
    records = []
    for key in data:
        if len(records) == 0:
            records = [{} for _ in range(len(data[key]))]
        assert len(records) == len(data[key])
        if data[key].shape == 1:
            for i, val in enumerate(data[key]):
                records[i][key] = val
        else:
            for i, val in enumerate(data[key]):
                records[i][key] = val.tolist()
    with open(data_file, 'w', encoding="utf8") as json_file:
        for record in records:
            record = json.dumps(record)
            json_file.write(record + "\n")

def read_data_parquet(data_file, data_fields=None):
    """ Read data from a parquet file.

    A row of a multi-dimension data is stored as an object in Parquet.
    We need to stack them to form a tensor.

    Parameters
    ----------
    data_file : str
        The parquet file that contains the data
    data_fields : list of str
        The data fields to read from the data file.

    Returns
    -------
    dict : map from data name to data.
    """
    table = pq.read_table(data_file)
    data = {}
    df_table = table.to_pandas()
    if data_fields is None:
        data_fields = list(df_table.keys())
    for key in data_fields:
        assert key in df_table, f"The data field {key} does not exist in {data_file}."
        val = df_table[key]
        d = np.array(val)
        # For multi-dimension arrays, we split them by rows and
        # save them as objects in parquet. We need to merge them
        # together and store them in a tensor.
        if d.dtype.hasobject and isinstance(d[0], np.ndarray):
            d = [d[i] for i in range(len(d))]
            d = np.stack(d)
        data[key] = d
    return data

def write_data_parquet(data, data_file):
    """ Write data in parquet files.

    Normally, Parquet cannot support multi-dimension arrays.
    This function splits a multi-dimensiion array into N arrays
    (each row is an array) and store the arrays as objects in the parquet file.

    Parameters
    ----------
    data : dict
        The data to be saved to the Parquet file.
    data_file : str
        The file name of the Parquet file.
    """
    arr_dict = {}
    for key in data:
        arr = data[key]
        assert len(arr.shape) == 1 or len(arr.shape) == 2, \
                "We can only write a vector or a matrix to a parquet file."
        if len(arr.shape) == 1:
            arr_dict[key] = arr
        else:
            arr_dict[key] = [arr[i] for i in range(len(arr))]
    table = pa.Table.from_arrays(list(arr_dict.values()), names=list(arr_dict.keys()))
    pq.write_table(table, data_file)

class HDF5Handle:
    """ HDF5 file handle

    This is to reference the HDF5 handle and close it when no one
    uses the HDF5 file.

    Parameters
    ----------
    f : HDF5 file handle
        The handle to access the HDF5 file.
    """
    def __init__(self, f):
        self._f = f

    def __del__(self):
        return self._f.close()


class HDF5Array:
    """ This is an array wrapper class for HDF5 array.

    The main purpose of this class is to make sure that we can close
    the HDF5 files when the array is destroyed.

    Parameters
    ----------
    arr : HDF5 dataset
        The array-like object for accessing the HDF5 file.
    handle : HDF5Handle
        The handle that references to the opened HDF5 file.
    """
    def __init__(self, arr, handle):
        self._arr = arr
        self._handle = handle
        self._out_dtype = None # Use the dtype of self._arr

    def __len__(self):
        return self._arr.shape[0]

    def __getitem__(self, idx):
        """ Slicing data from the array.

        Parameters
        ----------
        idx : Numpy array or Pytorch tensor or slice.
            The index.

        Returns
        -------
        Numpy array : the data from the HDF5 array indexed by `idx`.
        """
        if isinstance(idx, slice):
            return self._arr[idx]

        if isinstance(idx, th.Tensor):
            idx = idx.numpy()
        # If the idx are sorted.
        if np.all(idx[1:] - idx[:-1] > 0):
            arr = self._arr[idx]
        else:
            # There are two cases here: 1) there are duplicated IDs,
            # 2) the IDs are not sorted. Unique can return unique
            # IDs in the ascending order that meets the requirement of
            # HDF5 indexing.
            uniq_ids, reverse_idx = np.unique(idx, return_inverse=True)
            arr = self._arr[uniq_ids][reverse_idx]

        if self._out_dtype is not None:
            arr = arr.astype(self._out_dtype)
        return arr

    def to_tensor(self):
        """ Return Pytorch tensor.
        """
        arr = th.tensor(self._arr)
        if self._out_dtype is not None:
            if self._out_dtype is np.float32:
                arr = arr.to(th.float32)
            elif self._out_dtype is np.float16:
                arr = arr.to(th.float16)
        return arr

    def to_numpy(self):
        """ Return Numpy array.
        """
        res = self._arr[:]
        if self._out_dtype is not None:
            res = res.astype(self._out_dtype)
        return res

    def astype(self, dtype):
        """ Set the output dtype.

        Parameters
        ----------
        dtype: numpy.dtype
            Output dtype
        """
        self._out_dtype = dtype
        return self

    @property
    def shape(self):
        """ The shape of the HDF5 array.
        """
        return self._arr.shape

    @property
    def dtype(self):
        """ The data type of the HDF5 array.
        """
        return self._arr.dtype

def read_data_hdf5(data_file, data_fields=None, in_mem=True):
    """ Read the data from a HDF5 file.

    If `in_mem` is False, we don't read data into memory.

    Parameters
    ----------
    data_file : str
        The parquet file that contains the data
    data_fields : list of str
        The data fields to read from the data file.
    in_mem : bool
        Whether to read the data into memory.

    Returns
    -------
    dict : map from data name to data.
    """
    data = {}
    f = h5py.File(data_file, "r")
    handle = HDF5Handle(f)
    data_fields = data_fields if data_fields is not None else f.keys()
    for name in data_fields:
        assert name in f, f"The data field {name} does not exist in the hdf5 file."
        data[name] = f[name][:] if in_mem else HDF5Array(f[name], handle)
    return data

def write_data_hdf5(data, data_file):
    """ Write data into a HDF5 file.

    Parameters
    ----------
    data : dict
        The data to be saved to the Parquet file.
    data_file : str
        The file name of the Parquet file.
    """
    with h5py.File(data_file, "w") as f:
        for key, val in data.items():
            arr = f.create_dataset(key, val.shape, dtype=val.dtype)
            arr[:] = val

def _parse_file_format(conf, is_node, in_mem):
    """ Parse the file format blob

    Parameters
    ----------
    conf : dict
        Describe the config for the node type or edge type.
    is_node : bool
        Whether this is a node config or edge config
    in_mem : bool
        Whether or not to read the data in memory.

    Returns
    -------
    callable : the function to read the data file.
    """
    fmt = conf["format"]
    assert 'name' in fmt, "'name' field must be defined in the format."
    if is_node and "node_id_col" in conf:
        keys = [conf["node_id_col"]]
    elif is_node:
        keys = []
    elif "source_id_col" in conf and "dest_id_col" in conf:
        keys = [conf["source_id_col"], conf["dest_id_col"]]
    else:
        keys = []
    if "features" in conf:
        for feat_conf in conf["features"]:
            assert "feature_col" in feat_conf, "A feature config needs a feature_col."
            keys.append(feat_conf["feature_col"])
    if "labels" in conf:
        for label_conf in conf["labels"]:
            if "label_col" in label_conf:
                keys.append(label_conf["label_col"])
    if fmt["name"] == "parquet":
        return partial(read_data_parquet, data_fields=keys)
    elif fmt["name"] == "json":
        return partial(read_data_json, data_fields=keys)
    elif fmt["name"] == "hdf5":
        return partial(read_data_hdf5, data_fields=keys, in_mem=in_mem)
    else:
        raise ValueError('Unknown file format: {}'.format(fmt['name']))

parse_node_file_format = partial(_parse_file_format, is_node=True)
parse_edge_file_format = partial(_parse_file_format, is_node=False)

def get_in_files(in_files):
    """ Get the input files.

    The input file string may contains a wildcard. This function
    gets all files that meet the requirement.

    Parameters
    ----------
    in_files : a str or a list of str
        The input files.

    Returns
    -------
    a list of str : the full name of input files.
    """
    # If the input file has a wildcard, get all files that matches the input file name.
    if '*' in in_files:
        in_files = glob.glob(in_files)
    # This is a single file.
    elif not isinstance(in_files, list):
        in_files = [in_files]

    # Verify the existence of the input files.
    for in_file in in_files:
        assert os.path.isfile(in_file), \
                f"The input file {in_file} does not exist or is not a file."
    in_files.sort()
    return in_files
