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
import logging

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import h5py
import pandas as pd

from .utils import HDF5Handle, HDF5Array


def read_index(split_info):
    """ Read the index from a JSON/parquet file.

    Parameters
    ----------
    split_info : dict
        Customized split information

    Returns
    -------
    tuple of numpy.ndarray
        Returns a tuple containing three numpy arrays:
        - First element: Data from the training split, if not available, [].
        - Second element: Data from the validation split, if not available, [].
        - Third element: Data from the test split, if not available, [].
        If the file extension is not '.json' or '.parquet', a ValueError is raised.
    """
    res = []
    for idx in ['train', 'valid', 'test']:
        if idx not in split_info:
            res.append([])
            continue
        if isinstance(split_info[idx], str):
            _, extension = os.path.splitext(split_info[idx])
        else:
            extensions = [os.path.splitext(path)[1] for path in split_info[idx]]
            assert len(set(extensions)) == 1, f"each file should be in the same format, " \
                                   f"but get {extensions}"
            extension = extensions[0]

        # Normalize the extension to ensure case insensitivity
        extension = extension.lower()

        # json files should be ended with .json and parquet files should be ended with parquet
        if extension == '.json':
            res.append(read_index_json(split_info[idx]))
        elif extension == '.parquet':
            # We should make sure there are multiple parquet files instead of one
            res.append(read_index_parquet(split_info[idx], split_info['column']))
        else:
            raise ValueError(f"Expect mask data format be one of parquet "
                             f"and json, but get {extension}")
    return res[0], res[1], res[2]


def expand_wildcard(data_files):
    """
    Expand the wildcard to the actual file lists.

    Parameters
    ----------
    data_files : list[str]
        The parquet files that contain the index.

    """
    expanded_files = []
    for item in data_files:
        if '*' in item:
            matched_files = glob.glob(item)
            assert len(matched_files) > 0, \
                f"There is no file matching {item} pattern"
            expanded_files.extend(matched_files)
        else:
            expanded_files.append(item)
    return expanded_files

def read_index_parquet(data_file, column):
    """
    Read the index from a parquet file.

    Parameters
    ----------
    data_file : str or list[str]
        The parquet file that contains the index.
    column: list[str]
        Column names on parquet which contain the index

    """
    if isinstance(data_file, str):
        data_file = [data_file]
    data_file = expand_wildcard(data_file)
    df_list = [pd.read_parquet(file) for file in data_file]
    df = pd.concat(df_list, ignore_index=True)

    if len(column) == 1:
        res_array = df[column[0]].to_numpy()
    elif len(df.columns) == 2:
        res_array = list(zip(df[column[0]].to_numpy(), df[column[1]].to_numpy()))
    else:
        raise ValueError("The Parquet file on node mask must contain exactly one column, "
                         "and on edge mask must contain exactly two columns.")

    return res_array

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
            parsed_line = json.loads(line)
            if isinstance(parsed_line, list):
                processed_item = tuple(parsed_line)
            else:
                processed_item = parsed_line

            indices.append(processed_item)
    return indices

def write_index_json(data, data_file):
    """ Write the index to a json file.

    Parameters
    ----------
    data : Numpy array
        The index array
    data_file : str
        The data file where the indices are written to.
    """
    if isinstance(data, np.ndarray):
        data = data.tolist()
    with open(data_file, 'w', encoding="utf8") as json_file:
        for index in data:
            json_file.write(json.dumps(index) + "\n")

def read_data_csv(data_file, data_fields=None, delimiter=','):
    """ Read data from a CSV file.

    Parameters
    ----------
    data_file : str
        The file that contains the data.
    data_fields : list of str
        The name of the data fields.
    delimiter : str
        The delimiter to separate the fields.

    Returns
    -------
    dict of Numpy arrays.
    """
    data = pd.read_csv(data_file, delimiter=delimiter)
    assert data.shape[0] > 0, \
        f"{data_file} has an empty data. The data frame shape is {data.shape}"

    if data_fields is not None:
        for field in data_fields:
            assert field in data, f"The data field {field} does not exist in the data file."
        return {field: data[field].to_numpy() for field in data_fields}
    else:
        return {field: data[field].to_numpy() for field in data}

def write_data_csv(data, data_file, delimiter=','):
    """ Write data to a CSV file.

    Parameters
    ----------
    data : dict of Numpy arrays
        The data arrays that need to be written to the CSV file.
    data_file : str
        The path of the data file.
    delimiter : str
        The delimiter that separates the fields.
    """
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(data_file, index=True, sep=delimiter)

def _pad_stack(arrs):
    max_len = max(len(arr) for arr in arrs)
    new_arrs = np.zeros((len(arrs), max_len), dtype=arrs[0].dtype)
    for i, arr in enumerate(arrs):
        new_arrs[i][:len(arr)] = arr
    return new_arrs

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
    assert len(data_records) > 0, \
        f"{data_file} is empty {data_records}."

    data = {key: [] for key in data_fields}
    for record in data_records:
        for key in data_fields:
            assert key in record, \
                    f"The data field {key} does not exist in the record {record} of {data_file}."
            record1 = np.array(record[key]) if isinstance(record[key], list) else record[key]
            data[key].append(record1)
    for key in data:
        if isinstance(data[key][0], np.ndarray):
            data[key] = _pad_stack(data[key])
        else:
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
        if isinstance(data[key], np.ndarray):
            if data[key].shape == 1:
                for i, val in enumerate(data[key]):
                    records[i][key] = val
            else:
                for i, val in enumerate(data[key]):
                    records[i][key] = val.tolist()
        elif isinstance(data[key], list):
            if isinstance(data[key][0], np.ndarray):
                for i, val in enumerate(data[key]):
                    records[i][key] = val.tolist()
            else:
                for i, val in enumerate(data[key]):
                    records[i][key] = val
        else:
            raise ValueError("Invalid data.")
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
    assert df_table.shape[0] > 0, \
        f"{data_file} has an empty data. The data frame shape is {df_table.shape}"

    if data_fields is None:
        data_fields = list(df_table.keys())
    for key in data_fields:
        assert key in df_table, f"The data field {key} does not exist in {data_file}."
        d = df_table[key].to_numpy()

        # For multi-dimension arrays, we split them by rows and
        # save them as objects in parquet. We need to merge them
        # together and store them in a tensor.
        if d.dtype.hasobject and isinstance(d[0], np.ndarray):
            new_d = [d[i] for i in range(len(d))]
            try:
                # if each row has the same shape
                # merge them together
                d = np.stack(new_d)
            except Exception: # pylint: disable=broad-exception-caught
                # keep it as an ndarry of ndarrys
                # It may happen when loading hard negatives for hard negative transformation.
                logging.warning("The %s column of parquet file %s has " \
                    "variable length of feature, it is only suported when " \
                    "transformation is a hard negative transformation", key, data_file)
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
        assert np.prod(arr.shape) < 2 * 1024 * 1024 * 1024, \
                "Some PyArrow versions do not support a column with over 2 billion elements."
        assert len(arr.shape) == 1 or len(arr.shape) == 2, \
                "We can only write a vector or a matrix to a parquet file."
        if len(arr.shape) == 1:
            arr_dict[key] = arr
        else:
            arr_dict[key] = [arr[i] for i in range(len(arr))]
    table = pa.Table.from_arrays(list(arr_dict.values()), names=list(arr_dict.keys()))
    pq.write_table(table, data_file)

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

def stream_dist_tensors_to_hdf5(data, data_file, chunk_size=100000):
    """ Stream write dict of dist tensor into a HDF5 file.

    Parameters
    ----------
    data : dict of dist tensor
        The data to be saved to the hdf5 file.
    data_file : str
        The file name of the hdf5 file.
    chunk_size : int
        The size of a chunk to extract from dist tensor.
    """
    chunk_size = 100000
    with h5py.File(data_file, "w") as f:
        for key, val in data.items():
            arr = f.create_dataset(key, val.shape, dtype=np.array(val[0]).dtype)
            if len(val) > chunk_size:
                num_chunks = len(val) // chunk_size
                remainder = len(val) % chunk_size
                for i in range(num_chunks):
                    # extract a chunk from dist tensor
                    chunk_val = np.array(val[i*chunk_size:(i+1)*chunk_size])
                    arr[i*chunk_size:(i+1)*chunk_size] = chunk_val
                # need to write remainder
                if remainder != 0:
                    remainder_val = np.array(val[num_chunks*chunk_size:len(val)])
                    arr[num_chunks*chunk_size:] = remainder_val
            else:
                arr[:] = np.array(val[0:len(val)])

def write_data_hdf5(data, data_file):
    """ Write data into a HDF5 file.

    Parameters
    ----------
    data : dict
        The data to be saved to the hdf5 file.
    data_file : str
        The file name of the hdf5 file.
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
            if isinstance(feat_conf["feature_col"], str):
                keys.append(feat_conf["feature_col"])
            elif isinstance(feat_conf["feature_col"], list):
                for feat_key in feat_conf["feature_col"]:
                    keys.append(feat_key)
            else:
                raise TypeError("Feature column must be a str or a list of string.")
    if "labels" in conf:
        for label_conf in conf["labels"]:
            if "label_col" in label_conf:
                keys.append(label_conf["label_col"])

    # We need to remove duplicated keys to avoid retrieve the same data multiple times.
    keys = list(set(keys))
    if fmt["name"] == "parquet":
        return partial(read_data_parquet, data_fields=keys)
    elif fmt["name"] == "json":
        return partial(read_data_json, data_fields=keys)
    elif fmt["name"] == "hdf5":
        return partial(read_data_hdf5, data_fields=keys, in_mem=in_mem)
    elif fmt["name"] == "csv":
        delimiter = fmt["separator"] if "separator" in fmt else ","
        return partial(read_data_csv, data_fields=keys, delimiter=delimiter)
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
        assert len(in_files) > 0, \
            f"There is no file matching {in_files} pattern"
    # This is a single file.
    elif not isinstance(in_files, list):
        in_files = [in_files]

    # Verify the existence of the input files.
    for in_file in in_files:
        assert os.path.isfile(in_file), \
                f"The input file {in_file} does not exist or is not a file."
    in_files.sort()
    return in_files
