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

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

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
        assert key in df_table, f"The data field {key} does not exist in the data file."
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

def _parse_file_format(conf, is_node):
    """ Parse the file format blob

    Parameters
    ----------
    conf : dict
        Describe the config for the node type or edge type.
    is_node : bool
        Whether this is a node config or edge config

    Returns
    -------
    callable : the function to read the data file.
    """
    fmt = conf["format"]
    assert 'name' in fmt, "'name' field must be defined in the format."
    keys = [conf["node_id_col"]] if is_node \
            else [conf["source_id_col"], conf["dest_id_col"]]
    if "features" in conf:
        keys += [feat_conf["feature_col"] for feat_conf in conf["features"]]
    if "labels" in conf:
        keys += [label_conf["label_col"] for label_conf in conf["labels"]]
    if fmt["name"] == "parquet":
        return partial(read_data_parquet, data_fields=keys)
    elif fmt["name"] == "json":
        return partial(read_data_json, data_fields=keys)
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
    if '*' in in_files:
        in_files = glob.glob(in_files)
    elif not isinstance(in_files, list):
        in_files = [in_files]
    in_files.sort()
    return in_files
