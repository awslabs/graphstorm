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
import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from .file_io import HDF5Array

class NoopMap:
    """ It doesn't map IDs.

    This is an identity map. It doesn't do any mapping on IDs.

    Parameters
    ----------
    size : int
        The map size.
    """
    def __init__(self, size):
        self._size = size

    def __len__(self):
        return self._size

    def map_id(self, ids):
        """ Map the input IDs to the new IDs.

        This is identity map, so we don't need to do anything.

        Parameters
        ----------
        ids : tensor
            The input IDs

        Returns
        -------
        tuple of tensors : the tensor of new IDs, the location of the IDs in the input ID tensor.
        """
        return ids, np.arange(len(ids))

    def save(self, file_path):
        """ Save the ID map.

        Parameters
        ----------
        file_path : str
            The file where the ID map is saved to.
        """

class IdMap:
    """ Map an ID to a new ID.

    This creates an ID map for the input IDs.

    Parameters
    ----------
    ids : Array
        The input IDs
    """
    def __init__(self, ids):
        # If the IDs are stored in HDF5Array, we should convert it to Numpy array.
        # HDF5Array stores data on disks. Loading all IDs to memory can accelerate
        # the following operations.
        if isinstance(ids, HDF5Array):
            ids = ids.to_numpy()
        self._ids = {id1: i for i, id1 in enumerate(ids)}

    def __len__(self):
        return len(self._ids)

    def map_id(self, ids):
        """ Map the input IDs to the new IDs.

        Parameters
        ----------
        ids : tensor
            The input IDs

        Returns
        -------
        tuple of tensors : the tensor of new IDs, the location of the IDs in the input ID tensor.
        """
        for id_ in self._ids:
            # If the data type of the key is string, the input Ids should also be strings.
            if isinstance(id_, str):
                assert isinstance(ids[0], str), \
                        "The key of ID map is string, input IDs should also be strings."
            elif isinstance(id_, int) or np.issubdtype(id_.dtype, np.integer):
                # If the data type of the key is integer, the input Ids should
                # also be integers.
                assert np.issubdtype(ids.dtype, np.integer), \
                        "The key of ID map is integer, input IDs should also be integers. " \
                        + f"But get {type(ids[0])}."
            else:
                raise ValueError(f"Unsupported key data type: {type(id_)}")
            break

        # If the input ID exists in the ID map, map it to a new ID
        # and keep its location in the input ID array.
        # Otherwise, skip the ID.
        new_ids = []
        idx = []
        for i, id_ in enumerate(ids):
            if id_ in self._ids:
                new_ids.append(self._ids[id_])
                idx.append(i)
        return np.array(new_ids), np.array(idx)

    def save(self, file_path):
        """ Save the ID map to a parquet file.

        Parameters
        ----------
        file_path : str
            The file where the ID map will be saved to.

        Returns
        -------
        bool : whether the ID map is saved to a file.
        """
        keys = list(self._ids.keys())
        vals = list(self._ids.values())
        table = pa.Table.from_pandas(pd.DataFrame({'orig': keys, 'new': vals}))
        pq.write_table(table, file_path)
        return True

def map_node_ids(src_ids, dst_ids, edge_type, node_id_map, skip_nonexist_edges):
    """ Map node IDs of source and destination nodes of edges.

    In the ID mapping, we need to handle multiple errors in the input data:
    1) we handle the case that endpoint nodes of edges don't exist;
    2) we handle the case that the data type of node IDs of the endpoint nodes don't
    match the data type of the keys of the ID map.

    Parameters
    ----------
    src_ids : tensor
        The source nodes.
    dst_ids : tensor
        The destination nodes.
    edge_type : tuple
        It contains source node type, relation type, destination node type.
    node_id_map : dict
        The key is the node type and value is IdMap or NoopMap.
    skip_nonexist_edges : bool
        Whether or not to skip edges whose endpoint nodes don't exist.

    Returns
    -------
    tuple of tensors : the remapped source and destination node IDs.
    """
    src_type, _, dst_type = edge_type
    new_src_ids, orig_locs = node_id_map[src_type].map_id(src_ids)
    # If some of the source nodes don't exist in the node set.
    if len(orig_locs) != len(src_ids):
        bool_mask = np.ones(len(src_ids), dtype=bool)
        bool_mask[orig_locs] = False
        if skip_nonexist_edges:
            logging.warning("source nodes of %s do not exist: %s",
                            src_type, str(src_ids[bool_mask]))
        else:
            raise ValueError(f"source nodes of {src_type} do not exist: {src_ids[bool_mask]}")
        dst_ids = dst_ids[orig_locs]
    src_ids = new_src_ids

    new_dst_ids, orig_locs = node_id_map[dst_type].map_id(dst_ids)
    # If some of the dest nodes don't exist in the node set.
    if len(orig_locs) != len(dst_ids):
        bool_mask = np.ones(len(dst_ids), dtype=bool)
        bool_mask[orig_locs] = False
        if skip_nonexist_edges:
            logging.warning("dest nodes of %s do not exist: %s",
                            dst_type, str(dst_ids[bool_mask]))
        else:
            raise ValueError(f"dest nodes of {dst_type} do not exist: {dst_ids[bool_mask]}")
        # We need to remove the source nodes as well.
        src_ids = src_ids[orig_locs]
    dst_ids = new_dst_ids
    return src_ids, dst_ids
