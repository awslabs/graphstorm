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
import os
import logging
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from .file_io import read_data_parquet
from .utils import ExtMemArrayWrapper

GIB_BYTES = 1024**3

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

class IdReverseMap:
    """ Map GraphStorm node ID into original Node ID

        This loads an ID map for output IDs.

        Parameters
        ----------
        id_map_prefix : str
            Id mapping file prefix
    """
    def __init__(self, id_map_prefix):
        assert os.path.exists(id_map_prefix), \
            f"{id_map_prefix} does not exist."
        try:
            data = read_data_parquet(id_map_prefix, ["orig", "new"])
        except AssertionError:
            # To maintain backwards compatibility with GraphStorm v0.2.1
            data = read_data_parquet(id_map_prefix, ["node_str_id", "node_int_id"])
            data["new"] = data["node_int_id"]
            data["orig"] = data["node_str_id"]
            data.pop("node_int_id")
            data.pop("node_str_id")

        sort_idx = np.argsort(data['new'])
        self._ids = data['orig'][sort_idx]

    def __len__(self):
        return len(self._ids)

    def map_range(self, start, end):
        """ Map a range of GraphStorm IDs to the raw IDs.

        Parameters
        ----------
        start : int
            Starting idx
        end: int
            Ending indx

        Returns
        -------
        tensor: A numpy array of raw IDs.
        """
        return self._ids[start:end]

    def map_id(self, ids):
        """ Map the GraphStorm IDs to the raw IDs.

        Parameters
        ----------
        ids : numpy array
            The input IDs

        Returns
        -------
        tensor: A numpy array of raw IDs.
        """
        if len(ids) == 0:
            return np.array([], dtype=np.str)

        return self._ids[ids]

class IdMap:
    """ Map an ID to a new ID.

    This creates an ID map for the input IDs.

    Parameters
    ----------
    ids : Array
        The input IDs
    """
    def __init__(self, ids):
        # If the IDs are stored in ExtMemArray, we should convert it to Numpy array.
        # ExtMemArray stores data on disks. Loading all IDs to memory can accelerate
        # the following operations.
        if isinstance(ids, ExtMemArrayWrapper):
            ids = ids.to_numpy()

        # We can not expect the dtype of ids is always integer or string
        # it can be any type. So we will cast ids into string if it is not integer.
        if isinstance(ids[0], int) or np.issubdtype(ids.dtype, np.integer):
            # node_ids are integer ids
            self._ids = {id1: i for i, id1 in enumerate(ids)}
        else:
            # cast everything else into string
            self._ids = {str(id1): i for i, id1 in enumerate(ids)}

    def __len__(self):
        return len(self._ids)

    @property
    def map_key_dtype(self):
        """ Return the data type of map keys.
        """
        for id_ in self._ids:
            if isinstance(id_, np.ndarray):
                return id_.dtype
            else:
                return type(id_)

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
        if len(ids) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        for id_ in self._ids:
            # If the data type of the key is string, the input Ids should not be integer.
            if isinstance(id_, str):
                assert (not isinstance(ids[0], int)) and \
                       (not np.issubdtype(ids.dtype, np.integer)), \
                    "The key of ID map is string, input IDs are integers."
            elif isinstance(id_, int) or np.issubdtype(id_.dtype, np.integer):
                # If the data type of the key is integer, the input Ids should
                # also be integers.
                assert np.issubdtype(ids.dtype, np.integer), \
                        "The key of ID map is integer, input IDs should also be integers. " \
                        + f"But get {type(ids[0])}."
            else:
                logging.warning("The input data type is %s. Will treat IDs as string.",
                                type(id_))
            break

        # If the input ID exists in the ID map, map it to a new ID
        # and keep its location in the input ID array.
        # Otherwise, skip the ID.
        new_ids = []
        idx = []
        for i, id_ in enumerate(ids):
            id_ = id_ if np.issubdtype(ids.dtype, np.integer) else str(id_)
            if id_ in self._ids:
                new_ids.append(self._ids[id_])
                idx.append(i)
        return np.array(new_ids), np.array(idx)

    def save(self, file_prefix):
        """ Save the ID map to a set of parquet files.

        Files are split such that they are not significantly larger
        than 1GB per file.

        Parameters
        ----------
        file_prefix : str
            The file prefix under which the ID map will be saved to.

        """
        os.makedirs(file_prefix, exist_ok=True)
        table = pa.Table.from_arrays([pa.array(self._ids.keys()), self._ids.values()],
                                     names=["orig", "new"])
        bytes_per_row = table.nbytes // table.num_rows
        # Split table in parts, such that the max expected file size is ~1GB
        max_rows_per_file = GIB_BYTES // bytes_per_row
        rows_written = 0
        file_idx = 0
        while rows_written < table.num_rows:
            start = rows_written
            end = min(rows_written + max_rows_per_file, table.num_rows)
            filename = f"part-{str(file_idx).zfill(5)}.parquet"
            pq.write_table(table.slice(start, end), os.path.join(file_prefix, filename))
            rows_written = end
            file_idx += 1

def map_node_ids(src_ids, dst_ids, edge_type, node_id_map, skip_nonexist_edges):
    """ Map node IDs of source and destination nodes of edges.

    In the ID mapping, we need to handle multiple errors in the input data:
    1) we handle the case that endpoint nodes of edges don't exist; if all endpoint nodes
    do not exist, we return an empty edge list.
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
    tuple of tensors :
        src_ids: The remapped source node IDs.
        dst_ids: The remapped destination node IDs.
        src_exist_locs: The locations of source node IDs that
                        have existing edges. Only valid when
                        skip_nonexist_edges is True.
        dst_exist_locs: The location of destination node IDs that
                        have existing edges. Only valid when
                        skip_nonexist_edges is True.

        How to use src_exist_locs and dst_exist_locs:
        feat_data = feat_data[src_exist_locs][dst_exist_locs]
    """
    src_type, _, dst_type = edge_type
    new_src_ids, orig_locs = node_id_map[src_type].map_id(src_ids)
    src_exist_locs = None
    dst_exist_locs = None
    # If some of the source nodes don't exist in the node set.
    if len(orig_locs) != len(src_ids):
        bool_mask = np.ones(len(src_ids), dtype=bool)
        if len(orig_locs) > 0:
            bool_mask[orig_locs] = False
        if skip_nonexist_edges:
            logging.warning("source nodes of %s do not exist. Skip %d edges",
                            src_type, len(src_ids[bool_mask]))
        else:
            raise ValueError(f"source nodes of {src_type} do not exist: {src_ids[bool_mask]}")
        dst_ids = dst_ids[orig_locs] if len(orig_locs) > 0 else np.array([], dtype=dst_ids.dtype)
        src_exist_locs = orig_locs
    src_ids = new_src_ids

    new_dst_ids, orig_locs = node_id_map[dst_type].map_id(dst_ids)
    # If some of the dest nodes don't exist in the node set.
    if len(orig_locs) != len(dst_ids):
        bool_mask = np.ones(len(dst_ids), dtype=bool)
        if len(orig_locs) > 0:
            bool_mask[orig_locs] = False
        if skip_nonexist_edges:
            logging.warning("dest nodes of %s do not exist. Skip %d edges",
                            dst_type, len(dst_ids[bool_mask]))
        else:
            raise ValueError(f"dest nodes of {dst_type} do not exist: {dst_ids[bool_mask]}")
        # We need to remove the source nodes as well.
        src_ids = src_ids[orig_locs] if len(orig_locs) > 0 else np.array([], dtype=src_ids.dtype)
        dst_exist_locs = orig_locs
    dst_ids = new_dst_ids
    return src_ids, dst_ids, src_exist_locs, dst_exist_locs
