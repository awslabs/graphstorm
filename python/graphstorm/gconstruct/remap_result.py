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

    Remapping GraphStorm outputs (edge prediction results
    node prediction results and node embeddings)
    into the Raw Node ID space.
"""

import os
import argparse
import logging
import json
import time
import sys
import math
from functools import partial

import pandas as pd
import torch as th
from ..model.utils import pad_file_index
from .file_io import write_data_parquet
from .id_map import IdReverseMap
from ..utils import get_log_level
from .utils import multiprocessing_exec_no_return as multiprocessing_remap

from ..config import (GSConfig,
                      get_argument_parser,
                      BUILTIN_TASK_EDGE_CLASSIFICATION,
                      BUILTIN_TASK_EDGE_REGRESSION,
                      BUILTIN_TASK_NODE_CLASSIFICATION,
                      BUILTIN_TASK_NODE_REGRESSION,
                      BUILTIN_TASK_LINK_PREDICTION)

GS_OUTPUT_FORMAT_PARQUET = "parquet"
GS_OUTPUT_FORMAT_CSV = "csv"

GS_REMAP_NID_COL = "nid"
GS_REMAP_PREDICTION_COL = "pred"
GS_REMAP_SRC_NID_COL = "src_nid"
GS_REMAP_DST_NID_COL = "dst_nid"
GS_REMAP_EMBED_COL = "emb"

GS_REMAP_BUILTIN_COLS = [GS_REMAP_NID_COL,
                         GS_REMAP_PREDICTION_COL,
                         GS_REMAP_SRC_NID_COL,
                         GS_REMAP_DST_NID_COL,
                         GS_REMAP_EMBED_COL]

# Id_maps is a global variable.
# When using multi-processing to do id remap,
# we do not want to pass id_maps to each worker process
# through argument which uses Python pickle to copy
# data. By making id_maps as a global variable, we
# can rely on Linux copy-on-write to provide a zero-copy
# id_maps to each worker process.
id_maps = {}

def write_data_parquet_file(data, file_prefix, col_name_map=None):
    """ Write data into disk using parquet format.

        Parameters
        ----------
        data: dict of numpy Arrays
            Data to be written into disk.
        file_prefix: str
            File prefix. The output will be <file_prefix>.parquet.
        col_name_map: dict
            A mapping from builtin column name to user defined column name.
    """
    if col_name_map is not None:
        updated_data = {}
        for key, val in data.items():
            if key in col_name_map:
                updated_data[col_name_map[key]] = val
            else:
                updated_data[key] = val
        data = updated_data

    output_fname = f"{file_prefix}.parquet"
    write_data_parquet(data, output_fname)

def write_data_csv_file(data, file_prefix, delimiter=",", col_name_map=None):
    """ Write data into disk using csv format.

        Multiple values for a field are specified with a semicolon (;) between values.

        Example:

        .. code::

            nide, emb
            0, 0.001;1.2000;0.736;...

        Parameters
        ----------
        data: dict of numpy Arrays
            Data to be written into disk.
        file_prefix: str
            File prefix. The output will be <file_prefix>.parquet.
        delimiter: str
            Delimiter used to separate columns.
        col_name_map: dict
            A mapping from builtin column name to user defined column name.
    """
    if col_name_map is not None:
        updated_data = {}
        for key, val in data.items():
            if key in col_name_map:
                updated_data[col_name_map[key]] = val
            else:
                updated_data[key] = val
        data = updated_data

    output_fname = f"{file_prefix}.csv"
    csv_data = {}
    for key, vals in data.items():
        # Each <key, val> pair represents the column name and
        # the column data of a column.
        if len(vals.shape) == 1:
            # vals is a 1D matrix.
            # The data will be saved as
            #   key,
            #   0.1,
            #   0.2,
            #   ...
            csv_data[key] = vals.tolist()
        elif len(vals.shape) == 2:
            # vals is a 2D matrix.
            # The data will be saved as
            #   key,
            #   0.001;1.2000;0.736;...,
            #   0.002;1.1010;0.834;...,
            #   ...
            csv_data[key] = [";".join([str(v) for v in val]) \
                             for val in vals.tolist()]
    data_frame = pd.DataFrame(csv_data)
    data_frame.to_csv(output_fname, index=False, sep=delimiter)

def worker_remap_node_data(data_file_path, nid_path, ntype, data_col_key,
    output_fname_prefix, chunk_size, output_func):
    """ Do one node prediction remapping task

        Parameters
        ----------
        data_file_path: str
            The path to the node data.
        nid_path: str
            The path to the file storing node ids
        ntype: str
            Node type.
        data_col_key: str
            Column key of the node data
        output_fname_prefix: str
            Output file name prefix.
        chunk_size: int
            Max number of raws per output file.
        output_func: func
            Function used to write data to disk.
    """
    node_data = th.load(data_file_path).numpy()
    nids = th.load(nid_path).numpy()
    nid_map = id_maps[ntype]
    num_chunks = math.ceil(len(node_data) / chunk_size)

    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i + 1 < num_chunks else len(node_data)
        data = node_data[start:end]
        nid = nid_map.map_id(nids[start:end])
        data = {data_col_key: data,
                GS_REMAP_NID_COL: nid}
        output_func(data, f"{output_fname_prefix}_{pad_file_index(i)}")

def worker_remap_edge_pred(pred_file_path, src_nid_path,
    dst_nid_path, src_type, dst_type,
    output_fname_prefix, chunk_size,
    output_func):
    """ Do one edge remapping task

        Parameters
        ----------
        pred_file_path: str
            The path to the prediction result.
        src_nid_path: str
            The path to the file storing source node ids
        dst_nid_path: str
            The path to the file storing destination node ids
        src_type: str
            Src node type.
        dst_type: str
            Dst node type.
        output_fname_prefix: str
            Output file name prefix.
        chunk_size: int
            Max number of raws per output file.
        output_func: func
            Function used to write data to disk.
    """
    pred_result = th.load(pred_file_path).numpy()
    src_nids = th.load(src_nid_path).numpy()
    dst_nids = th.load(dst_nid_path).numpy()
    src_id_map = id_maps[src_type]
    dst_id_map = id_maps[dst_type]
    num_chunks = math.ceil(len(pred_result) / chunk_size)
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i + 1 < num_chunks else len(pred_result)
        pred = pred_result[start:end]
        src_nid = src_id_map.map_id(src_nids[start:end])
        dst_nid = dst_id_map.map_id(dst_nids[start:end])
        data = {GS_REMAP_PREDICTION_COL: pred,
                GS_REMAP_SRC_NID_COL: src_nid,
                GS_REMAP_DST_NID_COL: dst_nid}

        output_func(data, f"{output_fname_prefix}_{pad_file_index(i)}")

def _get_file_range(num_files, rank, world_size):
    """ Get the range of files to process by the current instance.

        Parameters
        ----------
        num_files: int
            Total number of files.
        rank: int
            Current worker rank.
        world_size: int
            Total number of workers.

        Return
        ------
        (int, int): start idx and end idx
    """
    if world_size == 0:
        return 0, num_files

    files_per_inst = num_files // world_size
    rest = num_files % world_size
    # evenly spread the fiels
    # For example, with 10 files and world_size == 4
    # number files for each rank will be:
    # 2 2 3 3
    if rank + rest < world_size:
        start = files_per_inst * rank
        end = start + files_per_inst
    else:
        start = files_per_inst * rank + rank + rest - world_size
        end = start + files_per_inst + 1

    return start, end

def _remove_inputs(with_shared_fs, files_to_remove,
                   rank, world_size, work_dir):
    if with_shared_fs is False:
        # Not using shared file system. There is no contention.
        # Each process will remove the files itself
        for file in files_to_remove:
            os.remove(file)
    else:
        # Shared file system is used.
        # Only rank 0 is going to remove the files.
        if rank == 0:
            for i in range(1, world_size):
                while not os.path.isfile(os.path.join(work_dir, f"SUCC_{i}")):
                    time.sleep(1)
                os.remove(os.path.join(work_dir, f"SUCC_{i}"))
            for file in files_to_remove:
                os.remove(file)
        else:
            # Tell rank 0, rank n has finished its work.
            with open(os.path.join(work_dir, f"SUCC_{rank}"),
                      'w', encoding='utf-8') as f: # pylint: disable=unused-variable
                pass

def remap_node_emb(emb_ntypes, node_emb_dir,
                   output_dir, out_chunk_size,
                   num_proc, rank, world_size,
                   with_shared_fs, output_func):
    """ Remap node embeddings.

        The function will iterate all the node types that
        have embeddings and spin num_proc workers
        to do the remapping jos.

        The directory storing node embeddings looks like:

        Example
        --------
        # embedddings:
        #   ntype0:
        #     embed_nids-00000.pt
        #     embed_nids-00001.pt
        #     ...
        #     embed-00000.pt
        #     embed-00001.pt
        #     ...
        #   ntype1:
        #     embed_nids-00000.pt
        #     embed_nids-00001.pt
        #     ...
        #     embed-00000.pt
        #     embed-00001.pt
        #     ...

        The output files will be

        Example
        --------
        # embedddings:
        #   ntype0:
        #     embed-00000_00000.parquet
        #     embed-00000_00001.parquet
        #     ...
        #   ntype1:
        #     embed-00000_00000.parquet
        #     embed-00000_00001.parquet
        #     ...

        Parameters
        ----------
        emb_ntypes: list of str
            List of node types that have node embeddings to be remapped。
        node_emb_dir: str
            The directory storing the node embeddings.
        output_dir: str
            The directory storing the remapped node embeddings.
        out_chunk_size: int
            Max number of rows per output file.
        num_proc: int
            Number of workers used in processing.
        rank: int
            The global rank of current processes.
        world_size: int
            The total number of processes in the cluster.
        with_shared_fs: bool
            Whether shared file system is avaliable.
        output_func: func
            Function used to write data to disk.

        Return
        --------
        list of str
            The list of files to be removed.
    """
    task_list = []
    files_to_remove = []
    for ntype in emb_ntypes:
        input_emb_dir = os.path.join(node_emb_dir, ntype)
        out_embdir = os.path.join(output_dir, ntype)
        ntype_emb_files = os.listdir(input_emb_dir)
        # please note nid_files can be empty.
        nid_files = [fname for fname in ntype_emb_files \
                     if fname.startswith("embed_nids-") and fname.endswith("pt")]
        emb_files = [fname for fname in ntype_emb_files \
                     if fname.startswith("embed-") and fname.endswith("pt")]

        nid_files.sort()
        emb_files.sort()
        num_parts = len(emb_files)
        logging.debug("{%s} has {%d} embedding files", ntype, num_parts)
        assert len(nid_files) == len(emb_files), \
            "Number of nid files must match number of embedding files. " \
            f"But get {len(nid_files)} and {len(emb_files)}."
        files_to_remove += [os.path.join(input_emb_dir, nid_file) \
                            for nid_file in nid_files]
        files_to_remove += [os.path.join(input_emb_dir, emb_file) \
                            for emb_file in emb_files]

        if with_shared_fs:
            # If the data are stored in a shared filesystem,
            # each instance only needs to process
            # a subset of edge prediction files
            start, end = _get_file_range(num_parts, rank, world_size)
        else:
            # If the data are stored in local filesystem (not shared),
            # each instance needs to process all
            # the edge prediction files stored locally
            start, end = 0, num_parts

        logging.debug("{%d} handle {%d}-{%d}", rank, start, end)
        for i in range(start, end):
            emb_file = emb_files[i]
            nid_file = nid_files[i]

            task_list.append({
                "data_file_path": os.path.join(input_emb_dir, emb_file),
                "nid_path": os.path.join(input_emb_dir, nid_file),
                "ntype": ntype,
                "data_col_key": GS_REMAP_EMBED_COL,
                "output_fname_prefix": os.path.join(out_embdir, \
                    f"{emb_file[:emb_file.rindex('.')]}"),
                "chunk_size": out_chunk_size,
                "output_func": output_func,
            })

    multiprocessing_remap(task_list, num_proc, worker_remap_node_data)
    return files_to_remove

def remap_node_pred(pred_ntypes, pred_dir,
                    output_dir, out_chunk_size,
                    num_proc, rank, world_size, with_shared_fs,
                    output_func):
    """ Remap node prediction result.

        The function wil iterate all the node types that
        have prediction results and spin num_proc workers
        to do the rampping jos.

        The directory storing prediction results looks like:
        # Predicionts:
        #    predict-00000.pt
        #    predict-00001.pt
        #    ...
        #    predict_nids-00000.pt
        #    predict_nids-00001.pt

        The output files will be
        #    predict-00000_00000.parquet
        #    predict-00000_00001.parquet
        #    ...

        Parameters
        ----------
        pred_ntypes: list of str
            List of node types that have prediction results to be remapped。
        pred_dir: str
            The directory storing the prediction results.
        output_dir: str
            The directory storing the remapped prediction results.
        out_chunk_size: int
            Max number of raws per output file.
        num_proc: int
            Number of workers used in processing.
        rank: int
            The global rank of current processes.
        world_size: int
            The total number of processes in the cluster.
        with_shared_fs: bool
            Whether shared file system is avaliable.
        output_func: func
            Function used to write data to disk.

        Return
        --------
        list of str
            The list of files to be removed.
    """
    start_time = time.time()
    task_list = []
    files_to_remove = []
    for ntype in pred_ntypes:
        input_pred_dir = os.path.join(pred_dir, ntype)
        out_pred_dir = os.path.join(output_dir, ntype)
        ntype_pred_files = os.listdir(input_pred_dir)
        nid_files = [fname for fname in ntype_pred_files if fname.startswith("predict_nids-")]
        pred_files = [fname for fname in ntype_pred_files if fname.startswith("predict-")]

        nid_files.sort()
        pred_files.sort()
        num_parts = len(pred_files)
        logging.debug("{%s} has {%d} prediction files", ntype, num_parts)
        assert len(nid_files) == len(pred_files), \
            "Expect the number of nid files equal to " \
            "the number of prediction result files, but get " \
            f"{len(nid_files)} and {len(pred_files)}"

        files_to_remove += [os.path.join(input_pred_dir, nid_file) \
                            for nid_file in nid_files]
        files_to_remove += [os.path.join(input_pred_dir, pred_file) \
                            for pred_file in pred_files]

        if with_shared_fs:
            # If the data are stored in a shared filesystem,
            # each instance only needs to process
            # a subset of edge prediction files
            start, end = _get_file_range(num_parts, rank, world_size)
        else:
            # If the data are stored in local filesystem (not shared),
            # each instance needs to process all
            # the edge prediction files stored locally
            start, end = 0, num_parts

        logging.debug("{%d} handle {%d}-{%d}", rank, start, end)
        for i in range(start, end):
            pred_file = pred_files[i]
            nid_file = nid_files[i]

            task_list.append({
                "data_file_path": os.path.join(input_pred_dir, pred_file),
                "nid_path": os.path.join(input_pred_dir, nid_file),
                "ntype": ntype,
                "data_col_key": GS_REMAP_PREDICTION_COL,
                "output_fname_prefix": os.path.join(out_pred_dir, \
                    f"pred.{pred_file[:pred_file.rindex('.')]}"),
                "chunk_size": out_chunk_size,
                "output_func": output_func,
            })

    multiprocessing_remap(task_list, num_proc, worker_remap_node_data)

    dur = time.time() - start_time
    logging.info("{%d} Remapping node predictions takes {%f} secs", rank, dur)
    return files_to_remove

def remap_edge_pred(pred_etypes, pred_dir,
                    output_dir, out_chunk_size,
                    num_proc, rank, world_size, with_shared_fs,
                    output_func):
    """ Remap edge prediction result.

        The function will iterate all the edge types that
        have prediction results and spin num_proc workers
        to do the rampping jos.

        The directory storing prediction results looks like:
        # Predicionts:
        #    predict-00000.pt
        #    predict-00001.pt
        #    ...
        #    src_nids-00000.pt
        #    src_nids-00001.pt
        #    ...
        #    dst_nids-00000.pt
        #    dst_nids-00001.pt
        #    ...

        The output prediction files will be
        #    predict-00000_00000.parquet
        #    predict-00000_00001.parquet
        #    ...

        Parameters
        ----------
        pred_etypes: list of tuples
            List of edge types that have prediction results to be remapped。
        pred_dir: str
            The directory storing the prediction results.
        output_dir: str
            The directory storing the remapped prediction results.
        out_chunk_size: int
            Max number of raws per output file.
        num_proc: int
            Number of workers used in processing.
        rank: int
            The global rank of current processes.
        world_size: int
            The total number of processes in the cluster.
        with_shared_fs: bool
            Whether shared file system is avaliable.
        output_func: func
            Function used to write data to disk.

        Return
        --------
        list of str
            The list of files to be removed.
    """
    start_time = time.time()
    task_list = []
    files_to_remove = []
    for etype in pred_etypes:
        input_pred_dir = os.path.join(pred_dir, "_".join(etype))
        out_pred_dir = os.path.join(output_dir, "_".join(etype))
        etype_pred_files = os.listdir(input_pred_dir)
        src_nid_files = [fname for fname in etype_pred_files if fname.startswith("src_nids-")]
        dst_nid_files = [fname for fname in etype_pred_files if fname.startswith("dst_nids-")]
        pred_files = [fname for fname in etype_pred_files if fname.startswith("predict-")]
        src_nid_files.sort()
        dst_nid_files.sort()
        pred_files.sort()

        num_parts = len(pred_files)
        logging.debug("%s has %d embedding files", etype, num_parts)
        assert len(src_nid_files) == len(pred_files), \
            "Expect the number of source nid files equal to " \
            "the number of prediction result files, but get " \
            f"{len(src_nid_files)} and {len(pred_files)}"
        assert len(dst_nid_files) == len(pred_files), \
            "Expect the number of destination nid files equal to " \
            "the number of prediction result files, but get " \
            f"{len(dst_nid_files)} and {len(pred_files)}"
        files_to_remove += [os.path.join(input_pred_dir, src_nid_file) \
                            for src_nid_file in src_nid_files]
        files_to_remove += [os.path.join(input_pred_dir, dst_nid_file) \
                            for dst_nid_file in dst_nid_files]
        files_to_remove += [os.path.join(input_pred_dir, pred_file) \
                            for pred_file in pred_files]

        if with_shared_fs:
            # If the data are stored in a shared filesystem,
            # each instance only needs to process
            # a subset of edge prediction files
            start, end = _get_file_range(num_parts, rank, world_size)
        else:
            # If the data are stored in local filesystem (not shared),
            # each instance needs to process all
            # the edge prediction files stored locally
            start, end = 0, num_parts

        logging.debug("%d handle %d-%d}", rank, start, end)
        for i in range(start, end):
            pred_file = pred_files[i]
            src_nid_file = src_nid_files[i]
            dst_nid_file = dst_nid_files[i]
            src_type = etype[0]
            # if src ntype == dst ntype, there is no need to
            # pickle nid mappings twice
            dst_type = etype[2]

            task_list.append({
                "pred_file_path": os.path.join(input_pred_dir, pred_file),
                "src_nid_path": os.path.join(input_pred_dir, src_nid_file),
                "dst_nid_path": os.path.join(input_pred_dir, dst_nid_file),
                "src_type": src_type,
                "dst_type": dst_type,
                "output_fname_prefix": os.path.join(out_pred_dir, \
                    f"pred.{pred_file[:pred_file.rindex('.')]}"),
                "chunk_size": out_chunk_size,
                "output_func": output_func,
            })

    multiprocessing_remap(task_list, num_proc, worker_remap_edge_pred)

    dur = time.time() - start_time
    logging.debug("%d Finish edge rempaing in %f secs}", rank, dur)
    return files_to_remove

def _parse_gs_config(config):
    """ Get remapping related information from GSConfig

        Parameters
        ----------
        config: GSConfig
            config object

        Return
        ------
        str: path to node id mapping parquet
        str: path to prediction results
        str: path to saved node embeddings
        list of str: ntypes that have prediction results
        list of str: etypes that have prediction results
    """
    part_config = config.part_config
    node_id_mapping = os.path.join(os.path.dirname(part_config), "raw_id_mappings")
    predict_dir = config.save_prediction_path
    emb_dir = config.save_embed_path
    task_emb_dirs = []

    pred_ntypes = []
    pred_etypes = []
    if config.multi_tasks is not None:
        node_predict_dirs = []
        edge_predict_dirs = []
        # multi-task setting
        tasks = config.multi_tasks

        for task in tasks:
            task_config = task.task_config
            task_id = task.task_id
            if task.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
                if task_config.lp_embed_normalizer is not None:
                    # There are link prediction node embedding normalizer
                    # Need to handled the normalized embeddings.
                    task_emb_dirs.append(task_id)

        if predict_dir is None:
            return node_id_mapping, None, emb_dir, task_emb_dirs, pred_ntypes, pred_etypes

        for task in tasks:
            task_config = task.task_config
            task_id = task.task_id
            pred_path = os.path.join(predict_dir, task_id)
            if task.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION,
                                  BUILTIN_TASK_NODE_REGRESSION]:
                pred_ntype = task_config.target_ntype
                pred_ntypes.append(pred_ntype)
                node_predict_dirs.append(pred_path)
            elif task.task_type in (BUILTIN_TASK_EDGE_CLASSIFICATION,
                               BUILTIN_TASK_EDGE_REGRESSION):
                pred_etype = task_config.target_etype[0]
                pred_etypes.append(list(pred_etype))
                edge_predict_dirs.append(pred_path)

        predict_dir = (node_predict_dirs, edge_predict_dirs)
        return node_id_mapping, predict_dir, emb_dir, task_emb_dirs, pred_ntypes, pred_etypes
    else:
        task_type = config.task_type
        if task_type in (BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION):
            pred_etypes = config.target_etype
            pred_etypes = pred_etypes \
                if isinstance(pred_etypes, list) else [pred_etypes]
            pred_etypes = [list(pred_etype) for pred_etype in pred_etypes]
        elif task_type in (BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION):
            pred_ntypes = config.target_ntype
            pred_ntypes = pred_ntypes \
                if isinstance(pred_ntypes, list) else [pred_ntypes]

        return node_id_mapping, predict_dir, emb_dir, task_emb_dirs, pred_ntypes, pred_etypes

def main(args, gs_config_args):
    """ main function
    """
    rank = args.rank
    world_size = args.world_size
    with_shared_fs = args.with_shared_fs

    if args.yaml_config_file is not None:
        # Case 1: remap_result is called right after the
        # train/inference script.
        # GraphStorm yaml exists, extract information from
        # train or inference configs.
        gs_config_args = ["--cf", args.yaml_config_file,
                          "--logging-level", args.logging_level] + gs_config_args
        gs_parser = get_argument_parser()
        gs_args, _ = gs_parser.parse_known_args(gs_config_args)
        config = GSConfig(gs_args)
        config.verify_arguments(False)
        id_mapping_path, predict_dir, node_emb_dir, task_emb_dirs, pred_ntypes, pred_etypes = \
            _parse_gs_config(config)
    else:
        # Case 2: remap_result is called alone.
        # GraphStorm train/inference configs are not available.
        # We collect information from input arguments.
        logging.basicConfig(level=get_log_level(args.logging_level), force=True)
        id_mapping_path = args.node_id_mapping
        predict_dir = args.prediction_dir
        node_emb_dir = args.node_emb_dir
        # We do not handle the case when there are task specific embeddings
        # in multi-task learning, if remap_result is called alone.
        # Users need to clean up the node_emb_dir themselves.
        task_emb_dirs = []
        pred_etypes = args.pred_etypes
        pred_ntypes = args.pred_ntypes
        if pred_etypes is not None:
            assert len(pred_etypes) > 0, \
                "prediction etypes is empty"
            pred_etypes = [etype.split(",") for etype in pred_etypes]
        else:
            pred_etypes = []

        if pred_ntypes is not None:
            assert len(pred_ntypes) > 0, \
                "prediction ntypes is empty"
        else:
            pred_ntypes = []

        if not with_shared_fs:
            # When shared file system is not available and the world_size is
            # larger than 1, it means it is a distributed remap task.
            # If remapping prediction result is required, i.e., predict_dir
            # is not None, either pred_etypes or pred_ntypes must be
            # provided.
            if predict_dir is not None and world_size > 0 \
                and len(pred_etypes) == 0 and len(pred_ntypes) == 0:
                raise RuntimeError("You want to do prediction result remap as" \
                                   f"predict-dir {predict_dir} is not None. " \
                                   "but both pred-etypes and pred-ntypes are empty.")

    assert world_size > 0, \
        f"World size must be larger than 0, but get {world_size}."
    assert rank < world_size, \
        f"Expecting {world_size} workers but the worker ID is {rank}."
    out_chunk_size = args.output_chunk_size
    assert out_chunk_size > 0, \
        f"Output chunk size should be larger than 0 but get {out_chunk_size}."

    ################## remap embeddings #############
    emb_ntypes = []
    if node_emb_dir is not None:
        # If node embedding exists, we are going to remap all the embeddings.
        if with_shared_fs:
            if os.path.exists(os.path.join(node_emb_dir, "emb_info.json")):
                with open(os.path.join(node_emb_dir, "emb_info.json"),
                        "r",  encoding='utf-8') as f:
                    info = json.load(f)
                    ntypes = info["emb_name"]
                    emb_ntypes = ntypes if isinstance(ntypes, list) else [ntypes]

        else: # There is no shared file system
            emb_names = os.listdir(node_emb_dir)
            # In single task learning, the node embed dir looks like:
            # emb_dir/
            #     ntype0
            #     ntype1
            #     ...
            #     emb_info.json
            #
            # In multi-task learning, the node embed dir looks like:
            # emb_dir/
            #     ntype0
            #     ntype1
            #     ...
            #     emb_info.json
            #     task_id0/
            #     task_id1/
            #     ...
            # We need to exclude both emb_info.json and task_id directories,
            # when we are collecting node types with node embeddings.
            emb_names = [e_name for e_name in emb_names \
                if e_name not in task_emb_dirs + ["emb_info.json"]]

            emb_ntypes = emb_names
    else:
        logging.info("Node embedding directory is not provided. "
                     "Skip remapping node embeddings.")

    ################## remap prediction #############
    if predict_dir is not None and isinstance(predict_dir, str):
        # predict_dir is a string
        # There is only one prediction task.
        assert os.path.exists(predict_dir), \
            f"Prediction dir {predict_dir} does not exist."
        # if pred_etypes (edges with prediction results)
        # is not empty, we need to remap edge prediction results.
        # Note: For distributed SageMaker runs, pred_etypes must be
        # provided if edge prediction result remap is required,
        # as result_info.json is only saved by rank0 and
        # there is no shared file system.
        if len(pred_etypes) > 0:
            exist_pred_etypes = []
            for pred_etype in pred_etypes:
                if os.path.exists(os.path.join(predict_dir, "_".join(pred_etype))):
                    exist_pred_etypes.append(pred_etype)
                else:
                    logging.warning("prediction results of %s "
                                    "do not exists. Skip doing remapping for it",
                                    pred_etype)
            pred_etypes = exist_pred_etypes

        # If pred_ntypes (nodes with prediction results)
        # is not empty, we need to remap node prediction results
        # Note: For distributed SageMaker runs, pred_ntypes must be
        # provided if node prediction result remap is required,
        # as result_info.json is only saved by rank0 and
        # there is no shared file system.
        if len(pred_ntypes) > 0:
            exist_pred_ntypes = []
            for pred_ntype in pred_ntypes:
                if os.path.exists(os.path.join(predict_dir, pred_ntype)):
                    exist_pred_ntypes.append(pred_ntype)
                else:
                    logging.warning("prediction results of %s"
                                    "do not exists. Skip doing remapping for it",
                                    pred_ntype)
            pred_ntypes = exist_pred_ntypes

        if with_shared_fs:
            # Only when shared file system is avaliable,
            # we will check result_info.json for
            # pred_etypes and pred_ntypes.
            # If shared file system is not avaliable
            # result_info.json is not guaranteed to exist
            # on each instances. So users must use
            # --pred-ntypes or --pred-etypes instead.
            #
            # In case when both --pred-ntypes or --pred-etypes
            # are provided while result_info.json is also avaliable,
            # GraphStorm remaping will follow --pred-ntypes or --pred-etypes
            # and ignore the result_info.json.
            if os.path.exists(os.path.join(predict_dir, "result_info.json")):
                # User does not provide pred_etypes.
                # Try to get it from saved prediction config.
                with open(os.path.join(predict_dir, "result_info.json"),
                            "r",  encoding='utf-8') as f:
                    info = json.load(f)
                    if len(pred_etypes) == 0:
                        pred_etypes = [list(etype) for etype in info["etypes"]] \
                            if "etypes" in info else []
                    if len(pred_ntypes) == 0:
                        pred_ntypes = info["ntypes"] if "ntypes" in info else []
    elif predict_dir is not None and isinstance(predict_dir, tuple):
        # This is multi-task learning.
        # we only get predict_dir with type list
        # from yaml config
        node_predict_dirs, edge_predict_dirs = predict_dir

        if len(node_predict_dirs) == 0 and \
            len(edge_predict_dirs) == 0:
            logging.info("Prediction results are empty."
                         "Skip remapping prediction result.")
            pred_etypes = []
            pred_ntypes = []
        else:
            # check the prediciton result paths
            exist_pred_ntypes = []
            for pred_dir, pred_ntype in zip(node_predict_dirs, pred_ntypes):
                if os.path.exists(pred_dir):
                    exist_pred_ntypes.append(pred_ntype)
                    # if the prediction path exists
                    # the <prediction-path>/<ntype> must exists.
                    assert os.path.exists(os.path.join(pred_dir, pred_ntype)), \
                        f"Prediction dir {os.path.join(pred_dir, pred_ntype)}" \
                        f"for {pred_ntype} does not exist."
                else:
                    # The prediction path may not exist.
                    logging.warning("prediction results of %s"
                                    "do not exists. Skip doing remapping for it",
                                    pred_dir)

            exist_pred_etypes = []
            for pred_dir, pred_etype in zip(edge_predict_dirs, pred_etypes):
                if os.path.exists(pred_dir):
                    # if the prediction path exists
                    # the <prediction-path>/<etype> must exists.
                    pred_path = os.path.join(pred_dir, "_".join(pred_etype))
                    assert os.path.exists(pred_path), \
                        f"Prediction dir {pred_path}" \
                        f"for {pred_etype} does not exist."
                    exist_pred_etypes.append(pred_etype)
                else:
                    # The prediction path may not exist.
                    logging.warning("prediction results of %s"
                                    "do not exists. Skip doing remapping for it",
                                    pred_dir)
            pred_ntypes = exist_pred_ntypes
            pred_etypes = exist_pred_etypes
    else:
        pred_etypes = []
        pred_ntypes = []
        logging.info("Prediction result directory is not provided. "
                     "Skip remapping prediction result.")


    ntypes = []
    ntypes += emb_ntypes
    ntypes += [etype[0] for etype in pred_etypes] + \
        [etype[2] for etype in pred_etypes]
    ntypes += pred_ntypes
    if len(ntypes) == 0:
        # Nothing to remap
        logging.warning("No nodes to remap, skipping remapping edge/node "
                        "predictions and node embeddings. "
                        "Embeddings will remain in PyTorch format.")
        sys.exit(0)

    for ntype in set(ntypes):
        mapping_prefix = os.path.join(id_mapping_path, ntype)
        logging.debug("loading mapping file %s",
                      mapping_prefix)
        if os.path.exists(mapping_prefix):
            id_maps[ntype] = \
                IdReverseMap(mapping_prefix)
        else:
            logging.warning(
                ("ID mapping prefix %s does not exist, skipping remapping. "
                 "Embeddings will remain in PyTorch format."),
                mapping_prefix)
            sys.exit(0)

    num_proc = args.num_processes if args.num_processes > 0 else 1
    col_name_map = None
    if args.column_names is not None:
        col_name_map = {}
        # Load customized column names
        for col_rename_pair in args.column_names:
            # : has special meaning in Graph Database like Neptune
            # Here, we use ,  as the delimiter.
            orig_name, new_name = col_rename_pair.split(",")
            assert orig_name in GS_REMAP_BUILTIN_COLS, \
                f"Expect the original col name is in {GS_REMAP_BUILTIN_COLS}, " \
                f"but get {orig_name}"
            col_name_map[orig_name] = new_name
    if args.output_format == GS_OUTPUT_FORMAT_PARQUET:
        output_func = partial(write_data_parquet_file,
                              col_name_map=col_name_map)
    elif args.output_format == GS_OUTPUT_FORMAT_CSV:
        output_func = partial(write_data_csv_file,
                              delimiter=args.output_delimiter,
                              col_name_map=col_name_map)
    else:
        raise TypeError(f"Output format not supported {args.output_format}")

    files_to_remove = []
    if len(emb_ntypes) > 0:
        emb_output = node_emb_dir
        # We need to do ID remapping for node embeddings
        emb_files_to_remove = \
            remap_node_emb(emb_ntypes,
                           node_emb_dir,
                           emb_output,
                           out_chunk_size,
                           num_proc,
                           rank,
                           world_size,
                           with_shared_fs,
                           output_func)
        files_to_remove += emb_files_to_remove

        for task_emb_dir in task_emb_dirs:
            task_emb_dir = os.path.join(node_emb_dir, task_emb_dir)
            # We need to do ID remapping for node embeddings
            emb_files_to_remove = \
                remap_node_emb(emb_ntypes,
                               task_emb_dir,
                               task_emb_dir,
                               out_chunk_size,
                               num_proc,
                               rank,
                               world_size,
                               with_shared_fs,
                               output_func)
            files_to_remove += emb_files_to_remove

    if len(pred_etypes) > 0:
        if isinstance(predict_dir, tuple):
            _, edge_predict_dirs = predict_dir
            # In multi-task learning,
            # each edge predict task only does prediction on one edge type
            edge_pred_etypes = [[pred_etype] for pred_etype in pred_etypes]
        else:
            edge_predict_dirs = [predict_dir]
            edge_pred_etypes = [pred_etypes]

        for pred_dir, pred_et in zip(edge_predict_dirs, edge_pred_etypes):
            pred_output = pred_dir
            # We need to do ID remapping for edge prediction result
            pred_files_to_remove = \
                remap_edge_pred(pred_et,
                                pred_dir,
                                pred_output,
                                out_chunk_size,
                                num_proc,
                                rank,
                                world_size,
                                with_shared_fs,
                                output_func)
            files_to_remove += pred_files_to_remove

    if len(pred_ntypes) > 0:
        if isinstance(predict_dir, tuple):
            node_predict_dirs, _ = predict_dir
            # In multi-task learning,
            # each node predict task only does prediction on one node type
            node_pred_ntypes = [[pred_ntype] for pred_ntype in pred_ntypes]
        else:
            node_predict_dirs = [predict_dir]
            node_pred_ntypes = [pred_ntypes]

        for pred_dir, pred_nt in zip(node_predict_dirs, node_pred_ntypes):
            pred_output = pred_dir
            # We need to do ID remapping for node prediction result
            pred_files_to_remove = \
                remap_node_pred(pred_nt,
                                pred_dir,
                                pred_output,
                                out_chunk_size,
                                num_proc,
                                rank,
                                world_size,
                                with_shared_fs,
                                output_func)
            files_to_remove += pred_files_to_remove

    if args.preserve_input is False and len(files_to_remove) > 0:
        # If files_to_remove is not empty, at least node_emb_dir or
        # predict_dir is not None.
        _remove_inputs(with_shared_fs, files_to_remove, rank, world_size,
                       node_emb_dir if node_emb_dir is not None else predict_dir)

def add_distributed_remap_args(parser):
    """ Distributed remapping only

        The arguments under this argument group are mainly
        designed for distributed remapping results.
        Users can ignore arguments in this argument group if
        they are not doing distributed remapping.

        Parameter
        ---------
        parser: argparse.ArgumentParser
            Argument parser

        Return
        ------
        parser: Argument parser
    """
    group = parser.add_argument_group(title="dist_remap")
    group.add_argument("--with-shared-fs",
                       type=lambda x: (str(x).lower() in ['true', '1']),default=True,
                       help="Whether all files are stored in a shared file system"
                            "False when it is running on SageMaker")
    group.add_argument("--rank", type=int, default=0,
                       help="The rank of current worker.")
    group.add_argument("--world-size", type=int, default=1,
                       help="Total number of workers in the cluster.")
    return parser

def generate_parser():
    """ Generate an argument parser
    """
    parser = argparse.ArgumentParser("Remapping graph node IDs")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="pointer to the yaml configuration file of the experiment",
        type=str,
        default=None
    )

    group = parser.add_argument_group(title="remap")
    group.add_argument("--num-processes", type=int, default=4,
                       help="The number of processes to process the data simultaneously.")
    group.add_argument("--node-id-mapping", type=str,
                       help="The directory storing the id mappings")
    group.add_argument("--prediction-dir", type=str,
                       default=None,
                       help="The directory storing the graph prediction results.")
    group.add_argument("--node-emb-dir", type=str,
                       default=None,
                       help="The directory storing the node embeddings.")
    group.add_argument("--output-format", type=str,
                       default=GS_OUTPUT_FORMAT_PARQUET,
                       choices=[GS_OUTPUT_FORMAT_PARQUET, GS_OUTPUT_FORMAT_CSV],
                       help="The format of the output.")
    group.add_argument("--output-delimiter", type=str, default=",",
                       help="The delimiter used when saving data in CSV format.")
    group.add_argument("--column-names", type=str, nargs="+", default=None,
                       help="Defines how to rename default column names to new names."
                       f"For example, given --column-names {GS_REMAP_NID_COL},~id "
                       f"{GS_REMAP_EMBED_COL},embedding. The column "
                       f"{GS_REMAP_NID_COL} will be renamed to ~id. "
                       f"The column {GS_REMAP_EMBED_COL} will be renamed to embedding.")
    group.add_argument("--logging-level", type=str, default="info",
                       help="The logging level. The possible values: debug, info, warning, \
                                   error. The default value is info.")

    group.add_argument("--pred-etypes", type=str, nargs="+", default=None,
                       help="[Optional] A list of canonical edge types which have"
                                "prediction results For example, "
                                "--pred-etypes user,rate,movie user,watch,movie"
                                "If pred_etypes is not provided, result_info.json "
                                "under prediction_dir will be used to retrive the pred_etypes")
    group.add_argument("--pred-ntypes", type=str, nargs="+", default=None,
                       help="[Optional] A list of node types which have"
                                "prediction results For example, "
                                "--pred-ntypes user movie"
                                "If pred_ntypes is not provided, result_info.json "
                                "under prediction_dir will be used to retrive the pred_ntypes")
    group.add_argument("--output-chunk-size", type=int, default=sys.maxsize,
                       help="Number of rows per output file."
                       f"By default, it is set to {sys.maxsize}")
    group.add_argument("--preserve-input",
                       type=lambda x: (str(x).lower() in ['true', '1']),default=False,
                       help="Whether we preserve the input data.")
    return add_distributed_remap_args(parser)

if __name__ == '__main__':
    remap_parser = generate_parser()
    remap_args, unknown_args = remap_parser.parse_known_args()

    main(remap_args, unknown_args)
