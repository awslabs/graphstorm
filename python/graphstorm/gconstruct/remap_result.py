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
                      BUILTIN_TASK_NODE_REGRESSION)

# Id_maps is a global variable.
# When using multi-processing to do id remap,
# we do not want to pass id_maps to each worker process
# through argument which uses Python pickle to copy
# data. By making id_maps as a global variable, we
# can rely on Linux copy-on-write to provide a zero-copy
# id_maps to each worker process.
id_maps = {}

def worker_remap_node_data(data_file_path, nid_path, ntype, data_col_key,
    output_fname_prefix, chunk_size, preserve_input):
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
        preserve_input: bool
            Whether the input data should be removed.
    """
    node_data = th.load(data_file_path).numpy()
    nids = th.load(nid_path).numpy()
    nid_map = id_maps[ntype]
    num_chunks = math.ceil(len(node_data) / chunk_size)

    for i in range(num_chunks):
        output_fname = f"{output_fname_prefix}_{pad_file_index(i)}.parquet"

        start = i * chunk_size
        end = (i + 1) * chunk_size if i + 1 < num_chunks else len(node_data)
        data = node_data[start:end]
        nid = nid_map.map_id(nids[start:end])
        data = {data_col_key: data,
                "nid": nid}

        write_data_parquet(data, output_fname)

    if preserve_input is False:
        os.remove(data_file_path)
        os.remove(nid_path)

def worker_remap_edge_pred(pred_file_path, src_nid_path,
    dst_nid_path, src_type, dst_type,
    output_fname_prefix, chunk_size, preserve_input):
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
        preserve_input: bool
            Whether the input data should be removed.
    """
    pred_result = th.load(pred_file_path).numpy()
    src_nids = th.load(src_nid_path).numpy()
    dst_nids = th.load(dst_nid_path).numpy()
    src_id_map = id_maps[src_type]
    dst_id_map = id_maps[dst_type]
    num_chunks = math.ceil(len(pred_result) / chunk_size)
    for i in range(num_chunks):
        output_fname = f"{output_fname_prefix}_{pad_file_index(i)}.parquet"

        start = i * chunk_size
        end = (i + 1) * chunk_size if i + 1 < num_chunks else len(pred_result)
        pred = pred_result[start:end]
        src_nid = src_id_map.map_id(src_nids[start:end])
        dst_nid = dst_id_map.map_id(dst_nids[start:end])
        data = {"pred": pred,
                "src_nid": src_nid,
                "dst_nid": dst_nid}

        write_data_parquet(data, output_fname)

    if preserve_input is False:
        os.remove(pred_file_path)
        os.remove(src_nid_path)
        os.remove(dst_nid_path)

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

def remap_node_emb(emb_ntypes, node_emb_dir,
                   output_dir, out_chunk_size,
                   num_proc, rank,  world_size,
                   with_shared_fs, preserve_input=False):
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
            Whether shared file system is avaliable
        preserve_input: bool
            Whether the input data should be removed.
    """
    task_list = []
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
                "data_col_key": "emb",
                "output_fname_prefix": os.path.join(out_embdir, \
                    f"{emb_file[:emb_file.rindex('.')]}"),
                "chunk_size": out_chunk_size,
                "preserve_input": preserve_input
            })

    multiprocessing_remap(task_list, num_proc, worker_remap_node_data)

def remap_node_pred(pred_ntypes, pred_dir,
                    output_dir, out_chunk_size,
                    num_proc, rank, world_size, with_shared_fs,
                    preserve_input=False):
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
            Whether shared file system is avaliable
        preserve_input: bool
            Whether the input data should be removed.
    """
    start_time = time.time()
    task_list = []
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
                "data_col_key": "pred",
                "output_fname_prefix": os.path.join(out_pred_dir, \
                    f"pred.{pred_file[:pred_file.rindex('.')]}"),
                "chunk_size": out_chunk_size,
                "preserve_input": preserve_input
            })

    multiprocessing_remap(task_list, num_proc, worker_remap_node_data)

    dur = time.time() - start_time
    logging.info("{%d} Remapping edge predictions takes {%f} secs", rank, dur)


def remap_edge_pred(pred_etypes, pred_dir,
                    output_dir, out_chunk_size,
                    num_proc, rank, world_size, with_shared_fs,
                    preserve_input=False):
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
            Whether shared file system is avaliable
        preserve_input: bool
            Whether the input data should be removed.
    """
    start_time = time.time()
    task_list = []
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
                "preserve_input": preserve_input
            })

    multiprocessing_remap(task_list, num_proc, worker_remap_edge_pred)
    dur = time.time() - start_time
    logging.debug("%d Finish edge rempaing in %f secs}", rank, dur)

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
    node_id_mapping = os.path.dirname(part_config)
    predict_dir = config.save_prediction_path
    emb_dir = config.save_embed_path
    task_type = config.task_type
    pred_ntypes = []
    pred_etypes = []
    if task_type in (BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION):
        pred_etypes = config.target_etype
        pred_etypes = pred_etypes \
            if isinstance(pred_etypes, list) else [pred_etypes]
        pred_etypes = [list(pred_etype) for pred_etype in pred_etypes]
    elif task_type in (BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION):
        pred_ntypes = config.target_ntype
        pred_ntypes = pred_ntypes \
            if isinstance(pred_ntypes, list) else [pred_ntypes]

    return node_id_mapping, predict_dir, emb_dir, pred_ntypes, pred_etypes

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
        gs_args = gs_parser.parse_args(gs_config_args)
        config = GSConfig(gs_args)
        config.verify_arguments(False)
        id_mapping_path, predict_dir, node_emb_dir, pred_ntypes, pred_etypes = \
            _parse_gs_config(config)
    else:
        # Case 2: remap_result is called alone.
        # GraphStorm train/inference configs are not avaliable.
        # We collect information from input arguments.
        logging.basicConfig(level=get_log_level(args.logging_level), force=True)
        id_mapping_path = args.node_id_mapping
        predict_dir = args.prediction_dir
        node_emb_dir = args.node_emb_dir
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
            # When shared file system is not avaliable and the world_size is
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
            emb_names = [e_name for e_name in emb_names if e_name != "emb_info.json"]

            emb_ntypes = emb_names

    ################## remap prediction #############
    if predict_dir is not None:
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
    else:
        pred_etypes = []
        pred_ntypes = []


    ntypes = []
    ntypes += emb_ntypes
    ntypes += [etype[0] for etype in pred_etypes] + \
        [etype[2] for etype in pred_etypes]
    ntypes += pred_ntypes

    if len(ntypes) == 0:
        # Nothing to remap
        logging.warning("Skip remapping edge/node predictions and node embeddings")
        return

    for ntype in set(ntypes):
        mapping_file = os.path.join(id_mapping_path, ntype + "_id_remap.parquet")
        logging.debug("loading mapping file %s",
                      mapping_file)
        if os.path.exists(mapping_file):
            id_maps[ntype] = \
                IdReverseMap(mapping_file)
        else:
            logging.warning("ID mapping file %s does not exists, skip remapping", mapping_file)
            return

    num_proc = args.num_processes if args.num_processes > 0 else 1

    if len(emb_ntypes) > 0:
        emb_output = node_emb_dir
        # We need to do ID remapping for node embeddings
        remap_node_emb(emb_ntypes,
                       node_emb_dir,
                       emb_output,
                       out_chunk_size,
                       num_proc,
                       rank,
                       world_size,
                       with_shared_fs,
                       args.preserve_input)

    if len(pred_etypes) > 0:
        pred_output = predict_dir
        # We need to do ID remapping for edge prediction result
        remap_edge_pred(pred_etypes,
                        predict_dir,
                        pred_output,
                        out_chunk_size,
                        num_proc,
                        rank,
                        world_size,
                        with_shared_fs,
                        args.preserve_input)

    if len(pred_ntypes) > 0:
        pred_output = predict_dir
        # We need to do ID remapping for node prediction result
        remap_node_pred(pred_ntypes,
                        predict_dir,
                        pred_output,
                        out_chunk_size,
                        num_proc,
                        rank,
                        world_size,
                        with_shared_fs,
                        args.preserve_input)


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
                       help="Totoal number of workers in the cluster.")
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
                       default="parquet",
                       choices=['parquet'],
                       help="The format of the output.")
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
