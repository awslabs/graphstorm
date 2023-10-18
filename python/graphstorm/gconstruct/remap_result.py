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
    into the original node ID space.
"""

import os
import argparse
import logging
import json
import time
import queue
import math

import torch as th
from torch import multiprocessing
from torch.multiprocessing import Process
from ..model.utils import pad_file_index
from .file_io import write_data_parquet
from .id_map import IdReverseMap
from ..utils import get_log_level

from ..config import (GSConfig,
                      get_argument_parser,
                      BUILTIN_TASK_EDGE_CLASSIFICATION,
                      BUILTIN_TASK_EDGE_REGRESSION,
                      BUILTIN_TASK_NODE_CLASSIFICATION,
                      BUILTIN_TASK_NODE_REGRESSION)

id_maps = {}

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
        end = (i + 1) * chunk_size if i + 1 < chunk_size else len(pred_result)
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

def worker_fn(worker_id, task_queue, func):
    """ Process remap tasks with multiprocessing

        Parameters
        ----------
        worker_id: int
            Worker id.
        task_queue: Queue
            Task queue.
        func: function
            Function to be executed.
    """
    try:
        while True:
            # If the queue is empty, it will raise the Empty exception.
            idx, task_args = task_queue.get_nowait()
            logging.debug("worker %d Processing %s task", worker_id, idx)
            func(**task_args)
    except queue.Empty:
        pass

def multiprocessing_remap(tasks, num_proc, remap_func):
    """ Do multi-processing remap

        Parameters
        ----------
        task: list
            List of remap tasks.
        num_proc: int
            Number of workers to spin up.
        remap_func: func
            Reampping function
    """
    if num_proc > 1 and len(tasks) > 1:
        if num_proc > len(tasks):
            num_proc = len(tasks)
        processes = []
        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        for i, task in enumerate(tasks):
            task_queue.put((i, task))

        for i in range(num_proc):
            proc = Process(target=worker_fn, args=(i, task_queue, remap_func))
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()
    else:
        for i, task_args in enumerate(tasks):
            logging.debug("worker 0 Processing %s task", i)
            remap_func(**task_args)

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

def remap_edge_pred(pred_etypes, pred_dir,
                    output_dir, out_chunk_size,
                    num_proc, rank, world_size, with_shared_fs,
                    preserve_input=False):
    """ Remap edge prediction result.

        The function wil iterate all the edge types that
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

        The output emb files will be
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
    start = time.time()
    task_list = []
    for etype in pred_etypes:
        input_pred_dir = os.path.join(pred_dir, "_".join(etype))
        out_pred_dir = os.path.join(output_dir, "_".join(etype))
        etype_pred_files = os.listdir(input_pred_dir)
        src_nid_files = [fname for fname in etype_pred_files if fname.startswith("src_nids")]
        dst_nid_files = [fname for fname in etype_pred_files if fname.startswith("dst_nids")]
        pred_files = [fname for fname in etype_pred_files if fname.startswith("predict")]
        src_nid_files.sort()
        dst_nid_files.sort()
        pred_files.sort()

        num_parts = len(pred_files)
        logging.debug("%s has %d embedding files", etype, num_parts)
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
    dur = time.time() - start

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
    pred_ntypes = None
    pred_etypes = None
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
        id_mapping_path, predict_dir, _, _, pred_etypes = \
            _parse_gs_config(config)
    else:
        # Case 2: remap_result is called alone.
        # GraphStorm train/inference configs are not avaliable.
        # We collect information from input arguments.
        logging.basicConfig(level=get_log_level(args.logging_level), force=True)
        id_mapping_path = args.node_id_mapping
        predict_dir = args.prediction_dir
        pred_etypes = args.pred_etypes
        if pred_etypes is not None:
            assert len(pred_etypes) > 0, \
                "prediction etypes is empty"
            pred_etypes = [etype.split(",") for etype in pred_etypes]

    rank = args.rank
    world_size = args.world_size
    with_shared_fs = args.with_shared_fs

    assert world_size > 0, \
        f"World size must be larger than 0, but get {world_size}."
    assert rank < world_size, \
        f"Expecting {world_size} workers but the worker ID is {rank}."
    out_chunk_size = args.output_chunk_size
    assert out_chunk_size > 0, \
        f"Output chunk size should larger than 0 but get {out_chunk_size}."

    # if pred_etypes (edges with prediction results)
    # is not None, We need to remap edge prediction results.
    # Note: For distributed SageMaker runs, pred_etypes must be
    # provided if edge prediction result rempa is required,
    # as result_info.json is only saved by rank0 and there is no shared fs.
    if pred_etypes is not None:
        for pred_etype in pred_etypes:
            assert os.path.exists(os.path.join(predict_dir, "_".join(pred_etype))), \
                f"prediction results of {pred_etype} do not exists."
    elif os.path.exists(os.path.join(predict_dir, "result_info.json")):
        # User does not provide pred_etypes.
        # Try to get it from saved prediction config.
        with open(os.path.join(predict_dir, "result_info.json"),
                    "r",  encoding='utf-8') as f:
            info = json.load(f)
            pred_etypes = [etype.split(",") for etype in info["etypes"]] \
                    if "etypes" in info else []

    ntypes = \
        [etype[0] for etype in pred_etypes] + \
        [etype[2] for etype in pred_etypes]
    for ntype in set(ntypes):
        logging.debug("loading mapping file %s",
                      os.path.join(id_mapping_path, ntype + "_id_remap.parquet"))
        id_maps[ntype] = \
            IdReverseMap(os.path.join(id_mapping_path, ntype + "_id_remap.parquet"))

    num_proc = args.num_processes if args.num_processes > 0 else 1

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


def add_distributed_remap_args(parser):
    """ Distributed remapping only

        Users can ignore arguments in this argument group.
        The arguments under this argument group are mainly
        designed for distributed remapping results in SageMaker,
        where a shared file system is not avaliable.

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
                       help="The directory storing the graph prediction results.")
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
    group.add_argument("--output-chunk-size", type=int, default=100000,
                       help="Number of rows per output file.")
    group.add_argument("--preserve-input",
                       type=lambda x: (str(x).lower() in ['true', '1']),default=False,
                       help="Whether we preserve the input data.")
    return add_distributed_remap_args(parser)

if __name__ == '__main__':
    remap_parser = generate_parser()
    remap_args, gs_config_args = remap_parser.parse_known_args()

    main(remap_args, gs_config_args)
