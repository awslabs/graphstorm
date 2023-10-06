import os
import argparse
import logging
import json
import time
import multiprocessing
import queue
import math

import torch as th
from torch import multiprocessing
from torch.multiprocessing import Process
from ..model.utils import pad_file_index
from .file_io import write_data_parquet
from .id_map import IdReverseMap
from ..utils import get_log_level

from graphstorm.config import (GSConfig,
                               get_argument_parser,
                               BUILTIN_TASK_EDGE_CLASSIFICATION,
                               BUILTIN_TASK_EDGE_REGRESSION,
                               BUILTIN_TASK_NODE_CLASSIFICATION,
                               BUILTIN_TASK_NODE_REGRESSION)

from .utils import SHARED_MEM_OBJECT_THRESHOLD

def worker_fn(worker_id, task_queue, func):
    """ Process remap tasks with multiprocessing
    """
    try:
        while True:
            # If the queue is empty, it will raise the Empty exception.
            idx, task_args = task_queue.get_nowait()
            logging.debug("worker %d Processing %s task", worker_id, idx)
            func(**task_args)
    except queue.Empty:
        pass

def worker_remap_edge_pred(pred_file_path, src_nid_path,
    dst_nid_path, src_id_map, dst_id_map,
    output_fname_prefix, chunk_size, preserve_input):
    pred_result = th.load(pred_file_path).numpy()
    src_nids = th.load(src_nid_path).numpy()
    dst_nids = th.load(dst_nid_path).numpy()

    num_chunks = math.ceil(len(pred_result) / chunk_size)
    for i in range(num_chunks):
        output_fname = f"{output_fname_prefix}_{pad_file_index(i)}.parquet"

        start = i * num_chunks
        end = (i + 1) * num_chunks if i + 1 < num_chunks else len(pred_result)
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

def multiprocessing_remap(tasks, num_processes, remap_func):
    if num_processes > 1 and len(tasks) > 1:
        processes = []
        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        for i, task in enumerate(tasks):
            task_queue.put((i, task))

        for i in range(num_processes):
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
                    num_proc, rank, world_size, id_maps, with_shared_fs,
                    preserve_input=False):
    """
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
        logging.debug(f"{etype} has {num_parts} embedding files")
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

        for i in range(start, end):
            pred_file = pred_files[i]
            src_nid_file = src_nid_files[i]
            dst_nid_file = dst_nid_files[i]
            src_nid_map = id_maps[etype[0]]
            # if src ntype == dst ntype, there is no need to
            # pickle nid mappings twice
            dst_nid_map = id_maps[etype[2]] if etype[0] != etype[2] else None
            nid_mapping_size = src_nid_map.size + \
                dst_nid_map.size if dst_nid_map is not None else 0

            # Max pickle obj size is 2 GByte
            # We need to handle the case when orig_nids > 2 GByte
            if nid_mapping_size > SHARED_MEM_OBJECT_THRESHOLD:
                num_proc = 1

            # file is in format of emb.part00000.bin
            # the output emb files will be
            #     emb.part00000_00000.parquet
            #     emb.part00000_00001.parquet
            #     ...
            task_list.append({
                "pred_file_path": os.path.join(input_pred_dir, pred_file),
                "src_nid_path": os.path.join(input_pred_dir, src_nid_file),
                "dst_nid_path": os.path.join(input_pred_dir, dst_nid_file),
                "src_id_map": src_nid_map,
                "dst_id_map": dst_nid_map,
                "output_fname_prefix": os.path.join(out_pred_dir, pred_file[:pred_file.rindex(".")]),
                "chunk_size": out_chunk_size,
                "preserve_input": preserve_input
            })

    multiprocessing_remap(task_list, num_proc, worker_remap_edge_pred)
    dur = time.time() - start

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
    elif task_type in (BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION):
        pred_ntypes = config.target_ntype
        pred_ntypes = pred_ntypes \
            if isinstance(pred_ntypes, list) else [pred_ntypes]

    return node_id_mapping, predict_dir, emb_dir, pred_ntypes, pred_etypes

def main(args, gs_config_args):
    """ main function
    """
    if args.yaml_config_file is not None:
        gs_config_args = ["--cf", args.yaml_config_file,
                          "--logging-level", args.logging_level] + gs_config_args
        gs_parser = get_argument_parser()
        gs_args = gs_parser.parse_args(gs_config_args)
        config = GSConfig(gs_args)
        config.verify_arguments(False)
        id_mapping_path, predict_dir, _, _, pred_etypes = \
            _parse_gs_config(config)
    else:
        logging.basicConfig(level=get_log_level(args.logging_level), force=True)
        id_mapping_path = args.node_id_mapping
        predict_dir = args.prediction_dir
        pred_etypes = args.pred_etypes

    rank = args.rank
    world_size = args.world_size
    with_shared_fs = args.with_shared_fs

    assert world_size > 0, \
        f"World size must be larger than 0, but get {world_size}."
    assert rank < world_size, \
        f"Expecting {world_size} workers but the worker ID is {rank}"
    out_chunk_size = args.output_chunk_size
    assert out_chunk_size > 0, \
        f"Output chunk size should larger than 0 but get {out_chunk_size}"

    if pred_etypes is not None:
        assert len(pred_etypes) > 0, \
            f"prediction etypes is empty"
        pred_etypes = [etype.split(",") for etype in pred_etypes]

        for pred_etype in pred_etypes:
            assert os.path.exists(os.path.join(predict_dir, "_".join(pred_etype))), \
                f"prediction results of {pred_etype} do not exists"
    # Note: For distributed SageMaker runs
    # pred_etypes must be provided as result_info.json
    # is only saved by rank0 and there is no shared fs
    elif os.path.exists(os.path.join(predict_dir, "result_info.json")):
        # User does not provide pred_etypes.
        # Try to get it from saved prediction config.
        with open(os.path.join(predict_dir, "result_info.json"),
                    "r",  encoding='utf-8') as f:
            info = json.load(f)
            pred_etypes = [etype.split(",") for etype in info["etypes"]] \
                    if "etypes" in info else []

    id_maps = {}
    ntypes = \
        [etype[0] for etype in pred_etypes] + \
        [etype[2] for etype in pred_etypes]
    for ntype in set(ntypes):
        logging.debug("loading mapping file %s",
                      os.path.join(id_mapping_path, ntype + "_id_remap.parquet"))
        id_maps[ntype] = \
            IdReverseMap(os.path.join(id_mapping_path, ntype + "_id_remap.parquet"))

    num_proc = args.num_processes if args.num_processes > 0 else 1

    pred_output = predict_dir
    if len(pred_etypes) > 0:
        # We need to do ID remapping for edge prediction result
        remap_edge_pred(pred_etypes,
                        predict_dir,
                        pred_output,
                        out_chunk_size,
                        num_proc,
                        rank,
                        world_size,
                        id_maps,
                        with_shared_fs,
                        args.preserve_input)


def _add_distributed_remap_args(parser):
    """ Distributed remapping only

        Users can ignore arguments in this argument group.
        The arguments under this argument graph are mainly
        designed for distributed remapping results in SageMaker,
        where a shared file system is not avaliable.
    """
    group = parser.add_argument_group(title="dist_remap")
    group.add_argument("--with-shared-fs", type=bool, default=True,
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
    parser = argparse.ArgumentParser("Preprocess graphs")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="pointer to the yaml configuration file of the experiment",
        type=str,
        default=None
    )

    group = parser.add_argument_group(title="remap")
    group.add_argument("--num-processes", type=int, default=4,
                       help="The number of processes to process the data simulteneously.")
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
                       help="Number of raws per output file.")
    group.add_argument("--preserve-input", type=bool, default=False,
                       help="Whether we preserve the input data.")
    parser = _add_distributed_remap_args(parser)
    return parser

if __name__ == '__main__':
    parser = generate_parser()
    args, gs_config_args = parser.parse_known_args()

    main(args, gs_config_args)
