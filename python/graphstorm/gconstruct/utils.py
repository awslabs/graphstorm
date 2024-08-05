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
import queue
import gc
import logging
import copy
import traceback
import shutil
import uuid

import numpy as np
import dgl
import torch as th
from torch import multiprocessing
from torch.multiprocessing import Process

from ..utils import sys_tracker

SHARED_MEM_OBJECT_THRESHOLD = 1.9 * 1024 * 1024 * 1024 # must < 2GB
SHARED_MEMORY_CROSS_PROCESS_STORAGE = "shared_memory"
PICKLE_CROSS_PROCESS_STORAGE = "pickle"
EXT_MEMORY_STORAGE = "ext_memory"

def _is_numeric(arr):
    """ Check if the input array has the numeric data type.
    """
    return np.issubdtype(arr.dtype, np.number) or arr.dtype == bool

def _to_ext_memory(name, data, path):
    if isinstance(data, np.ndarray):
        assert name is not None
        path = os.path.join(path, f"{name}.npy")
        # We only save a data array with numeric or boolean values to the disk.
        # This avoids the problem of saving an array of objects to disks.
        # Note: There is a bug in Numpy when saving an array of objects to disks.
        if len(data) > 0 and _is_numeric(data):
            logging.debug("save data %s in %s.", name, path)
            data = convert_to_ext_mem_numpy(path, data)
            # We need to pass the array to another process. We don't want it
            # to reference to data in the file.
            data.cleanup()
        return data
    elif isinstance(data, dict):
        new_data = {}
        for key, val in data.items():
            new_data[key] = _to_ext_memory(key, val,
                    os.path.join(path, name) if name is not None else path)
        return new_data
    elif isinstance(data, list):
        new_data = []
        for i, val in enumerate(data):
            new_data.append(_to_ext_memory(f"item-{i}", val,
                os.path.join(path, name) if name is not None else path))
        return new_data
    elif isinstance(data, tuple):
        new_data = []
        for i, val in enumerate(list(data)):
            new_data.append(_to_ext_memory(f"item-{i}", val,
                os.path.join(path, name) if name is not None else path))
        return tuple(new_data)
    else:
        return data

def _to_shared_memory(data):
    """ Move all tensor objects into torch shared memory

    Parameters
    ----------
    data: dict/tuple/list of tensors
        Data returned by user_parser
    """
    if th.is_tensor(data):
        return data.share_memory_()
    elif isinstance(data, np.ndarray):
        assert data.dtype is not np.object_, \
            "Numpy array of python objects can not be handled by graph construction"

        # only handle data that can be converted to torch tensor
        if data.dtype in [np.float64, np.float32, np.float16,
                          np.complex64, np.complex128, np.int64,
                          np.int32, np.int16, np.int8,
                          np.uint8]:
            return th.tensor(data).share_memory_()
        return data
    elif isinstance(data, dict):
        new_data = {}
        for name, val in data.items():
            new_data[name] = _to_shared_memory(val)
        return new_data
    elif isinstance(data, list):
        new_data = []
        for val in data:
            new_data.append(_to_shared_memory(val))
        return new_data
    elif isinstance(data, tuple):
        new_data = []
        for val in list(data):
            new_data.append(_to_shared_memory(val))
        return tuple(new_data)

    # ignore other types
    return data

def _to_numpy_array(data):
    """ Move all data objects back to numpy array

    Parameters
    ----------
    data: dict/tuple/list of tensors
        Data returned by user_parser
    """
    if th.is_tensor(data):
        return data.numpy()
    elif isinstance(data, np.ndarray):
        return data # do nothing
    elif isinstance(data, dict):
        new_data = {}
        for name, val in data.items():
            new_data[name] = _to_numpy_array(val)
        return new_data
    elif isinstance(data, list):
        new_data = []
        for val in data:
            new_data.append(_to_numpy_array(val))
        return new_data
    elif isinstance(data, tuple):
        new_data = []
        for val in list(data):
            new_data.append(_to_numpy_array(val))
        return tuple(new_data)

    # ignore other types
    return data

def _estimate_sizeof(data):
    """ Estimate the size of a data.
        We assume the most memory consuming objects are tensors.

    Parameters
    ----------
    data: dict/tuple/list of tensors
        Data returned by user_parser
    """
    if th.is_tensor(data):
        data_size = data.element_size() * data.nelement()
    elif isinstance(data, np.ndarray):
        assert data.dtype is not np.object_, \
            "Numpy array of python objects can not be handled by graph construction"

        data_size = data.size * data.itemsize
    elif isinstance(data, dict):
        data_size = 0
        for _, val in data.items():
            data_size += _estimate_sizeof(val)
    elif isinstance(data, list):
        data_size = 0
        for val in data:
            data_size += _estimate_sizeof(val)
    elif isinstance(data, tuple):
        data_size = 0
        for val in list(data):
            data_size += _estimate_sizeof(val)
    else:
        # for other types like primitives
        # ignore their size.
        data_size = 0

    return data_size

def generate_hash():
    """ Generate unique hashcode
    """
    random_uuid = uuid.uuid4()
    return str(random_uuid)

def worker_fn(worker_id, task_queue, res_queue, user_parser, ext_mem_workspace):
    """ The worker function in the worker pool

    Parameters
    ----------
    worker_id : int
        The worker ID, starting from 0.
    task_queue : Queue
        The queue that contains all tasks
    res_queue : Queue
        The queue that contains the processed data. This is used for
        communication between the worker processes and the master process.
    user_parser : callable
        The user-defined function to read and process the data files.
    ext_mem_workspace : str
        The path of the external-memory work space.
    """
    # We need to set a GPU device for each worker process in case that
    # some transformations (e.g., computing BERT embeddings) require GPU computation.
    if th.cuda.is_available():
        num_gpus = th.cuda.device_count()
        gpu = worker_id % num_gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        if worker_id >= num_gpus:
            logging.warning("There are more than 1 processes are attachd to GPU %d.", gpu)
    try:
        i = 0
        while True:
            # If the queue is empty, it will raise the Empty exception.
            i, in_file = task_queue.get_nowait()
            logging.debug("%d Processing %s", worker_id, in_file)
            data = user_parser(in_file)
            size = _estimate_sizeof(data)
            if ext_mem_workspace is not None:
                data = (EXT_MEMORY_STORAGE, _to_ext_memory(f"file-{i}", data, ext_mem_workspace))
            # Max pickle obj size is 2 GByte
            elif size > SHARED_MEM_OBJECT_THRESHOLD:
                # Use torch shared memory as a workaround
                # This will consume shared memory and cause an additional
                # data copy, i.e., general memory to torch shared memory.
                data = (SHARED_MEMORY_CROSS_PROCESS_STORAGE, _to_shared_memory(data))
            else:
                data = (PICKLE_CROSS_PROCESS_STORAGE, data)
            res_queue.put((i, data))
            gc.collect()
    except queue.Empty:
        pass
    except Exception as e:  # pylint: disable=broad-exception-caught
        e = ''.join(traceback.TracebackException.from_exception(e).format())
        res_queue.put((i, e))

def update_two_phase_feat_ops(phase_one_info, ops):
    """ Update the ops for the second phase feat processing

    Parameters
    ----------
    phase_one_info: dict
        A dict mapping file index to node/edge features info corresponding to ops.
    ops: dict of FeatTransform
        The operations run on the node/edge features of the node/edge files.
    """
    feat_info = {}
    for _, finfo in phase_one_info.items():
        for feat_name, info in finfo.items():
            if feat_name not in feat_info:
                feat_info[feat_name] = [info]
            else:
                feat_info[feat_name].append(info)
    for op in ops:
        # It is possible that there is no information from phase one.
        if op.feat_name in feat_info:
            op.update_info(feat_info[op.feat_name])

def multiprocessing_data_read(in_files, num_processes, user_parser, ext_mem_workspace=None):
    """ Read data from multiple files with multiprocessing.

    It creates a set of worker processes, each of which runs a worker function.
    It first adds all tasks in a queue and each worker gets a task from the queue
    at a time until the workers process all tasks. It maintains a result queue
    and each worker adds the processed results in the result queue.
    To ensure the order of the data, each data file is associated with
    a number and the processed data are ordered by that number.

    If there are only one input file, it reads the data from the input file in the main process.

    Parameters
    ----------
    in_files : list of str
        The input data files.
    num_processes : int
        The number of processes that run in parallel.
    user_parser : callable
        The user-defined function to read and process the data files.
    ext_mem_workspace : str
        The path of the external-memory work space.

    Returns
    -------
    a dict : key is the file index, the value is processed data.
    """
    if num_processes > 1 and len(in_files) > 1:
        processes = []
        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        res_queue = manager.Queue(8 if num_processes < 8 else num_processes)
        num_files = len(in_files)
        for i, in_file in enumerate(in_files):
            task_queue.put((i, in_file))
        for i in range(num_processes):
            proc = Process(target=worker_fn, args=(i, task_queue, res_queue, user_parser,
                                                   ext_mem_workspace))
            proc.start()
            processes.append(proc)

        return_dict = {}
        while len(return_dict) < num_files:
            try:
                file_idx, vals= res_queue.get(timeout=3600)
            except queue.Empty:
                # check whether every processes are alive
                for proc in processes:
                    if not proc.is_alive() and proc.exitcode < 0:
                        raise RuntimeError("One of the work process crashed with"
                                           f"{proc.exitcode}. In most of cases, it is "
                                           "due to out-of-memory. Please check your "
                                           "instance memory size and the shared memory "
                                           "size.") from None
                logging.warning("One of the processes has been processing the "
                                "input data for more than one hour. This will "
                                "not cause any error but please check whether "
                                "the input data files are too large. "
                                "We suggest you to spit the file(s) into "
                                "smaller chunks.")
                continue

            if not isinstance(vals, tuple):
                logging.error("Processing file %d fails.", file_idx)
                logging.error(vals)
                raise RuntimeError("One of the worker processes fails. Stop processing.")
            # If the size of `vals`` is larger than utils.SHARED_MEM_OBJECT_THRESHOLD
            # we will automatically convert tensors in `vals` into torch tensor
            # and copy the tensor into shared memory.
            # This helps avoid the pickle max obj size issue.
            storage_type, vals = vals
            if storage_type == SHARED_MEMORY_CROSS_PROCESS_STORAGE:
                vals = _to_numpy_array(vals)
            return_dict[file_idx] = vals
            sys_tracker.check(f'process data file: {file_idx}')
            gc.collect()

        for proc in processes:
            proc.join()

        return return_dict
    else:
        return_dict = {}
        for i, in_file in enumerate(in_files):
            return_dict[i] = user_parser(in_file)
            if ext_mem_workspace is not None:
                return_dict[i] = _to_ext_memory(f"file-{i}", return_dict[i], ext_mem_workspace)
        return return_dict

def worker_fn_no_return(worker_id, task_queue, func):
    """ Process tasks in the task_queue with multiprocessing
        without returning any value.

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

def multiprocessing_exec_no_return(tasks, num_proc, exec_func):
    """ Do multi-processing execution without
        returning any value.

        Each worker process will call exec_func
        independently.

        Parameters
        ----------
        task: list
            List of remap tasks.
        num_proc: int
            Number of workers to spin up.
        exec_func: func
            function to execute.
    """
    if num_proc > 1 and len(tasks) > 1:
        num_proc = min(len(tasks), num_proc)
        processes = []
        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        for i, task in enumerate(tasks):
            task_queue.put((i, task))

        for i in range(num_proc):
            proc = Process(target=worker_fn_no_return, args=(i, task_queue, exec_func))
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()
    else:
        for i, task_args in enumerate(tasks):
            logging.debug("worker 0 Processing %s task", i)
            exec_func(**task_args)


def _get_tot_shape(arrs):
    """ Get the shape after merging the arrays.

    Parameters
    ----------
    arrs : list of arrays

    Returns
    -------
    tuple : the shape of the merged array.
    """
    num_rows = 0
    shape1 = arrs[0].shape[1:]
    for arr in arrs:
        num_rows += arr.shape[0]
        assert shape1 == arr.shape[1:]
    shape = [num_rows] + list(shape1)
    return tuple(shape)

def _get_arrs_out_dtype(arrs):
    """ To get the output dtype by accessing the
        first element of the arrays (numpy array or HDFArray)

        Note: We use arrs[0][0] instead of arrs[0] because
            arrs[0][0] is a transformed data with out_dtype
            while arrs[0] can be a HDFArray and has not
            been cast to out_dtype.

    Parameters
    ----------
    arrs : list of arrays.
        The input arrays.
    """
    return arrs[0][0].dtype

class ExtMemArrayWrapper:
    """ An array wrapper for external-memory array.
    """

    def to_numpy(self):
        """ Convert the array to Numpy array.
        """

    def to_tensor(self):
        """ Return Pytorch tensor.
        """

    def astype(self, dtype):
        """ Set the output dtype.

        Parameters
        ----------
        dtype: numpy.dtype
            Output dtype
        """

    def cleanup(self):
        """ Clean up the external-memory array.
        """

    @property
    def shape(self):
        """ The shape of the array.
        """

    @property
    def dtype(self):
        """ The data type of the array.
        """

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


class HDF5Array(ExtMemArrayWrapper):
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
        idx : Numpy array or Pytorch tensor or slice or int.
            The index.

        Returns
        -------
        Numpy array : the data from the HDF5 array indexed by `idx`.
        """
        if isinstance(idx, (slice, int)):
            return self._arr[idx].astype(self._out_dtype)

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
        arr = copy.copy(self)
        arr._out_dtype = dtype
        return arr

    @property
    def shape(self):
        """ The shape of the HDF5 array.
        """
        return self._arr.shape

    @property
    def dtype(self):
        """ The data type of the HDF5 array.
        """
        if self._out_dtype is not None:
            return self._out_dtype
        else:
            return self._arr.dtype

class ExtNumpyWrapper(ExtMemArrayWrapper):
    """ The wrapper to memory-mapped Numpy array.

    Parameters
    ----------
    arr_path : str
        The path of memory-mapped numpy file.
    shape : tuple
        The shape of the array.
    dtype : numpy dtype
        The data type.
    """
    def __init__(self, arr_path, shape, dtype):
        self._arr_path = arr_path
        self._shape = shape
        self._orig_dtype = self._dtype = dtype
        self._arr = None

    @property
    def dtype(self):
        """ The data type
        """
        return self._dtype

    @property
    def shape(self):
        """ The shape of the array.
        """
        return self._shape

    def astype(self, dtype):
        """ Return an array with converted data type.
        """
        arr = copy.copy(self)
        arr._dtype = dtype
        return arr

    def __len__(self):
        return self._shape[0]

    def __getitem__(self, idx):
        if self._arr is None:
            self._arr = np.memmap(self._arr_path, self._orig_dtype, mode="r", shape=self._shape)
        return self._arr[idx].astype(self.dtype)

    def cleanup(self):
        """ Clean up the array.
        """
        if self._arr is not None:
            self._arr.flush()
        self._arr = None

    def to_numpy(self):
        """ Convert the data to Numpy array.
        """
        if self._arr is None:
            arr = np.memmap(self._arr_path, self._orig_dtype, mode="r", shape=self._shape)
            if self._dtype != self._orig_dtype:
                arr = arr.astype(self._dtype)
            return arr
        else:
            return self._arr.astype(self._dtype)

    def to_tensor(self):
        """ Return Pytorch tensor.
        """
        return th.tensor(self.to_numpy())

class ExtFeatureWrapper(ExtNumpyWrapper):
    """ The wrapper to memory-mapped Numpy array when combining features

    Parameters
    ----------
    arr_path : str
        A path to the directory of different feature files.
    shape : tuple
        The shape of the array.
    dtype : numpy dtype
        The data type.
    merged_file: str
        The merged file name
    """
    def __init__(self, arr_path, shape=None, dtype=None, merged_file="merged_feature.npy"):
        super().__init__(arr_path, shape, dtype)
        self._directory_path = arr_path
        self._merged_file = merged_file
        self._wrapper = []
        self._arr_path = os.path.join(self._directory_path, self._merged_file)

    def __getitem__(self, idx):
        if not self._shape:
            raise RuntimeError("Call ExtFeatureWrapper.merge() first before calling __getitem__")
        if self._arr is None:
            self._arr = np.memmap(self._arr_path, self._dtype, mode="r",
                                  shape=self._shape)
        return self._arr[idx].astype(self.dtype)

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """ Clean up the array.
        """
        # Expected file structure:
        # merged_file file_feature1 file_feature2
        # rmtree will clean up all single feature files as ExtNumpyWrapper does not clean them up
        if self._arr is not None:
            self._arr.flush()
        self._arr = None
        shutil.rmtree(self._directory_path)

    def to_numpy(self):
        """ Convert the data to Numpy array.
        """
        if not self._shape:
            raise RuntimeError("Call ExtFeatureWrapper.merge() first before calling to_numpy()")
        if self._arr is None:
            arr = np.memmap(self._arr_path, self._orig_dtype, mode="r", shape=self._shape)
            if self._dtype != self._orig_dtype:
                arr = arr.astype(self._dtype)
            return arr
        else:
            return self._arr.astype(self._dtype)

    def append(self, feature):
        """Add an external memory wrapper or numpy array,
        it will convert the numpy array to a memory wrapper
        Parameters
        ----------
        feature : numpy.ndarray or ExtMemArrayWrapper
            The value needs to be packed.
        """
        if not self._shape:
            self._shape = (feature.shape[0], 0)
        if self._shape and self._shape[0] != feature.shape[0]:
            raise RuntimeError(f"Expect that ExtFeatureWrapper has a "
                               f"first dimension that is the same but get "
                               f"{self.shape[0]} and {feature.shape[0]}")
        # Convert the numpy array into an ExtNumpyWrapper
        # By converting it into an ExtNumpyWrapper, it will avoid loading
        # all the numpy arrays into the memory at the same time.
        if isinstance(feature, np.ndarray):
            hash_hex = generate_hash()
            path = self._directory_path + '/{}.npy'.format(hash_hex)
            feature = convert_to_ext_mem_numpy(path, feature)
        self._wrapper.append(feature)

    def merge(self):
        """ Merge feature col-wised.
        """
        assert self._wrapper, "Cannot merge an empty list, " \
                             "need to append external memory wrapper first"
        feat_dim = sum(wrap.shape[1] for wrap in self._wrapper)
        self._shape = (self._shape[0], feat_dim)
        self._orig_dtype = self._dtype = self._wrapper[0].dtype

        out_arr = np.memmap(self._arr_path, self._orig_dtype,
                            mode="w+", shape=self._shape)
        col_start = 0
        for wrap in self._wrapper:
            col_end = col_start + wrap.shape[1]
            # load the numpy array from the disk one by one
            out_arr[:, col_start:col_end] = wrap.to_numpy()
            col_start = col_end
        out_arr.flush()
        return self

def _merge_arrs(arrs, tensor_path):
    """ Merge the arrays.

    The merged array may be stored in a file specified by the path.

    Parameters
    ----------
    arrs : list of arrays.
        The input arrays.
    tensor_path : str
        The path where the Numpy array is stored.

    Returns
    -------
    Numpy array : the merged array.
    """
    assert isinstance(arrs, list)
    shape = _get_tot_shape(arrs)

    # To get the output dtype or arrs
    dtype = _get_arrs_out_dtype(arrs)
    if tensor_path is not None:
        out_arr = np.memmap(tensor_path, dtype, mode="w+", shape=shape)
        row_idx = 0
        for arr in arrs:
            out_arr[row_idx:(row_idx + arr.shape[0])] = arr[:]
            row_idx += arr.shape[0]
        out_arr.flush()
        return ExtNumpyWrapper(tensor_path, out_arr.shape, out_arr.dtype)
    elif isinstance(arrs[0], ExtMemArrayWrapper):
        arrs = [arr.to_numpy() for arr in arrs]
        return np.concatenate(arrs)
    else:
        return np.concatenate(arrs)

class ExtMemArrayMerger:
    """ Merge multiple Numpy arrays.

    The merged array may be stored on disks.

    Parameters
    ----------
    ext_mem_workspace : str
        The path of the directory where the array will be stored.
    ext_mem_feat_size : int
        The threshold of the feature size that triggers storing data on disks.
    """
    def __init__(self, ext_mem_workspace, ext_mem_feat_size):
        self._ext_mem_workspace = ext_mem_workspace
        self._ext_mem_feat_size = ext_mem_feat_size
        self._tensor_files = []

    def __del__(self):
        for tensor_file in self._tensor_files:
            os.remove(tensor_file)

    def __call__(self, arrs, name):
        """ Merge multiple Numpy array.

        Parameters
        ----------
        arrs : list of arrays.
            The input arrays.
        name : str
            The name of the external memory array.

        Returns
        -------
        Numpy array : an array stored in external memory.
        """
        assert isinstance(arrs, list)
        shape = _get_tot_shape(arrs)
        # If external memory workspace is not initialized or the feature size is smaller
        # than a threshold, we don't do anything.
        if self._ext_mem_workspace is None or np.prod(shape[1:]) < self._ext_mem_feat_size:
            if len(arrs) == 1 and isinstance(arrs[0], ExtMemArrayWrapper):
                return arrs[0].to_numpy()
            elif len(arrs) == 1:
                return arrs[0]
            else:
                return _merge_arrs(arrs, None)

        # We need to create the workspace directory if it doesn't exist.
        os.makedirs(self._ext_mem_workspace, exist_ok=True)
        tensor_path = os.path.join(self._ext_mem_workspace, name + ".npy")
        self._tensor_files.append(tensor_path)
        if len(arrs) > 1:
            return _merge_arrs(arrs, tensor_path)
        else:
            return convert_to_ext_mem_numpy(tensor_path, arrs[0])

def convert_to_ext_mem_numpy(tensor_path, arr):
    """ Convert a numpy array to memory mapped array.

    Parameters
    ----------
    tensor_path : str
        The path of the file to store the Numpy array.
    arr : Numpy array
        The Numpy array

    Returns
    -------
    ExtNumpyWrapper : the wrapper of the memory mapped array.
    """
    os.makedirs(os.path.dirname(tensor_path), exist_ok=True)
    em_arr = np.memmap(tensor_path, arr.dtype, mode="w+", shape=arr.shape)
    em_arr[:] = arr[:]
    em_arr.flush()
    return ExtNumpyWrapper(tensor_path, em_arr.shape, em_arr.dtype)

def save_maps(output_dir, fname, map_data):
    """ Save node id mapping or edge id mapping

    Parameters
    ----------
    output_dir : str
        The directory where we will save the partitioned results.
    fname: str
        Mapping file name
    map_data: dict of tensors
        ID mapping
    """
    map_file = f"{fname}.pt"
    map_file = os.path.join(output_dir, map_file)
    # Use torch save as tensors are torch tensors
    th.save(map_data, map_file)

def load_maps(output_dir, fname):
    """ Load saved node id mapping or edge id mapping

    Parameters
    ----------
    output_dir : str
        The directory where we will save the partitioned results.
    fname: str
        Mapping file name

    Return
    ------
    dict of tensors
        ID mappings
    """
    map_file = f"{fname}.pt"
    map_file = os.path.join(output_dir, map_file)

    return th.load(map_file)

def partition_graph(g, node_data, edge_data, graph_name, num_partitions, output_dir,
                    part_method=None, save_mapping=True):
    """ Partition a graph

    This takes advantage of the graph partition function in DGL.
    To save memory consumption for graph partition. We only pass the graph object
    with the graph structure to DGL's graph partition function.
    We will split the node/edge feature tensors based on the graph partition results.
    By doing so, we can keep the node/edge features in external memory to further
    save memory.

    Parameters
    ----------
    g : DGLGraph
        The full graph object.
    node_data : dict of tensors
        The node feature tensors.
    edge_data : dict of tensors
        The edge feature tensors.
    graph_name : str
        The graph name.
    num_partitions : int
        The number of partitions.
    output_dir : str
        The directory where we will save the partitioned results.
    part_method : str (optional)
        The partition algorithm used to partition the graph.
    save_mapping: bool
        Whether to store the mappings for the edges and nodes after partition.
        Default: True
    """
    from dgl.distributed.graph_partition_book import _etype_tuple_to_str
    orig_id_name = "__gs_orig_id"
    for ntype in g.ntypes:
        g.nodes[ntype].data[orig_id_name] = th.arange(g.number_of_nodes(ntype))
    for etype in g.canonical_etypes:
        g.edges[etype].data[orig_id_name] = th.arange(g.number_of_edges(etype))
    sys_tracker.check('Before partitioning starts')
    if part_method is None:
        part_method = "None" if num_partitions == 1 else "metis"

    balance_ntypes = {}
    # Only handle the single task case with the default mask names as
    # train_mask, val_mask and test_mask.
    # TODO: Support balance training set for multi-task learning.
    for ntype in node_data:
        balance_arr = th.zeros(g.number_of_nodes(ntype), dtype=th.int8)
        balance_tag = 1
        num_train_eval = num_train = num_valid = num_test = 0
        if "train_mask" in node_data[ntype]:
            num_train = np.sum(node_data[ntype]["train_mask"])
            num_train_eval += num_train
            # If there are no training nodes or if all nodes are training nodes,
            # we don't need to do anything.
            if 0 < num_train < g.number_of_nodes(ntype):
                balance_arr += node_data[ntype]["train_mask"] * balance_tag
                balance_tag += 1
            logging.debug("Balance %d training nodes on node %s.", num_train, ntype)
        if "val_mask" in node_data[ntype] and num_train_eval < g.number_of_nodes(ntype):
            num_valid = np.sum(node_data[ntype]["val_mask"])
            num_train_eval += num_valid
            if 0 < num_valid < g.number_of_nodes(ntype):
                balance_arr += node_data[ntype]["val_mask"] * balance_tag
                balance_tag += 1
            logging.debug("Balance %d validation nodes on node %s.", num_valid, ntype)
        if "test_mask" in node_data[ntype] and num_train_eval < g.number_of_nodes(ntype):
            num_test = np.sum(node_data[ntype]["test_mask"])
            num_train_eval += num_test
            if 0 < num_test < g.number_of_nodes(ntype):
                balance_arr += node_data[ntype]["test_mask"] * balance_tag
            logging.debug("Balance %d test nodes on node %s.", num_test, ntype)
        assert num_train_eval <= g.number_of_nodes(ntype), \
                f"There are {g.number_of_nodes(ntype)} nodes, ' \
                + 'we get {num_train_eval} nodes for train/valid/test."
        # If all nodes are in training/valid/test sets and none of the sets contain
        # all nodes, we assign 1, 2, 3, etc to training/validation/test nodes.
        # However, DistDGL requires the values start from 0. We need to subtract
        # the values by 1.
        if num_train_eval == g.number_of_nodes(ntype) and \
                (num_train != g.number_of_nodes(ntype) and \
                 num_valid != g.number_of_nodes(ntype) and \
                 num_test != g.number_of_nodes(ntype)):
            balance_arr -= 1
        balance_ntypes[ntype] = balance_arr
    mapping = \
        dgl.distributed.partition_graph(g, graph_name, num_partitions, output_dir,
                                        part_method=part_method,
                                        balance_ntypes=balance_ntypes,
                                        balance_edges=True,
                                        return_mapping=save_mapping)
    sys_tracker.check('Graph partitioning')

    # If num_partitions is 1, node IDs are not shuffled.
    # There is no need to reorder the features.
    if num_partitions == 1:
        part_dir = os.path.join(output_dir, "part0")
        # Get the node features for the partition and save the node features in node_feat.dgl.
        nfeat_data = {}
        for ntype in node_data:
            for name, ndata in node_data[ntype].items():
                nfeat_data[ntype + "/" + name] = th.tensor(ndata)
            sys_tracker.check(f'Get node data of node {ntype} in partition 0')

        dgl.data.utils.save_tensors(os.path.join(part_dir, "node_feat.dgl"), nfeat_data)

        # Get the edge features for the partition and save the edge features in edge_feat.dgl.
        efeat_data = {}
        for etype in edge_data:
            for name, edata in edge_data[etype].items():
                efeat_data[_etype_tuple_to_str(etype) + "/" + name] = th.tensor(edata)
            sys_tracker.check(f'Get edge data of edge {etype} in partition 0')
        dgl.data.utils.save_tensors(os.path.join(part_dir, "edge_feat.dgl"), efeat_data)
    else:
        for i in range(num_partitions):
            part_dir = os.path.join(output_dir, "part" + str(i))
            data = dgl.data.utils.load_tensors(os.path.join(part_dir, "node_feat.dgl"))
            # Get the node features for the partition and save the node features in node_feat.dgl.
            for ntype in node_data:
                # We store the original node IDs as a node feature when we partition the graph.
                # We can get the original node IDs from the node features and now
                # we use them to retrieve the right node features.
                orig_ids = data[ntype + "/" + orig_id_name]
                for name, ndata in node_data[ntype].items():
                    data[ntype + "/" + name] = th.tensor(ndata[orig_ids])
                sys_tracker.check(f'Get node data of node {ntype} in partition {i}')
            # Delete the original node IDs from the node data.
            for ntype in g.ntypes:
                del data[ntype + "/" + orig_id_name]
            dgl.data.utils.save_tensors(os.path.join(part_dir, "node_feat.dgl"), data)

            data = dgl.data.utils.load_tensors(os.path.join(part_dir, "edge_feat.dgl"))
            # Get the edge features for the partition and save the edge features in edge_feat.dgl.
            for etype in edge_data:
                # We store the original edge IDs as a edge feature when we partition the graph.
                # We can get the original edge IDs from the edge features and now
                # we use them to retrieve the right edge features.
                orig_ids = data[_etype_tuple_to_str(etype) + '/' + orig_id_name]
                for name, edata in edge_data[etype].items():
                    data[_etype_tuple_to_str(etype) + "/" + name] = th.tensor(edata[orig_ids])
                sys_tracker.check(f'Get edge data of edge {etype} in partition {i}')
            for etype in g.canonical_etypes:
                del data[_etype_tuple_to_str(etype) + '/' + orig_id_name]
            dgl.data.utils.save_tensors(os.path.join(part_dir, "edge_feat.dgl"), data)

    if save_mapping:
        new_node_mapping, new_edge_mapping = mapping

        # the new_node_mapping contains per entity type on the ith row
        # the original node id for the ith node.
        save_maps(output_dir, "node_mapping", new_node_mapping)
        # the new_edge_mapping contains per edge type on the ith row
        # the original edge id for the ith edge.
        save_maps(output_dir, "edge_mapping", new_edge_mapping)

def get_hard_edge_negs_feats(hard_edge_neg_ops):
    """ Get feature names of hard negatives for each edge type.

        Parameters
        ----------
        hard_edge_neg_ops: HardEdgeNegativeTransform
            A list of edge hard negative transformations.
    """
    hard_edge_neg_feats = {}
    for hard_edge_neg_op in hard_edge_neg_ops:
        edge_type = hard_edge_neg_op.target_etype
        neg_ntype = hard_edge_neg_op.neg_ntype

        if edge_type not in hard_edge_neg_feats:
            hard_edge_neg_feats[edge_type] = {neg_ntype: [hard_edge_neg_op.feat_name]}
        else:
            if neg_ntype in hard_edge_neg_feats[edge_type]:
                hard_edge_neg_feats[edge_type][neg_ntype].append(hard_edge_neg_op.feat_name)
            else:
                hard_edge_neg_feats[edge_type][neg_ntype] = [hard_edge_neg_op.feat_name]

    return hard_edge_neg_feats

def shuffle_hard_nids(data_path, num_parts, hard_edge_neg_feats):
    """ Shuffle node ids of hard negatives from Graph node id space to
        Partition Node id space.

        Parameters
        ----------
        data_path: str
            Path to the directory storing the partitioned graph.
        num_parts: int
            Number of partitions.
        hard_edge_neg_feats: dict of lists
            A directory storing hard negative features for each edge type.
    """
    # Load node id remapping
    # The node mapping stores the mapping from Partition Node IDs to Graph Node IDs
    node_mapping = load_maps(data_path, "node_mapping")
    gnid2pnid_mapping = {}

    def get_gnid2pnid_map(ntype):
        if ntype in gnid2pnid_mapping:
            return gnid2pnid_mapping[ntype]
        else:
            pnid2gnid_map = node_mapping[ntype]
            gnid2pnid_map = th.argsort(pnid2gnid_map)
            gnid2pnid_mapping[ntype] = gnid2pnid_map
            # del ntype in node_mapping to save memory
            del node_mapping[ntype]
            return gnid2pnid_mapping[ntype]

    # iterate all the partitions to convert hard negative node ids.
    for i in range(num_parts):
        part_path = os.path.join(data_path, f"part{i}")
        edge_feat_path = os.path.join(part_path, "edge_feat.dgl")

        # load edge features first
        edge_feats = dgl.data.utils.load_tensors(edge_feat_path)

        for etype, hard_neg_feats in hard_edge_neg_feats.items():
            etype = ":".join(etype)
            for neg_ntype, neg_feats in hard_neg_feats.items():
                for neg_feat in neg_feats:
                    efeat_name = f"{etype}/{neg_feat}"
                    hard_nids = edge_feats[efeat_name]
                    hard_nid_idx = hard_nids > -1
                    gnid2pnid_map = get_gnid2pnid_map(neg_ntype)
                    hard_nids[hard_nid_idx] = gnid2pnid_map[hard_nids[hard_nid_idx]]

        # replace the edge_feat.dgl with the updated one.
        os.remove(edge_feat_path)
        dgl.data.utils.save_tensors(edge_feat_path, edge_feats)
