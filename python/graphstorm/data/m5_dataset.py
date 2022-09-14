# The general dataset for M5GNN

import os, sys
import json
import numpy as np
import h5py
import math
import warnings
import time
import tqdm
import dgl
import torch as th
import psutil
import pyarrow as pa
from functools import partial
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers import AutoTokenizer
from .utils import save_raw_text, save_maps
from .dataset import M5gnnDataset
from .utils import get_id, reverse_etype, add_reverse_edges
from .utils import alltoallv_cpu, alltoall_cpu
from .utils import all_reduce_sum
from .constants import EDGE_SRC_IDX, EDGE_DST_IDX, NODE_ID_IDX
from .constants import TOKEN_IDX, VALID_LEN_IDX, ATT_MASK_IDX, TOKEN_TID_IDX, \
        TRAIN_IDX, VALID_IDX, TEST_IDX

from multiprocessing import Manager
try:
    from m5_tokenizers.tokenizers import SentencepieceTokenizer
    from m5_dataloaders.datasets import JSONPredictionDatasetStream
    from m5_dataloaders.datasets.constants import CLASSIFICATION_TASK, REGRESSION_TASK
    from m5_dataloaders import DataProvider, WorkerInitObj
    from m5_dataloaders.shm import serialize

    has_m5 = True
except Exception as e:
    print(e)
    REGRESSION_TASK = "regression"
    CLASSIFICATION_TASK = "classification"
    print("M5 is not available")
    has_m5 = False

def _get_mask(json_data):
    ''' Get the train/valid/test mask from the JSON file

    Arguments:
        json_data : dict that contain data for a node or an edge.

    Returns:
        a tuple of 3 integers that indicate train/valid/test.
    '''
    train = 1 if 'train' in json_data and int(json_data['train']) != 0 else 0
    valid = 1 if 'valid' in json_data and int(json_data['valid']) != 0 else 0
    test = 1 if 'test' in json_data and int(json_data['test']) != 0 else 0
    return train, valid, test

def _read_nid_map(nid_file, manager):
    ''' Read the node ID file and construct a NID map.

    The node ID file contains node IDs for a node type. Each line stores a node ID,
    which can be any string. We use the line number as the numeric ID of the name in the DGL graph.
    The function returns a dict that maps the string node ID to the numeric ID.

    Arguments:
        nid_file : the node ID file path.
        manager: multiprocessing manager. We create a shared memory dict object using this manager.

    Returns:
        a dict that maps the string node ID to the numeric ID.
    '''
    # Create a shared memory dict object
    shd_nid_map = manager.dict()
    with open(nid_file, 'r', encoding='utf-8') as reader:
        tmp_map = {}
        for num, line in enumerate(tqdm.tqdm(reader)):
            line = line.strip("\n")

            # we should consolidate the format of id file to be json string
            line = json.loads(line)
            assert line != "", "Get an empty node ID"
            tmp_map[line] = num

            # To avoid the package size limitation in multiprocessing package,
            # we need to split the whole nid_map into multiple pieces
            # and then update the shared nid_map.
            if num % 100000 == 0:
                shd_nid_map.update(tmp_map)
                tmp_map = {}

        shd_nid_map.update(tmp_map)
    return shd_nid_map

class JSONEdgeDataset(Dataset):
    ''' Load edge data from JSON files.

    We support to load edge data from JSON files. Each row in a JSON file represents an edge
    in the graph. Users need to specify the fields where the edge data is stored in a row.
    Below shows an example of an edge:

    ```
    {'src_id': 'B00MRZIFD0', 'dst_id': '1', 'locale': 'US-EN', ...}
    ```

    The dataset supports loading data from
    multiple JSON files. If the edge data contain text features, the text will be tokenized.
    We use the M5 dataset to tokenize text. (currently, this is not supported)

    Arguments
    ---------
        input_files : list of str.
            The paths of the files where the edge data are stored.
        tokenizer : object
            The tokenizer used to tokenize text.
        src_field : str
            The field name where source node IDs are stored. A node ID can be a string
            or an integer. If the node ID is a string, `nid_map` needs to be provided to
            map a string to an integer.
        dst_field : str
            The field name where destination node IDs are stored. A node ID can be a string
            or an integer. If the node ID is a string, `nid_map` needs to be provided to
            map a string to an integer.
        nid_maps : dict (str, int)
            This maps string IDs of source nodes and destination nodes to numeric node IDs.
            If this is not provided, it assumes that
            the node IDs stored in the input files are numbers. Otherwise, it will report errors.
        text_fields : list of str
            The field names where text data are stored.
        label_field : str, optional
            The filed name where the label of an edge is stored.
        task_type : int
            This indicates what types of tasks we run on the edge data.
        max_seq_length : int, optional
            The maximal sequence length of the text used for tokenization.
        has_data_split : bool
            This indicates whether to get data split from the input files. If this is enabled,
            it will check the three special fields 'train', 'valid' and 'test' in the input files
            to see how to split the data.
    '''
    def __init__(self, input_files, tokenizer,
                 src_field, dst_field, nid_maps,
                 text_fields=[], feat_fields=[],label_field=None,
                 task_type=CLASSIFICATION_TASK,
                 max_seq_length=512, has_data_split=False):
        self.tokenizer = tokenizer
        self.input_files = input_files
        self.src_field = src_field
        self.dst_field = dst_field
        self.text_fields = text_fields
        self.feat_fields = feat_fields
        self.label_field = label_field
        self.task_type = task_type
        self.max_seq_length = max_seq_length
        self.has_data_split = has_data_split
        self.nid_maps = nid_maps
        assert self.nid_maps is not None, "nid_maps should not be None"
        assert len(self.nid_maps) == 2, \
            "nid_maps should include a map for src nodes and a map for dst nodes"

        self.inputs = self.get_input_arrays()

    def __len__(self):
        return len(self.inputs[EDGE_SRC_IDX])

    def __getitem__(self, index):
        return {key: self.inputs[key][index] for key in self.inputs}

    def get_input_arrays(self):
        ''' Parse input files and save the results in multiple arrays

        Currently, it cannot cannot handle text data on the edges and tokenize them.

        Returns
        -------
            a dict of arrays
               The keys are integers and the values are the arrays that store the parsed results.
               It always contains an array for source node IDs and an array for destination node IDs.
               It may contain an array for labels, an array for train mask, an array for validation mask
               or an array for test mask.
        '''
        src_ids = []
        dst_ids = []
        labels = []
        train_mask = []
        valid_mask = []
        test_mask = []
        feats = [[] for _ in range(len(self.feat_fields))]

        for i, input_file in enumerate(self.input_files):
            print ("Loading edge file {}: {} of {}".format(input_file, i, len(self.input_files)))
            with open(input_file, 'r', encoding='utf-8') as reader:
                for line in reader:
                    line = line.strip()
                    if not line:
                        continue
                    json_data = json.loads(line)
                    if self.src_field not in json_data or self.dst_field not in json_data:
                        print('Warning! a line with incomplete data, ignore')
                        continue
                    if (isinstance(json_data[self.src_field], str) and len(json_data[self.src_field]) == 0) or \
                        (isinstance(json_data[self.dst_field], str) and len(json_data[self.dst_field]) == 0):
                        # If nid is string, it should not be an empty string
                        print (('Warning! a line with empty data, ignore'))
                        continue
                    if self.label_field is not None and self.label_field not in json_data:
                        print('Warning! the edge data does not contain label')
                        continue
                    if len(self.text_fields) > 0 \
                            and not all(data_f in json_data for data_f in self.text_fields):
                        print('Warning! the edge data does not contain all data fileds')
                        continue
                    if len(self.feat_fields) > 0 \
                            and not all(data_f in json_data for data_f in self.feat_fields):
                        print('Warning! the edge data does not contain all data fileds')
                        continue
                    if self.nid_maps[0] is not None and self.nid_maps[1] is not None:
                        try:
                            # src node or dst node may not exist in node data, drop the edge.
                            src_id = self.nid_maps[0][json_data[self.src_field]]
                            dst_id = self.nid_maps[1][json_data[self.dst_field]]
                        except:
                            # if error happends here, drop the edge
                            continue
                        src_ids.append(src_id)
                        dst_ids.append(dst_id)
                    else:
                        src_ids.append(int(json_data[self.src_field]))
                        dst_ids.append(int(json_data[self.dst_field]))
                    if len(self.feat_fields) > 0:
                        for feat_i, feat_name in enumerate(self.feat_fields):
                            feats[feat_i].append(json_data[feat_name])
                    if self.label_field is not None:
                        if self.task_type == CLASSIFICATION_TASK:
                            label = json_data[self.label_field]
                            if isinstance(label, list):
                                # For multilabel, labels are stored as a list of 0s and 1s.
                                labels.append([int(l) for l in label])
                            else:
                                # For single label, labels are stored as integers
                                labels.append(int(json_data[self.label_field]))
                        elif self.task_type == REGRESSION_TASK:
                            labels.append(float(json_data[self.label_field]))
                        else:
                            raise ValueError(f"Unknown label type found f{label_data} type:{type(label_data)}")
                    if self.has_data_split:
                        train, valid, test = _get_mask(json_data)
                        train_mask.append(train)
                        valid_mask.append(valid)
                        test_mask.append(test)
                    # TODO(zhengda) we need to get tokens in the edges if there are any.

        src_ids = th.tensor(src_ids)
        dst_ids = th.tensor(dst_ids)
        ret_dict = {
            EDGE_SRC_IDX: src_ids,
            EDGE_DST_IDX: dst_ids,
        }
        if len(labels) > 0:
            labels = th.tensor(labels)
            ret_dict[self.label_field] = labels
            assert len(labels) == len(src_ids)
        if self.has_data_split:
            train_mask = th.tensor(train_mask, dtype=th.bool)
            valid_mask = th.tensor(valid_mask, dtype=th.bool)
            test_mask = th.tensor(test_mask, dtype=th.bool)
            assert len(train_mask) == len(src_ids)
            ret_dict[TRAIN_IDX] = train_mask
            ret_dict[VALID_IDX] = valid_mask
            ret_dict[TEST_IDX] = test_mask
        if len(self.feat_fields) > 0:
            for feat_i, feat_name in enumerate(self.feat_fields):
                feat = th.tensor(feats[feat_i])
                assert len(feat) == len(src_ids)
                ret_dict[feat_name] = feat
        return ret_dict

def _pad_arrs_to_max_length(arrs, pad_axis, pad_val, round_to=None, max_length=None):
    """Inner Implementation of the Pad collate

    Note: this method is replicated to M5Models for dependency decoupling. Please make sure to sync any related changes
        to https://tiny.amazon.com/14outu2a8

    Arguments:
        arrs (list)
        pad_axis (int)
        pad_val (number)
        round_to (int, optional). (default: ``None``)
        max_length (int, optional). (default: ``None``)

    Returns:
        ret : torch.Tensor
        original_length : torch.Tensor
    """
    if not isinstance(arrs[0], th.Tensor):
        arrs = [th.as_tensor(ele) for ele in arrs]

    original_length = [ele.size(pad_axis) for ele in arrs]
    max_arr_len = max(original_length)

    if round_to is not None:
        max_arr_len = round_to * math.ceil(max_arr_len / round_to)
    elif max_length is not None:
        if max_length < max_arr_len:
            raise ValueError(
                "If max_length is specified, max_length={} must be larger "
                "than the maximum length {} of the given arrays at axis={}.".format(
                    max_length, max_arr_len, pad_axis
                )
            )
        max_arr_len = max_length

    size = arrs[0].size()
    prev_trailing_dims = size[:pad_axis]
    after_trailing_dims = size[pad_axis + 1 :]

    out_dims = (len(arrs),) + prev_trailing_dims + (max_arr_len,) + after_trailing_dims
    out_tensor = th.as_tensor(np.full(out_dims, pad_val, dtype=np.int64))
    for i, tensor in enumerate(arrs):
        length = tensor.size(pad_axis)
        out_tensor[i].narrow(pad_axis, 0, length)[:] = tensor

    original_length = th.as_tensor(original_length)

    return out_tensor, original_length

class Pad:
    r"""Returns a callable that pads and stacks data.

    Note: this class (along with the helper method `_pad_arrs_to_max_length`) is replicated to M5Models
        for dependency decoupling. Please make sure to sync any related changes to
        https://tiny.amazon.com/14outu2a8

    Arguments:
        axis (int, optional): The axis to pad the arrays.
            The arrays will be padded to the largest dimension at :attr:`axis`.
            For example, assume the input arrays have shape (10, 8, 5), (6, 8, 5), (3, 8, 5)
            and the `axis` is 0.
            Each input will be padded into (10, 8, 5) and then stacked to form the final output,
            which has shape(3, 10, 8, 5). (default ``0``)
        pad_val (float or int, optional): The padding value. (default ``0``)
        round_to (int, optional):
            If specified, the padded dimension will be rounded to be multiple of this argument.
            Mutually exclusive with :attr:`max_length`. (default ``None``)
        max_length (int, optional):
            If specified, the padded dimension will have length :attr:`max_length`,
            and it must be larger than the maximum length in the arrays at :attr:`axis`.
            Mutually exclusive with :attr:`round_to`.  (default ``None``)
        ret_length (bool, optional): Whether to return the valid length in the output.
            (default ``False``)
    """
    def __init__(self, axis=0, pad_val=None, round_to=None, max_length=None, ret_length=False):
        self._axis = axis
        if not isinstance(axis, int):
            raise ValueError(
                "axis must be an integer! Received axis={}, type={}.".format(
                    str(axis), str(type(axis))
                )
            )

        if round_to is not None and max_length is not None:
            raise ValueError(
                "Only either round_to={} or max_length={} can be specified.".format(
                    round_to, max_length
                )
            )

        self._pad_val = 0 if pad_val is None else pad_val
        self._round_to = round_to
        self._max_length = max_length
        self._ret_length = ret_length

        if pad_val is None:
            warnings.warn(
                "Padding value is not given and will be set automatically to 0 "
                "in data.Pad(). "
                "Please check whether this is intended "
                "(e.g. value of padding index in the tokenizer)."
            )

    def __call__(self, data):
        """Collate the input data.

        The arrays will be padded to the largest dimension at `axis` and then
        stacked to form the final output. In addition, the function will output
        the original dimensions at the `axis` if ret_length is turned on.

        Arguments:
            data : List[np.ndarray] or List[List[dtype]] or List[torch.Tensor]
                List of samples to pad and stack.

        Returns:
            batch_data (torch.Tensor): Data in the minibatch. Shape is (N, ...)
            valid_length (NDArray, optional):
                The sequences' original lengths at the padded axis. Shape is (N,). This will only be
                returned if `ret_length` is True.

        """
        if isinstance(data[0], (th.Tensor, np.ndarray, list, tuple)):
            padded_arr, original_length = _pad_arrs_to_max_length(
                data,
                pad_axis=self._axis,
                pad_val=self._pad_val,
                round_to=self._round_to,
                max_length=self._max_length,
            )
            if self._ret_length:
                return padded_arr, original_length
            else:
                return padded_arr
        else:
            raise NotImplementedError

class JSONNodeDataset(JSONPredictionDatasetStream):
    ''' Load node data from JSON files.

    We support to load node data from JSON files. Each row stores the data of a node.
    Users need to specify the fields where the node data is stored in a row.
    Below shows an example of a row that stores node data:

    ```
    {'id': 'B00MRZIFD0', 'item_name': 'Trainer Cup 7 Ounce 47022', 'locale': 'US-EN', 'train': 1, ...}
    ```

    The dataset supports loading data from
    multiple JSON files. If the node data contain text features, the text will be tokenized.
    We use the M5 dataset to tokenize text.

    Arguments
    ---------
        input_files : list of str.
            The paths of the files where the node data are stored.
        tokenizer : object
            The tokenizer used to tokenize text.
        use_hf_tokenizer : bool
            A flag indicates whether to use HuggingFace tokenizer.
        id_field : str
            The field name where node IDs are stored.
        text_fields : list of str
            The field names where text data are stored.
        feat_fields : list of str
            The field names where text data are stored.
        label_field : str, optional
            The filed name where the label of a node is stored.
        max_seq_length : int, optional
            The maximal sequence length of the text used for tokenization.
        has_data_split : bool
            This indicates whether to get data split from the input files. If this is enabled,
            we will check the three special fields 'train', 'valid' and 'test' in the input files
            to see how to split the data.
        nid_map : dict (str, int)
            This maps string node IDs to numeric node IDs. If this is not provided, we assume that
            the node IDs stored in the input files are numbers. Otherwise, it will report errors.
        task_type : int
            This indicates what types of tasks we run on the node data.
    '''
    def __init__(self, input_files, tokenizer, use_hf_tokenizer, id_field,
                 text_fields=[], feat_fields=[], label_field=None,
                 max_seq_length=512, has_data_split=True,
                 nid_map=None, task_type=CLASSIFICATION_TASK):
        self.input_files = input_files
        self.id_field = id_field
        self.has_data_split = has_data_split
        self.nid_map = nid_map
        self.text_fields = text_fields
        self.feat_fields = feat_fields
        self.use_hf_tokenizer = use_hf_tokenizer
        self.label_field = label_field

        super(JSONNodeDataset, self).__init__(tokenizer, id_field=id_field,
                                              label_field=label_field, fields=text_fields,
                                              max_seq_length=max_seq_length, task_type=task_type)
        # TODO(zhengda) The Pad function in M5 doesn't work. For some reason, Pytorch cannot allocate memory
        # in other processes. The customized Pad function allocate memory with NumPy. It works fine.
        self.pad = Pad(pad_val=tokenizer.pad_token_id, max_length=max_seq_length)
        self.inputs = self.get_input_arrays()

    def __len__(self):
        return len(self.inputs[NODE_ID_IDX])

    def __getitem__(self, index):
        return {key: self.inputs[key][index].share_memory_() for key in self.inputs}

    def _extract_label(self, label_data):
        """ Extract label data

            For node classification, label can be:
                1) An integer (signle label classification)
                2) A list of integer (multi label classification)

            Parameters
            ----------
            label_data:
                Label data.
        """
        label = None
        if self.task_type == CLASSIFICATION_TASK:
            if isinstance(label_data, list):
                label = [int(l) for l in label_data]
            else:
                label = int(label_data)
        elif self.task_type == REGRESSION_TASK:
            label = float(label_data)
        else:
            raise ValueError(f"Unknown label type found f{label_data} type:{type(label_data)}")
        return label

    def _get_instances_from_notext_inputs(self, inputs):
        """
        Get prediction instances without text data from the given inputs.

        Parameters
        ----------
        inputs: an iterable that yields lines of JSON strings.
            It can be an opened file handler containing the JSON lines, or a collection of strings.

        Returns
        -------
        A list of int. The ID for each row.
        A list of int or float. The label for each row.
        """
        labels = []
        ids = []
        for line in inputs:
            line = line.strip()
            if not line:
                continue

            json_data = json.loads(line)
            assert self.id_field is not None
            ids.append(json_data[self.id_field])
            if self.label_field:
                labels.append(self._extract_label(json_data[self.label_field]))
        if self.label_field:
            assert len(ids) == len(labels)
        return ids, labels

    def _get_instances_from_hf_inputs(self, inputs):
        """
        Get prediction instances with text tokenized by HuggingFace tokenizer.

        Parameters
        ----------
        inputs: an iterable that yields lines of JSON strings.
            It can be an opened file handler containing the JSON lines, or a collection of strings.

        Returns
        -------
        A list of Pytorch tensors. Each tensor stores token IDs.
        A list of Pytorch tensors. Each tensor is boolean mask that indicates the valid tokens
            in the token tensors.
        A list of Pytorch tensors. Each tensor stores the token types in the token tensors.
        A list of int or float. The label for each row.
        A list of int. The ID for each row.
        """
        labels = []
        ids = []
        input_ids = []
        valid_lens = []
        for i, line in enumerate(inputs):
            line = line.strip()
            if not line:
                continue

            json_data = json.loads(line)
            assert self.id_field is not None
            ids.append(json_data[self.id_field])
            if self.label_field:
                labels.append(self._extract_label(json_data[self.label_field]))

            text = '.'.join([field + ":" + json_data[field] for field in self.text_fields])
            tokens = self.tokenizer(text, max_length=self.max_seq_length, truncation=True,
                                    padding='max_length', return_tensors='pt')
            # The output from the HF tokenizer has the shape of (1, dim) while
            # the output from the M5 tokenizer has the shape of (dim,).
            # Let's remove the first dimension to unify the shape of the outputs from
            # the two tokenizers.
            input_ids.append(th.squeeze(tokens[TOKEN_IDX]))
            valid_lens.append(tokens[ATT_MASK_IDX].sum(dim=1))
            # TODO(zhengda) we may need to handle token type IDs in the future.

            if i % 100000  == 0 and \
                (th.distributed.is_initialized() is False or
                th.distributed.get_rank() == 0):
                # Either it is single machine data processing task (not distributed)
                # Or it is a distributed processing task and the current process is rank 0.
                print(f"Processed [{i}]")

        if self.label_field:
            assert len(ids) == len(labels)
        return input_ids, valid_lens, labels, ids

    def generate_inputs(self, inputs):
        if len(self.text_fields) > 0 and self.use_hf_tokenizer:
            input_tokens, valid_len, labels, inst_ids = self._get_instances_from_hf_inputs(inputs)
            return (input_tokens, valid_len), labels, inst_ids
        elif len(self.text_fields) > 0:
            input_tokens, valid_len, labels, inst_ids = self.generate_tokenized_inputs(inputs)
            return (input_tokens, valid_len), labels, inst_ids
        else:
            ids, labels = self._get_instances_from_notext_inputs(inputs)
            return None, labels, ids

    def get_input_arrays(self):
        ''' Parse input files and save the results in multiple arrays

        It relies on the M5 data loader to extract the node IDs, labels and text data from
        the input files and tokenize the text data.

        If `has_data_split` is enabled, it extracts train masks, validation masks and test masks.

        Returns
        -------
            a dict of arrays
               The keys are integers and the values are the arrays that store the parsed results.
               It always contains an array for node IDs. It may contain an array for labels,
               an array for tokenized text data and an array of valid length of the tokenized data
               on each node. It may also contain an array for train mask, an array for validation mask
               and an array for test mask.
        '''
        ids = []
        labels = []
        data = []
        train_mask = []
        valid_mask = []
        test_mask = []
        feats = [[] for _ in range(len(self.feat_fields))]

        print('Process {}'.format(self.input_files))
        for input_file in self.input_files:
            with open(input_file, 'r', encoding='utf-8') as reader:
                print(f">>>> Processing {input_file} <<<<")
                tokens_file, labels_file, inst_ids_file = self.generate_inputs(reader)
                if inst_ids_file is None:
                    print('The node data does not have node IDs.')
                    continue
                if self.nid_map is not None:
                    inst_ids_file = [self.nid_map[inst_id] for inst_id in inst_ids_file]
                else:
                    inst_ids_file = [int(inst_id) for inst_id in inst_ids_file]
                ids.extend(inst_ids_file)
                if labels_file is not None:
                    labels.extend(labels_file)
                if tokens_file is not None:
                    data.append(tokens_file)

            # need to open again the file otherwise the reader has finished the lines
            with open(input_file, 'r', encoding='utf-8') as reader:
                # If the file contains data split, we should read these fields from the file again.
                # When we read them again, the data is likely in the page cache. It shouldn't be a big problem.
                if self.has_data_split or len(self.feat_fields) > 0:
                    for line in reader:
                        line = line.strip()

                        if not line:
                            continue

                        json_data = json.loads(line)
                        if len(self.feat_fields) > 0 \
                                and not all(data_f in json_data for data_f in self.feat_fields):
                            print('Warning! the node data does not contain all data fields')
                            continue
                        if len(self.feat_fields) > 0:
                            for feat_i, feat_name in enumerate(self.feat_fields):
                                feats[feat_i].append(json_data[feat_name])

                        if self.has_data_split:
                            train, valid, test = _get_mask(json_data)
                            train_mask.append(train)
                            valid_mask.append(valid)
                            test_mask.append(test)
                    if self.has_data_split:
                        assert len(train_mask) == len(ids)

        ids = th.tensor(ids)
        ret_dict = {NODE_ID_IDX: ids}
        if len(labels) > 0:
            ret_dict[self.label_field] = th.tensor(labels)
            assert len(labels) == len(ids)
        if len(data) > 0:
            tokens = []
            valid_lengths = []
            for d in data:
                tokens.extend(d[0])
                valid_lengths.extend(d[1])
            assert len(tokens) == len(ids)
            assert len(valid_lengths) == len(ids)
            # tokens[0] has the shape of (dim,)
            ret_dict[TOKEN_IDX] = th.stack(tokens)
            ret_dict[VALID_LEN_IDX] = th.tensor(valid_lengths)
        if self.has_data_split:
            assert len(train_mask) == len(ids)
            ret_dict[TRAIN_IDX] = th.tensor(train_mask, dtype=th.bool)
            ret_dict[VALID_IDX] = th.tensor(valid_mask, dtype=th.bool)
            ret_dict[TEST_IDX] = th.tensor(test_mask, dtype=th.bool)
        if len(self.feat_fields) > 0:
            for feat_i, feat_name in enumerate(self.feat_fields):
                feat = th.tensor(feats[feat_i])
                assert len(feat) == len(ids)
                ret_dict[feat_name] = feat
        return ret_dict

class DataloaderGenerator:
    r"""Data Generator Interface. Generates pytorch dataloader based on the given file.

    Arguments:
        dataset_class (str, required): the dataset class from m5_dataloaders.datasets
            for processing the given files
        dataset_config (dict, required): the arguments (excluding input_files) for initializing Dataset class.
        batch_size (int, optional): how many samples per batch to load. (default: ``1``)
        data_sampler (pytorch Sampler type, optional): defines the strategy to draw
           samples from the dataset. Can be any ``Iterable`` class with ``__len__``
           implemented and supports taking as input a pytorch Dataset object to the initializer.
           If specified, :attr:`shuffle` must not be specified.
        num_workers (int, optional): how many subprocesses to use for data
           loading. ``0`` means that the data will be loaded in the main process.
           (default: ``0``)
        worker_init_fn (callable, optional): If not ``None``, this will be called on each
           worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
           input, after seeding and before data loading. (default: ``None``)
        shuffle (bool, optional): set to ``True`` to have the files reshuffled
           at every epoch. (default: ``False``)
        drop_incomplete_batches (bool, optional): If set to True, will drop the last batch from shard if it is
            incomplete, i.e. less than the specified micro-batch size. Needed when training models
            with batch norm to prevent batches having single data point. (default: ``False``)
            Should be set to False during evaluation or inference to avoid losing data.
        collate_fn (callable, optional): merges a list of samples to form a
           mini-batch of Tensor(s).  Used when using batched loading from a
           map-style dataset.
    """

    def __init__(
        self,
        dataset_class,
        dataset_config={},
        batch_size=1,
        data_sampler=None,
        num_workers=0,
        worker_init_fn=None,
        shuffle=False,
        drop_incomplete_batches=False,
        collate_fn=None,
    ):
        self.batch_size = batch_size

        if data_sampler is None:  # give default samplers
            if shuffle:
                data_sampler = RandomSampler
            else:
                data_sampler = SequentialSampler
        elif shuffle is not False:
            raise ValueError(
                "data_sampler is specified: expected unspecified "
                "shuffle option, but got shuffle={}".format(shuffle)
            )

        self.data_sampler = data_sampler

        self.num_workers = num_workers
        self.worker_init_fn = worker_init_fn
        self.drop_incomplete_batches = drop_incomplete_batches
        self.dataset_class = dataset_class
        self.dataset_config = dataset_config
        self.collate_fn = collate_fn

    def generate(self, input_files):
        if not isinstance(input_files, (list, tuple)):
            input_files = [input_files]
        data = self.dataset_class(input_files, **self.dataset_config)

        collate_fn = data.get_collate_fn() if self.collate_fn is None else self.collate_fn

        dataloader = DataLoader(
            data,
            sampler=self.data_sampler(data),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=self.drop_incomplete_batches,
        )
        return serialize(dataloader), len(data), None

def _collate_fn(data, max_len):
    ''' Collate function for Pytorch Dataloader

    The collate function is to convert data in the following format:
        [{key1: val11, key2: val21}, {key1: val12, key2: val22}, ...]
    to the following format:
        {key1: tensor([val11, val12, ...]), key2: tensor([val21, val22, ...])}
    '''
    keys = data[0].keys()
    ret = {key:[] for key in keys}
    for d in data:
        for key in keys:
            ret[key].append(d[key])
    for key in keys:
        # For TOKEN_IDX, ATT_MASK_IDX, TOKEN_TID_IDX, we need to create a 2-D tensor.
        # For multilabel classification labels, we also need to create a 2-D tensor
        # For single-label classification or regression labels, we use 1-D tensor
        ret[key] = th.stack(ret[key])\
             if (key in (TOKEN_IDX, ATT_MASK_IDX, TOKEN_TID_IDX)) or len(ret[key][0].shape) > 0 \
             else th.tensor(ret[key])
    return ret

def _load_edge_data(dataset_path, tokenizer, src_field, dst_field, label_field=None,
        task_type=CLASSIFICATION_TASK, text_fields=[], feat_fields=[],
        has_data_split=False, nid_maps=None, max_seq_length=512, num_dataset_workers=8,
        shuffle=False, seed=0,
        rank=-1):
    '''Load node data of one node type.

    Parameters
    ----------
    dataset_path: str
        Input data path.
    tokenizer: str
        Tensor tokenized. Reserved for future use.
    src_field: str
        The field name where source node IDs are stored.
    dst_field : str
        The field name where destination node IDs are stored.
    label_field : str
        The filed name where the label of an edge is stored.
    task_type : int
        This indicates what types of tasks we run on the edge data.
    text_fields : list of str
        The field names where text data are stored.
    has_data_split : bool
        This indicates whether to get data split from the input files.
    nid_maps : dict (str, int)
        This maps string IDs of source nodes and destination nodes to numeric node IDs.
    max_seq_length : int
        The maximal sequence length of the text used for tokenization.
    num_dataset_workers: int
        Number of prefetching worker process.
    shuffle: bool
        Whether shuffle data. Default: False
    seed: int
        Seed used in data shuffle.
    rank: int
        Rank used in distributed data processing
    '''
    extra_args = {
        'tokenizer': tokenizer,
        'src_field': src_field,
        'dst_field': dst_field,
        'text_fields': text_fields,
        'feat_fields': feat_fields,
        'label_field' : label_field,
        'task_type' : task_type,
        'max_seq_length': max_seq_length,
        'has_data_split': has_data_split,
        'nid_maps': nid_maps,
    }
    worker_init = WorkerInitObj(seed)

    dataloader_generator = DataloaderGenerator(
        dataset_class=JSONEdgeDataset,
        dataset_config=extra_args,
        batch_size=1000000,
        num_workers=0,
        shuffle=shuffle,
        worker_init_fn=worker_init,
        collate_fn=partial(_collate_fn, max_len=max_seq_length),
    )
    dataset_provider = DataProvider(
        dataloader_generator,
        dataset_path=dataset_path,
        num_prefetches=num_dataset_workers,
        shuffle=shuffle,
        #infinite=infinite,
        local_rank=rank,
        mpu=None,
    )

    ret = {}
    num_files = len(os.listdir(os.path.join(dataset_path)))
    t1 = time.time()
    dataloader, num_samples = dataset_provider.get_shard()
    i = 0
    while dataloader is not None:
        if i % 10 == 0:
            print(f"Loading {i} in {rank} | Global {num_files}")
        i += 1
        dataset_provider.prefetch_shard()
        for data in dataloader:
            for key in data:
                if key not in ret:
                    ret[key] = []
                ret[key].append(data[key])
        dataset_provider.release_shard()
        dataloader, num_samples = dataset_provider.get_shard()

    t2 = time.time()
    for key in ret:
        ret[key] = th.cat(ret[key])
    t3 = time.time()
    print('loading: {:.3f} seconds, concate: {:.3f} seconds'.format(t2 - t1, t3 - t2))

    return ret

def _load_node_data(dataset_path, tokenizer, use_hf_tokenizer, id_field, label_field,
        task_type, max_len, text_fields, has_data_split, nid_map,
        feat_fields = [],
        num_dataset_workers=8, shuffle=False, seed=0, rank=-1):
    '''Load node data of one node type.

    Parameters
    ----------
    dataset_path: str
        Input data path.
    tokenizer: str
        Tensor tokenized. Reserved for future use.
    use_hf_tokenizer: bool
        A flag indicates whether to use HuggingFace tokenizer.
    id_field : str
            The field name where node IDs are stored.
    label_field : str
        The filed name where the label of an edge is stored.
    task_type : int
        This indicates what types of tasks we run on the edge data.
    max_len : int
        The maximal sequence length of the text used for tokenization.
    text_fields : list of str
        The field names where text data are stored.
    has_data_split : bool
        This indicates whether to get data split from the input files.
    nid_maps : dict (str, int)
        This maps string IDs of source nodes and destination nodes to numeric node IDs.
    num_dataset_workers: int
        Number of prefetching worker process.
    shuffle: bool
        Whether shuffle data. Default: False
    seed: int
        Seed used in data shuffle.
    rank: int
        Rank used in distributed data processing
    '''
    extra_args = {
        'tokenizer': tokenizer,
        'use_hf_tokenizer': use_hf_tokenizer,
        'id_field': id_field,
        'label_field' : label_field,
        'task_type' : task_type,
        'max_seq_length': max_len,
        'text_fields': text_fields,
        'feat_fields': feat_fields,
        'has_data_split': has_data_split,
        'nid_map': nid_map,
    }
    worker_init = WorkerInitObj(seed)

    dataloader_generator = DataloaderGenerator(
        dataset_class=JSONNodeDataset,
        dataset_config=extra_args,
        batch_size=1000000,
        num_workers=0,
        shuffle=shuffle,
        worker_init_fn=worker_init,
        collate_fn=partial(_collate_fn, max_len=max_len),
    )
    dataset_provider = DataProvider(
        dataloader_generator,
        dataset_path=dataset_path,
        num_prefetches=num_dataset_workers,
        shuffle=shuffle,
        #infinite=infinite,
        local_rank=rank,
        mpu=None,
    )

    ret = {}
    num_files = len(os.listdir(os.path.join(dataset_path)))
    t1 = time.time()
    dataloader, num_samples = dataset_provider.get_shard()
    i = 0
    while dataloader is not None:
        if i % 10 == 0:
            print ("[Rank {}] Loading {} of {}".format(rank, i, num_files))
        i += 1
        dataset_provider.prefetch_shard()
        for data in dataloader:
            for key in data:
                if key not in ret:
                    ret[key] = []
                ret[key].append(data[key])
        dataset_provider.release_shard()
        dataloader, num_samples = dataset_provider.get_shard()

    t2 = time.time()
    for key in ret:
        ret[key] = th.cat(ret[key])
    t3 = time.time()
    print('loading: {:.3f} seconds, concate: {:.3f} seconds'.format(t2 - t1, t3 - t2))
    return ret

def _is_sorted(tensor):
    return np.all((tensor[1:] - tensor[:-1] > 0).numpy())

class StandardM5gnnDataset(M5gnnDataset):
    """r This class loads data in the standardized M5GNN data format.

    Parameters
    ----------
        raw_dir : str
            the path of the directory that contains all data of the dataset.
        name : str
            The name of the dataset
        rank : int
            The rank of the process in a distributed data processing cluster.
        verbose : bool
            This allows printing more information.
        max_node_seq_length : dict of string to int
            The dictionary that contains the max sequence length on each node type.
            The key is node type and the value is the max length.
        max_edge_seq_length : dict of string to int
            The dictionary that contains the max sequence length on each edge type.
            The key is edge type, the value is the max length
        m5_vocab : str
            The path of the m5 vocab used for tokenization
        hf_bert_model : str
            The name of the BERT model. We create a tokenizer accordingly based on the BERT model.
        num_worker : int
            The number of workers used in the multi processing data loading.
        nid_field : str
            The field name where node IDs are stored in the input files.
        src_field : str
            The field name of the source node IDs.
        dst_field : str
            The field name of the destination node IDs.
        nlabel_fields : dict of string
            The dictionary of field names that indicate labels on nodes
        elabel_fields : dict of string
            The dictionary of field names that indicate labels on edges
        ntask_types : dict of int
            They indicate the tasks on nodes
        etask_types : dict of int
            They indicate the tasks on edges
        split_ntypes : list of strings
            The node types where we split data.
        split_etypes : list of strings
            The edge types where we split data.
        ntext_fields : dict of string to list
            The dictionary that contains the list of field names to load text data on each node type.
        etext_fields : dict of string to list
            The dictionary that contains the list of field names to load text data on each edge type.
        nfeat_format : specifies the type of pregenerated node features to be added to the graph
        efeat_format : specifies the type of pregenerated edge features to be added to the graph
        nfeat_fields : dict of string to list
            The dictionary that contains the list of field names to load data on each node type.
        efeat_fields : dict of string to list
            The dictionary that contains the list of field names to load data on each edge type.
        file_format : str
            The format of the input files. The supported format: JSON.
        edge_name_delimiter : str
            The token that connects different entities in edge name. Default is '_'.
            For example, by default the edge name is 'EntA_Links_EntB', where 'Links' is the relation name.
            If we need composite name for the relation, we can use 'EntA::Links_Good::EntB', by setting edge_name_delimiter to '::'.
            The relation will then be 'Links_Good'.
    """
    def __init__(self, raw_dir, name, rank=-1,
                 force_reload=False,
                 verbose=True, max_node_seq_length={}, max_edge_seq_length={},
                 m5_vocab=None, hf_bert_model=None, num_worker=32, nfeat_format=None, efeat_format=None,
                 nid_fields={}, src_field='src_id', dst_field='dst_id',
                 nlabel_fields={}, elabel_fields={},
                 ntask_types={}, etask_types={},
                 split_ntypes=[], split_etypes=[],
                 ntext_fields={}, nfeat_fields={}, etext_fields={}, efeat_fields={},
                 file_format='JSON', edge_name_delimiter='_'):
        # change to directly load from the fsx and not by copied data. this way we will not loose track
        # of the original data.
        self._num_worker = num_worker
        self._name = name
        self._rank = rank
        self._world_size = th.distributed.get_world_size() if th.distributed.is_initialized() else 1
        self._max_ntext_length = max_node_seq_length
        self._max_etext_length = max_edge_seq_length
        self._ntext_fields = ntext_fields
        self._etext_fields = etext_fields
        self._e_graph_feat_fields = efeat_fields
        self._n_graph_feat_fields = nfeat_fields
        self._nid_fields = nid_fields
        self._src_field = src_field
        self._dst_field = dst_field
        self._nlabel_fields = nlabel_fields
        self._elabel_fields = elabel_fields
        self._ntask_types = ntask_types
        self._etask_types = etask_types
        self._split_ntypes = split_ntypes
        self._split_etypes = split_etypes
        self._file_format = file_format
        self._nfeat_format = nfeat_format
        self._efeat_format = efeat_format
        self._edge_name_delimiter = edge_name_delimiter

        assert m5_vocab is not None or hf_bert_model is not None
        if m5_vocab is not None:
            print('M5 tokenizer is used:', m5_vocab)
            self._tokenizer = SentencepieceTokenizer(m5_vocab)
            self._use_hf_tokenizer = False
        elif hf_bert_model is not None:
            print('HuggingFace tokenizer is used:', hf_bert_model)
            self._tokenizer = AutoTokenizer.from_pretrained(hf_bert_model)
            self._use_hf_tokenizer = True
        print("[{}/{}]Build M5gnnDataset.".format(self._rank, self._world_size))
        super(StandardM5gnnDataset, self).__init__(name,
                                                   url=None,
                                                   raw_dir=raw_dir,
                                                   force_reload=force_reload,
                                                   verbose=verbose)

    def load(self):
        ''' Load the constructed graph data from a file.
        '''
        # load from local storage
        root_path = self._raw_dir
        gname = self._name+'.bin'
        g, _ = dgl.load_graphs(os.path.join(root_path, gname))
        print(g[0])
        self._g = g[0]

    def has_cache(self):
        ''' Test if the graph data has been constructed.
        '''
        root_path = self._raw_dir
        gname = self._name+'.bin'
        return os.path.exists(os.path.join(root_path, gname))

    def save_graph(self, path):
        """ Save processed graph and the raw text data into disk.

        The raw texts for each node type are stored separately into different files.
        A query_asin_match.pkl file is created to store the meta-information of raw text data.

        Parameters
        ----------
        path : str
            Where to save the output
        """
        # save the processed data
        gname = self._name+'.bin'
        print("Save graph {} into {}".format(self._g, os.path.join(path, gname)))
        print('before save graph {}'.format(psutil.virtual_memory()))
        dgl.save_graphs(os.path.join(path, gname), [self._g])
        print('Done save graph {}'.format(psutil.virtual_memory()))

    def process(self):
        """ Parse raw data in JSON format and construct a DGL graph.
        """

        root_path = self.raw_dir

        file_names = [f for f in os.listdir(root_path)]
        ntypes = []
        canonical_etypes = []
        nfeats = []
        efeats = []
        print(f"Processing data in: {file_names}")
        for f in file_names:
            if 'nodes-' in f:
                ntypes.append(f[6:])
            if 'edges-' in f:
                ## Use user-defined edge_name_delimiter as seperator, default is '_'
                canonical_etypes.append(tuple(f[6:].split(self._edge_name_delimiter)))
            if 'nfeats-' in f:
                nfeats.append(f)
            if 'efeats-' in f:
                efeats.append(f)
        # We need to make sure all canonical etypes are valid. That is, the source and destination
        # node types must be one the node types.
        for src_type, etype, dst_type in canonical_etypes:
            print(f"Edge type: ({src_type},{dst_type},{ntypes})")
            assert src_type in ntypes
            assert dst_type in ntypes

        # If the node IDs are not integers, users need to list all node IDs in the dataset.
        # Here we generate the mapping from the original node IDs to integers if users provide
        # the node ID files.
        #
        # TODO(xiangsx) For distributed data process, now each worker will load the nid_map.
        # To save memory, we need a better solution.
        manager = Manager()
        self.nid_maps = {}
        for ntype in ntypes:
            nid_file = 'nid-' + ntype + '.txt'
            if nid_file in file_names:
                print('reading nid file from {}'.format(nid_file))
                path = os.path.join(root_path, nid_file)
                # The node ID file may not exist.
                if os.path.isfile(path):
                    self.nid_maps[ntype] = _read_nid_map(path, manager)
            else:
                # No nid map, we assume idx id is nid
                print("{} does not have nid map file".format(ntype))
                self.nid_maps[ntype] = None

        ndata = {}
        edata = {}
        edges = {}
        # Right now we only support loading data from the JSON format.
        assert self._file_format == 'JSON'
        # Load node data.
        # TODO assert that there is no None value in the text fields. This will fail the program for the HF tokenizer
        #  and will discard the rows for the M5 tokenizer. In the second case the number of nodes will be more that the
        #  number of features and the code will fail again.

        print('Start loading ndata')
        for ntype in ntypes:
            d = os.path.join(root_path, 'nodes-' + ntype)
            files = [os.path.join(d, f) for f in os.listdir(os.path.join(d))]
            max_len = self._max_ntext_length[ntype] if ntype in self._max_ntext_length else 512

            print('node type:', ntype)
            ndata[ntype] = _load_node_data(d, self._tokenizer, self._use_hf_tokenizer,
                                           self._nid_fields[ntype] if ntype in self._nid_fields else "id",
                                           self._nlabel_fields[ntype] if ntype in self._nlabel_fields else None,
                                           self._ntask_types[ntype] if ntype in self._ntask_types else CLASSIFICATION_TASK,
                                           max_len,
                                           self._ntext_fields[ntype] if ntype in self._ntext_fields else [],
                                           ntype in self._split_ntypes, self.nid_maps[ntype],
                                           feat_fields=self._n_graph_feat_fields[ntype] if ntype in self._n_graph_feat_fields else [],
                                           num_dataset_workers=self._num_worker,
                                           rank=self._rank)
            if self._world_size > 1:
                th.distributed.barrier()

        # Load edge data.
        print('Start loading edata')
        for etype in canonical_etypes:
            d = os.path.join(root_path, 'edges-' + self._edge_name_delimiter.join(list(etype)))
            files = [os.path.join(d, f) for f in os.listdir(os.path.join(d))]
            max_len = self._max_etext_length[etype] if etype in self._max_etext_length else 512

            print('etype type: [{}/{}]'.format(len(files), etype))
            dataset = _load_edge_data(d, self._tokenizer, self._src_field, self._dst_field,
                                      max_seq_length=max_len,
                                      label_field=self._elabel_fields[etype] if etype in self._elabel_fields else None,
                                      task_type=self._etask_types[etype] if etype in self._etask_types else CLASSIFICATION_TASK,
                                      text_fields=self._etext_fields[etype] if etype in self._etext_fields else [],
                                      feat_fields=self._e_graph_feat_fields[etype] if etype in self._e_graph_feat_fields else [],
                                      has_data_split=etype in self._split_etypes,
                                      nid_maps=(self.nid_maps[etype[0]], self.nid_maps[etype[2]]),
                                      num_dataset_workers=self._num_worker,
                                      rank=self._rank)
            if self._world_size > 1:
                th.distributed.barrier()
            edges[etype] = (dataset[EDGE_SRC_IDX], dataset[EDGE_DST_IDX])
            del dataset[EDGE_SRC_IDX]
            del dataset[EDGE_DST_IDX]
            edata[etype] = {}
            for k, v in dataset.items():
                edata[etype][k] = v

        if self._world_size > 1:
            # In distributed data processing,
            # we return a json object as a graph.
            # There is no need to create a DGLGraph.

            # Wait for everyone to finish the task
            th.distributed.barrier()
            # distributed data processing
            num_global_nodes = {}
            num_local_nodes = {}
            nfeat_split = {}
            for ntype in ndata:
                # split node feature in N part according to the corresponding node id
                # Rank 0 -> part 0
                # Rank 1 -> part 1
                # ...
                #
                # All features are sorted according to node id.
                num_local_node = ndata[ntype][NODE_ID_IDX].shape[0]
                num_nodes = th.zeros((self._world_size,),
                                        dtype=th.long)
                num_nodes[self._rank] = num_local_node
                th.distributed.all_reduce(num_nodes,
                    op=th.distributed.ReduceOp.SUM)
                num_nodes_per_chunk = num_nodes.tolist()
                num_local_nodes[ntype] = num_nodes_per_chunk
                num_nodes = sum(num_nodes_per_chunk)

                num_global_nodes[ntype] = num_nodes
                nfeat_split[ntype] = [self._world_size, \
                    (num_nodes + self._world_size - 1) // self._world_size, \
                    num_nodes] # number of partitions, part_size, total number of nodes
                nid = ndata[ntype][NODE_ID_IDX]
                nidx_split = ndata[ntype][NODE_ID_IDX] \
                    // ((num_nodes + self._world_size - 1) // self._world_size)
                nidx_split_size = []
                nidics_list = []
                node_feature = {}
                local_feats = {}

                for key in ndata[ntype]:
                    if key == NODE_ID_IDX:
                        continue
                    node_feature[key] = []
                    local_feats[key] = []

                for i in range(self._world_size):
                    mask = nidx_split == i
                    nid_i = nid[mask]
                    nidx_split_size.append(th.tensor([nid_i.shape[0]], dtype=th.int64))
                    nidics_list.append(nid_i)

                    for key in node_feature:
                        feat = ndata[ntype][key][mask]
                        node_feature[key].append(feat)

                # use scatter to sync across instances about the p2p tensor size
                gather_list = list(th.empty([self._world_size],
                                                dtype=th.int64).chunk(self._world_size))
                alltoall_cpu(self._rank, self._world_size, gather_list, nidx_split_size)

                # collect nids
                nidx_gather_list = [th.empty((int(num_feat),), dtype=nid.dtype) \
                    for num_feat in gather_list]
                alltoallv_cpu(self._rank, self._world_size, nidx_gather_list, nidics_list)

                # collect feature data
                local_nids = th.cat(nidx_gather_list)
                sort_idx = th.argsort(local_nids)
                for key in node_feature:
                    print("handle [{}]/[{}] shape {}".format(ntype, key, ndata[ntype][key].shape))
                    feat_gather_list = [th.empty((int(num_feat), ) if len(ndata[ntype][key].shape) == 1 \
                        else (int(num_feat), ndata[ntype][key].shape[1]), dtype=ndata[ntype][key].dtype) \
                        for num_feat in gather_list]
                    alltoallv_cpu(self._rank, self._world_size, feat_gather_list, node_feature[key])
                    local_feat = th.cat(feat_gather_list, dim=0)
                    # sort node features
                    local_feats[key] = local_feat[sort_idx]

                ndata[ntype] = local_feats

            # we do not need to shuffle the edge data
            # edge ids are treated as consecutive for edges starting from rank 0 to rank n
            # we only need to sync number edges info
            num_edges = {}
            num_local_edges = {}
            for etype, edge in edges.items():
                num_edge = th.tensor(edge[0].shape[0], dtype=th.int64)
                num_local_edge = th.zeros((self._world_size,), dtype=th.int64)
                num_local_edge[self._rank] = num_edge
                all_reduce_sum(num_local_edge)
                num_local_edges[etype] = num_local_edge.tolist()
                num_edges[etype] = sum(num_local_edges[etype])

            # return a json object as a graph
            self._g = {
                "edges": edges,
                "ndata": ndata,
                "edata": edata,
                "number_of_nodes": num_global_nodes,
                "num_local_nodes": num_local_nodes,
                "number_of_edges": num_edges,
                "nfeat_split": nfeat_split,
                "num_local_edges": num_local_edges,
            }
        else:
            # Construct the graph.
            self._g = dgl.heterograph(edges,
                    num_nodes_dict={name: len(nid_map) for name, nid_map in self.nid_maps.items()})
            for ntype in ndata:
                for name in ndata[ntype]:
                    self._g.nodes[ntype].data[name] = ndata[ntype][name]
                # if necessary, we should reshuffle node data according to the node IDs.
                if not _is_sorted(self._g.nodes[ntype].data[NODE_ID_IDX]):
                    node_ids, indices = th.sort(self._g.nodes[ntype].data[NODE_ID_IDX])
                    for key in self._g.nodes[ntype].data:
                        self._g.nodes[ntype].data[key] = self._g.nodes[ntype].data[key][indices]
                assert np.all((self._g.nodes[ntype].data[NODE_ID_IDX]
                    == th.arange(self._g.number_of_nodes(ntype))).numpy())
            for etype in edata:
                for name in edata[etype]:
                    self._g.edges[etype].data[name] = edata[etype][name]
            print('Done loading graph {}'.format(psutil.virtual_memory()))

            if self._nfeat_format:
                # load the node features
                self._load_node_features(nfeats, root_path)
                print ("Done loading node features")

            if self._efeat_format:
                # load the edge features
                self._load_edge_features(efeats, root_path)
                print ("Done loading edge features")

    def _load_edge_features(self, efeats, root_path):
        # load edge feature
        # e.g., efeat_dir can be "efeats-etype1"
        for efeat_dir in efeats:
            etype = efeat_dir[7:]
            feat_type_d = os.path.join(root_path, efeat_dir)
            efeat_type = [x for x in os.listdir(feat_type_d)]
            # load all types of features.
            # e.g., efeat_type_dir can be "embeddings"
            for efeat_type_dir in efeat_type:
                feat_data_d = os.path.join(feat_type_d, efeat_type_dir)
                files = [f for f in os.listdir(feat_data_d)]
                # sort files by alphabetical order to ensure the same order as node idx
                # note that the name of the file should be something like
                # "part-00000*", "part-00001*" ...
                # otherwise the order might not be correct
                files.sort()
                fdata = []
                for i, feat_f in enumerate(files):
                    print ("loading {}, {} left".format(os.path.join(efeat_dir, efeat_type_dir, feat_f), len(files) - i))
                    assert feat_f[-5:] == ".hdf5" or feat_f[-4:] == ".npy", \
                        "Node feature should be stored in hdf5 format or npy format"
                    fp = os.path.join(feat_data_d, feat_f)
                    if self._efeat_format == "hdf5":
                        with h5py.File(fp, "r") as f_read:
                            fdata.append(np.array(f_read['embeddings']))
                    elif self._efeat_format == "npy":
                        fdata.append(np.load(fp))
                    else:
                        raise ValueError(
                            "Only either 'hdf5' or 'npy' can be specified.")
                fdata = np.concatenate(fdata, axis=0)
                assert fdata.shape[0] == self._g.number_of_edges(etype), \
                    "Edge feature data {} should have the same count as the number of edges of {} and are not {} != {}".format(efeat_type_dir, etype, fdata.shape[0] , self._g.number_of_edges(etype))
                self._g.edges[etype].data[efeat_type_dir] = th.tensor(fdata)
            print ("Done loading {} edge features".format(efeat_dir))

    def _load_node_features(self, nfeats, root_path):
        # load node feature
        # e.g., nfeat_dir can be "nfeats-ntype1"
        for nfeat_dir in nfeats:
            ntype = nfeat_dir[7:]
            feat_type_d = os.path.join(root_path, nfeat_dir)
            nfeat_type = [x for x in os.listdir(feat_type_d)]
            # load all types of features.
            # e.g., nfeat_type_dir can be "embeddings"
            for nfeat_type_dir in nfeat_type:
                feat_data_d = os.path.join(feat_type_d, nfeat_type_dir)
                files = [f for f in os.listdir(feat_data_d)]
                # sort files by alphabetical order to ensure the same order as node idx
                # note that the name of the file should be something like
                # "part-00000*", "part-00001*" ...
                # otherwise the order might not be correct
                files.sort()
                fdata = []
                for i, feat_f in enumerate(files):
                    print ("loading {}, {} left".format(os.path.join(nfeat_dir, nfeat_type_dir, feat_f), len(files) - i))
                    assert feat_f[-5:] == ".hdf5" or feat_f[-4:] == ".npy", \
                        "Node feature should be stored in hdf5 format or npy format"
                    fp = os.path.join(feat_data_d, feat_f)
                    if self._nfeat_format == "hdf5":
                        with h5py.File(fp, "r") as f_read:
                            fdata.append(np.array(f_read['embeddings']))
                    elif self._nfeat_format == "npy":
                        fdata.append(np.load(fp))
                    else:
                        raise ValueError(
                            "Only either 'hdf5' or 'npy' can be specified.")
                fdata = np.concatenate(fdata, axis=0)
                assert fdata.shape[0] == self._g.number_of_nodes(ntype), \
                    "Node feature data {} should have the same count as the number of nodes of {}".format(nfeat_type_dir, ntype)
                self._g.nodes[ntype].data[nfeat_type_dir] = th.tensor(fdata)
            print ("Done loading {} node features".format(nfeat_dir))

    def __getitem__(self, idx):
        r"""Gets the data object at index.
        """
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1
