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
from transformers import BertTokenizer
import torch as th
import numpy as np

class Tokenizer:
    """ A wrapper to a tokenizer.

    It is defined to process multiple strings.

    Parameters
    ----------
    tokenizer : a tokenizer
    max_seq_length : int
        The maximal length of the tokenization results.
    """
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, strs):
        """ Tokenization function.

        Parameters
        ----------
        strs : list of strings.
            The text data to be tokenized.

        Returns
        -------
        a dict of tokenization results.
        """
        tokens = []
        att_masks = []
        type_ids = []
        for s in strs:
            t = self.tokenizer(s, max_length=self.max_seq_length,
                               truncation=True, padding='max_length', return_tensors='pt')
            tokens.append(t['input_ids'])
            # The masks are small integers. We can use int4 or int8 to store them.
            # This can signficantly reduce memory consumption.
            att_masks.append(t['attention_mask'].to(th.int8))
            type_ids.append(t['token_type_ids'].to(th.int8))
        return {'token_ids': th.cat(tokens, dim=0).numpy(),
                'attention_mask': th.cat(att_masks, dim=0).numpy(),
                'token_type_ids': th.cat(type_ids, dim=0).numpy()}

def parse_tokenize(op):
    """ Parse the tokenization configuration

    The parser returns a function that tokenizes text with HuggingFace tokenizer.
    The tokenization function returns a dict of three Pytorch tensors.

    Parameters
    ----------
    op : dict
        The configuration for the operation.

    Returns
    -------
    callable : a function to process the data.
    """
    tokenizer = BertTokenizer.from_pretrained(op['bert_model'])
    max_seq_length = int(op['max_seq_length'])
    return Tokenizer(tokenizer, max_seq_length)

def parse_feat_ops(confs):
    """ Parse the configurations for processing the features

    The feature transformation:
    {
        "feature_col":  ["<column name>", ...],
        "feature_name": "<feature name>",
        "data_type":    "<feature data type>",
        "transform":    {"name": "<operator name>", ...}
    }

    Parameters
    ----------
    confs : list
        A list of feature transformations.

    Returns
    -------
    list of tuple : The operations
    """
    ops = []
    assert isinstance(confs, list), \
            "The feature configurations need to be in a list."
    for feat in confs:
        # TODO(zhengda) we will support data type in the future.
        dtype = None
        if 'transform' not in feat:
            transform = None
        else:
            transform = feat['transform']
            assert 'name' in transform, \
                    "'name' must be defined in the transformation field."
            if transform['name'] == 'tokenize_hf':
                transform = parse_tokenize(transform)
            else:
                raise ValueError('Unknown operation: {}'.format(transform['name']))
        feat_name = feat['feature_name'] if 'feature_name' in feat else None
        assert 'feature_col' in feat, \
                "'feature_col' must be defined in a feature field."
        ops.append((feat['feature_col'], feat_name, dtype, transform))
    return ops

def process_features(data, ops):
    """ Process the data with the specified operations.

    This function runs the input operations on the corresponding data
    and returns the processed results.

    Parameters
    ----------
    data : dict
        The data stored as a dict.
    ops : list of tuples
        The operations. Each tuple contains two elements. The first element
        is the data name and the second element is a Python function
        to process the data.

    Returns
    -------
    dict : the key is the data name, the value is the processed data.
    """
    new_data = {}
    for feat_col, feat_name, dtype, op in ops:
        # If the transformation is defined on the feature.
        if op is not None:
            res = op(data[feat_col])
            if isinstance(res, dict):
                for key, val in res.items():
                    new_data[key] = val
            else:
                new_data[feat_name] = res
        # If the required data type is defined on the feature.
        elif dtype is not None:
            new_data[feat_name] = data[feat_col].astype(dtype)
        # If no transformation is defined for the feature.
        else:
            new_data[feat_name] = data[feat_col]
    return new_data

def get_valid_label_index(label):
    """ Get the index of the samples with valid labels.

    Some of the samples may not have labels. We require users to use
    NaN to indicate the invalid labels.

    Parameters
    ----------
    label : Numpy array
        The labels of the samples.

    Returns
    -------
    Numpy array : the index of the samples with valid labels in the list.
    """
    if np.issubdtype(label.dtype, np.floating):
        return np.logical_not(np.isnan(label)).nonzero()[0]
    elif np.issubdtype(label.dtype, np.integer):
        return np.arange(len(label))
    else:
        raise ValueError("GraphStorm only supports label data of integers and float." + \
                         f"This label data has data type of {label.dtype}.")

def process_labels(data, label_confs, is_node):
    """ Process labels

    Parameters
    ----------
    data : dict
        The data stored as a dict.
    label_conf : list of dict
        The list of configs to construct labels.
    is_node : bool
        Whether or not to process labels on nodes.

    Returns
    -------
    dict of tensors : labels (optional) and train/validation/test masks.
    """
    assert len(label_confs) == 1, "We only support one label per node/edge type."
    label_conf = label_confs[0]
    assert 'task_type' in label_conf, "'task_type' must be defined in the label field."
    if label_conf['task_type'] == 'classification':
        assert 'label_col' in label_conf, \
                "'label_col' must be defined in the label field."
        label_col = label_conf['label_col']
        label = data[label_col]
        assert np.issubdtype(label.dtype, np.integer) \
                or np.issubdtype(label.dtype, np.floating), \
                "The labels for classification have to be integers."
        valid_label_idx = get_valid_label_index(label)
        label = np.int32(label)
        num_samples = len(label)
    elif label_conf['task_type'] == 'regression':
        assert 'label_col' in label_conf, \
                "'label_col' must be defined in the label field."
        label_col = label_conf['label_col']
        label = data[label_col]
        valid_label_idx = get_valid_label_index(label)
        num_samples = len(label)
    else:
        assert label_conf['task_type'] == 'link_prediction', \
                "The task type must be classification, regression or link_prediction."
        assert not is_node, "link_prediction task must be defined on edges."
        label_col = label = None
        valid_label_idx = None
        # Any column in the data can define the number of samples in the data.
        for val in data.values():
            num_samples = len(val)
            break

    if 'split_pct' in label_conf:
        train_split, val_split, test_split = label_conf['split_pct']
        assert train_split + val_split + test_split <= 1, \
                "The data split of training/val/test cannot be more than the entire dataset."
        if valid_label_idx is None:
            rand_idx = np.random.permutation(num_samples)
        else:
            rand_idx = np.random.permutation(valid_label_idx)
        num_labels = len(rand_idx)
        num_train = int(num_labels * train_split)
        num_val = int(num_labels * val_split)
        num_test = int(num_labels * test_split)
        val_start = num_train
        val_end = num_train + num_val
        test_end = num_train + num_val + num_test
        train_idx = rand_idx[0:num_train]
        val_idx = rand_idx[val_start:val_end]
        test_idx = rand_idx[val_end:test_end]
        train_mask = np.zeros((num_samples,), dtype=np.int8)
        val_mask = np.zeros((num_samples,), dtype=np.int8)
        test_mask = np.zeros((num_samples,), dtype=np.int8)
        train_mask[train_idx] = 1
        val_mask[val_idx] = 1
        test_mask[test_idx] = 1
    if label_col is None:
        return {'train_mask': train_mask,
                'val_mask': val_mask,
                'test_mask': test_mask}
    else:
        return {label_col: label,
                'train_mask': train_mask,
                'val_mask': val_mask,
                'test_mask': test_mask}
