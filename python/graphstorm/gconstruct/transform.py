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

import numpy as np
import torch as th
from transformers import BertTokenizer

from .file_io import HDF5Array

class FeatTransform:
    """ The base class for feature transformation.

    Parameters
    ----------
    col_name : str
        The name of the column that contains the feature.
    feat_name : str
        The feature name used in the constructed graph.
    """
    def __init__(self, col_name, feat_name):
        self._col_name = col_name
        self._feat_name = feat_name

    @property
    def col_name(self):
        """ The name of the column that contains the feature.
        """
        return self._col_name

    @property
    def feat_name(self):
        """ The feature name.
        """
        return self._feat_name

class Tokenizer(FeatTransform):
    """ A wrapper to a tokenizer.

    It is defined to process multiple strings.

    Parameters
    ----------
    col_name : str
        The name of the column that contains the text features
    feat_name : str
        The prefix of the tokenized data
    tokenizer : HuggingFace Tokenizer
        a tokenizer
    max_seq_length : int
        The maximal length of the tokenization results.
    """
    def __init__(self, col_name, feat_name, tokenizer, max_seq_length):
        super(Tokenizer, self).__init__(col_name, feat_name)
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
            assert isinstance(s, str), "The input of the tokenizer has to be a string."
            t = self.tokenizer(s, max_length=self.max_seq_length,
                               truncation=True, padding='max_length', return_tensors='pt')
            tokens.append(t['input_ids'])
            # The masks are small integers. We can use int4 or int8 to store them.
            # This can signficantly reduce memory consumption.
            att_masks.append(t['attention_mask'].to(th.int8))
            type_ids.append(t['token_type_ids'].to(th.int8))
        token_id_name = 'input_ids'
        atten_mask_name = 'attention_mask'
        token_type_id_name = 'token_type_ids'
        return {token_id_name: th.cat(tokens, dim=0).numpy(),
                atten_mask_name: th.cat(att_masks, dim=0).numpy(),
                token_type_id_name: th.cat(type_ids, dim=0).numpy()}

class Noop(FeatTransform):
    """ This doesn't transform the feature.
    """

    def __call__(self, feats):
        """ This transforms the features.

        Parameters
        ----------
        feats : Numpy array
            The feature data

        Returns
        -------
        dict : The key is the feature name, the value is the feature.
        """
        assert isinstance(feats, (np.ndarray, HDF5Array)), \
                f"The feature {self.feat_name} has to be NumPy array."
        assert np.issubdtype(feats.dtype, np.integer) \
                or np.issubdtype(feats.dtype, np.floating), \
                f"The feature {self.feat_name} has to be integers or floats."
        return {self.feat_name: feats}

def parse_feat_ops(confs):
    """ Parse the configurations for processing the features

    The feature transformation:
    {
        "feature_col":  ["<column name>", ...],
        "feature_name": "<feature name>",
        "transform":    {"name": "<operator name>", ...}
    }

    Parameters
    ----------
    confs : list
        A list of feature transformations.

    Returns
    -------
    list of FeatTransform : The operations that transform features.
    """
    ops = []
    assert isinstance(confs, list), \
            "The feature configurations need to be in a list."
    for feat in confs:
        assert 'feature_col' in feat, \
                "'feature_col' must be defined in a feature field."
        feat_name = feat['feature_name'] if 'feature_name' in feat else feat['feature_col']
        if 'transform' not in feat:
            transform = Noop(feat['feature_col'], feat_name)
        else:
            conf = feat['transform']
            assert 'name' in conf, "'name' must be defined in the transformation field."
            if conf['name'] == 'tokenize_hf':
                assert 'bert_model' in conf, \
                        "'tokenize_hf' needs to have the 'bert_model' field."
                tokenizer = BertTokenizer.from_pretrained(conf['bert_model'])
                assert 'max_seq_length' in conf, \
                        "'tokenize_hf' needs to have the 'max_seq_length' field."
                max_seq_length = int(conf['max_seq_length'])
                transform = Tokenizer(feat['feature_col'], feat_name, tokenizer, max_seq_length)
            else:
                raise ValueError('Unknown operation: {}'.format(conf['name']))
        ops.append(transform)
    return ops

def process_features(data, ops):
    """ Process the data with the specified operations.

    This function runs the input operations on the corresponding data
    and returns the processed results.

    Parameters
    ----------
    data : dict
        The data stored as a dict.
    ops : list of FeatTransform
        The operations that transform features.

    Returns
    -------
    dict : the key is the data name, the value is the processed data.
    """
    new_data = {}
    for op in ops:
        res = op(data[op.col_name])
        assert isinstance(res, dict)
        for key, val in res.items():
            new_data[key] = val
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

class LabelProcessor:
    """ Process labels

    Parameters
    ----------
    col_name : str
        The column name that contains the labels.
    label_name : str
        The label name.
    split_pct : list of int
        The percentage of training, validation and test.
    """
    def __init__(self, col_name, label_name, split_pct):
        self._col_name = col_name
        self._label_name = label_name
        self._split_pct = split_pct

    @property
    def col_name(self):
        """ The column name that contains the label.
        """
        return self._col_name

    @property
    def label_name(self):
        """ The label name.
        """
        return self._label_name

    def data_split(self, get_valid_idx, num_samples):
        """ Split the data

        Parameters
        ----------
        get_valid_idx : callable
            The function that returns the index of samples with valid labels.
        num_samples : int
            The total number of samples.

        Returns
        -------
        a dict of tensors : the training, validation, test masks.
        """
        train_split, val_split, test_split = self._split_pct
        assert train_split + val_split + test_split <= 1, \
                "The data split of training/val/test cannot be more than the entire dataset."
        rand_idx = get_valid_idx()
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
        train_mask_name = 'train_mask'
        val_mask_name = 'val_mask'
        test_mask_name = 'test_mask'
        return {train_mask_name: train_mask,
                val_mask_name: val_mask,
                test_mask_name: test_mask}

class ClassificationProcessor(LabelProcessor):
    """ Process the label for the classification task.
    """

    def __call__(self, data):
        """ Process the label for classification.

        This performs data split on the nodes/edges and generates training/validation/test masks.

        Parameters
        ----------
        data : dict of Tensors
            All data associated with nodes/edges of a node/edge type.

        Returns
        -------
        dict of Tensors : it contains the labels as well as training/val/test splits.
        """
        assert self.col_name in data, f"The label column {self.col_name} does not exist."
        label = data[self.col_name]
        assert np.issubdtype(label.dtype, np.integer) \
                or np.issubdtype(label.dtype, np.floating), \
                "The labels for classification have to be integers or floating points."
        valid_label_idx = get_valid_label_index(label)
        def permute_idx():
            return np.random.permutation(valid_label_idx)
        res = self.data_split(permute_idx, len(label))
        res[self.label_name] = np.int32(label)
        return res

class RegressionProcessor(LabelProcessor):
    """ Process the label for the regression task.
    """

    def __call__(self, data):
        """ Process the label for regression.

        This performs data split on the nodes/edges and generates training/validation/test masks.

        Parameters
        ----------
        data : dict of Tensors
            All data associated with nodes/edges of a node/edge type.

        Returns
        -------
        dict of Tensors : it contains the labels as well as training/val/test splits.
        """
        assert self.col_name in data, f"The label column {self.col_name} does not exist."
        label = data[self.col_name]
        valid_label_idx = get_valid_label_index(label)
        def permute_idx():
            return np.random.permutation(valid_label_idx)
        res = self.data_split(permute_idx, len(label))
        res[self.label_name] = label
        return res

class LinkPredictionProcessor(LabelProcessor):
    """ Process the label for the link prediction task.
    """

    def __call__(self, data):
        """ Process the label for link prediction.

        This performs data split on the edges and generates training/validation/test masks.

        Parameters
        ----------
        data : dict of Tensors
            All data associated with nodes/edges of a node/edge type.

        Returns
        -------
        dict of Tensors : it contains training/val/test splits for link prediction.
        """
        # Any column in the data can define the number of samples in the data.
        assert len(data) > 0, "The edge data is empty."
        for val in data.values():
            num_samples = len(val)
            break
        def permute_idx():
            return np.random.permutation(num_samples)
        return self.data_split(permute_idx, num_samples)

def parse_label_ops(confs, is_node):
    """ Parse the configurations to generate the label processor.

    Parameters
    ----------
    confs : dict
        Contain the configuration for labels.
    is_node : bool
        Whether the configurations are defined for nodes.

    Returns
    -------
    list of LabelProcessor : the label processors generated from the configurations.
    """
    assert len(confs) == 1, "We only support one label per node/edge type."
    label_conf = confs[0]
    assert 'task_type' in label_conf, "'task_type' must be defined in the label field."
    task_type = label_conf['task_type']
    # By default, we use all labels for training.
    if 'split_pct' in label_conf:
        split_pct = label_conf['split_pct']
    else:
        print("'split_pct' is not found. " + \
                "Use the default data split: train(80%), valid(10%), test(10%).")
        split_pct = [0.8, 0.1, 0.1]

    if task_type == 'classification':
        assert 'label_col' in label_conf, \
                "'label_col' must be defined in the label field."
        label_col = label_conf['label_col']
        return [ClassificationProcessor(label_col, label_col, split_pct)]
    elif task_type == 'regression':
        assert 'label_col' in label_conf, \
                "'label_col' must be defined in the label field."
        label_col = label_conf['label_col']
        return [RegressionProcessor(label_col, label_col, split_pct)]
    else:
        assert task_type == 'link_prediction', \
                "The task type must be classification, regression or link_prediction."
        assert not is_node, "link_prediction task must be defined on edges."
        return [LinkPredictionProcessor(None, None, split_pct)]

def process_labels(data, label_processors):
    """ Process labels

    Parameters
    ----------
    data : dict
        The data stored as a dict.
    label_processors : list of LabelProcessor
        The list of operations to construct labels.

    Returns
    -------
    dict of tensors : labels (optional) and train/validation/test masks.
    """
    assert len(label_processors) == 1, "We only support one label per node/edge type."
    return label_processors[0](data)

def do_multiprocess_transform(conf, feat_ops, label_ops, in_files):
    """ Test whether the input data requires multiprocessing.

    If the input data is stored in HDF5 and we don't need to read
    the data in processing, we don't need to use multiprocessing to
    read data. It needs to meet two conditions to test if the data
    requires processing: 1) we don't need to transform the features
    and 2) there are no labels (finding the data split
    needs to read the labels in memory).

    Parameters
    ----------
    conf : dict
        The configuration of the input data.
    feat_ops : dict of FeatTransform
        The operations run on the input features.
    label_ops : list of LabelProcessor
        The operations run on the labels.
    in_files : list of strings
        The input files.

    Returns
    -------
    bool : whether we need to read the data with multiprocessing.
    """
    # If there is only one input file.
    if len(in_files) == 1:
        return False

    # If the input data are stored in HDF5, we need additional checks.
    if conf['format']['name'] == "hdf5" and label_ops is None:
        # If it doesn't have features.
        if feat_ops is None:
            return False

        for op in feat_ops:
            # If we need to transform the feature.
            if not isinstance(op, Noop):
                return True
        # If none of the features require processing.
        return False
    else:
        return True
