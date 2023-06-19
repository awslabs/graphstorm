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
import os
import sys

import numpy as np
import torch as th

from scipy.special import erfinv # pylint: disable=no-name-in-module
from transformers import BertTokenizer
from transformers import BertModel, BertConfig

from .file_io import HDF5Array, read_index_json

def _get_output_dtype(dtype_str):
    if dtype_str == 'float16':
        return np.float16
    elif dtype_str == 'float32':
        return np.float32
    else:
        assert False, f"Unknown dtype {dtype_str}, only support float16 and float32"

def _feat_astype(feats, dtype):
    return feats.astype(dtype) if dtype is not None else feats

class FeatTransform:
    """ The base class for feature transformation.

    Parameters
    ----------
    col_name : str
        The name of the column that contains the feature.
    feat_name : str
        The feature name used in the constructed graph.
    out_dtype:
        The dtype of the transformed feature.
        Default: None, we will not do data type casting.
    """
    def __init__(self, col_name, feat_name, out_dtype=None):
        self._col_name = col_name
        self._feat_name = feat_name
        self._out_dtype = out_dtype

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

class GlobalProcessFeatTransform(FeatTransform):
    """ The base class for transformations that can only be done using a single process.

        Some transformations need to do complex operations on the entire feature set,
        such as ranking. GlobalProcessFeatTransform loads features from files first,
        which can be done with multi-processing, and then do feature transformation
        after features are merged.
    """

    def after_merge_transform(self, feats):
        """ Do feature transformation after features are merged into a single
            array.

        Parameters
        ----------
        feats:
            feats to be processed
        """

class TwoPhaseFeatTransform(FeatTransform):
    """ The base class for two phasefeature transformation.

        The first phase is going to collect global information
        for data processing, e.g., collecting max and min value of floating
        point data or collecting the cardinality of categorical data.
        The second phase is to transform data using
        the global
        information collected in the first phase
    """

    def pre_process(self, feats):
        """ Pre-process data

        Parameters
        ----------
        feats:
            feats to be processed
        """

    def collect_info(self, info):
        """ Store global information for the second phase data processing

        Parameters
        ----------
        info:
            Information to be collected
        """

class CategoricalTransform(TwoPhaseFeatTransform):
    """ Convert the data into categorical values.

    The categorical values are stored as integers.

    Parameters
    ----------
    col_name : str
        The name of the column.
    feat_name : str
        The name of the feature.
    separator : str
        The separator to split data into multiple categorical values.
    transform_conf : dict
        The configuration for the feature transformation.
    """
    def __init__(self, col_name, feat_name, separator=None, transform_conf=None):
        self._val_dict = {}
        if transform_conf is not None and 'mapping' in transform_conf:
            self._val_dict = transform_conf['mapping']
            self._conf = transform_conf
        else:
            self._conf = transform_conf
        self._separator = separator
        super(CategoricalTransform, self).__init__(col_name, feat_name)

    def pre_process(self, feats):
        """ Pre-process data

        Parameters
        ----------
        feats: np.array
            Data to be processed
        """
        # If the mapping already exists, we don't need to do anything.
        if len(self._val_dict) > 0:
            return {}

        assert isinstance(feats, (np.ndarray, HDF5Array)), \
            "Feature of CategoricalTransform must be numpy array or HDF5Array"
        if isinstance(feats, HDF5Array):
            # TODO(xiangsx): This is not memory efficient.
            # It will load all data into main memory.
            feats = feats.to_numpy()

        if self._separator is None:
            return {self.feat_name: np.unique(feats)}
        else:
            assert feats.dtype.type is np.str_, \
                    "We can only convert strings to multiple categorical values with separaters."
            vals = []
            for feat in feats:
                vals.extend(feat.split(self._separator))
            return {self.feat_name: np.unique(vals)}

    def update_info(self, info):
        """ Store global information for the second phase data processing

        Parameters
        ----------
        info: list
            Information to be collected
        """
        # We already have the mapping.
        if len(self._val_dict) > 0:
            assert len(info) == 0
            return

        self._val_dict = {key: i for i, key in enumerate(np.unique(np.concatenate(info)))}
        # We need to save the mapping in the config object.
        if self._conf is not None:
            self._conf['mapping'] = self._val_dict

    def __call__(self, feats):
        """ Assign IDs to categorical values.

        Parameters
        ----------
        feats : np array
            Data with categorical values.

        Returns
        -------
        np.array
        """
        encoding = np.zeros((len(feats), len(self._val_dict)), dtype=np.int8)
        if self._separator is None:
            for i, feat in enumerate(feats):
                encoding[i, self._val_dict[feat]] = 1
        else:
            for i, feat in enumerate(feats):
                idx = [self._val_dict[val] for val in feat.split(self._separator)]
                encoding[i, idx] = 1
        return {self.feat_name: encoding}

class NumericalMinMaxTransform(TwoPhaseFeatTransform):
    """ Numerical value with Min-Max normalization.
        $val = (val-min) / (max-min)$

    Parameters
    ----------
    col_name : str
        The name of the column that contains the feature.
    feat_name : str
        The feature name used in the constructed graph.
    max_bound : float
        The maximum float value.
    min_bound : float
        The minimum float value
    out_dtype:
        The dtype of the transformed feature.
        Default: None, we will not do data type casting.
    """
    def __init__(self, col_name, feat_name,
                 max_bound=sys.float_info.max,
                 min_bound=-sys.float_info.max,
                 out_dtype=None):
        self._max_bound = max_bound
        self._min_bound = min_bound
        super(NumericalMinMaxTransform, self).__init__(col_name, feat_name, out_dtype)

    def pre_process(self, feats):
        """ Pre-process data

        Parameters
        ----------
        feats: np.array
            Data to be processed
        """
        assert isinstance(feats, (np.ndarray, HDF5Array)), \
            "Feature of NumericalMinMaxTransform must be numpy array or HDF5Array"
        if isinstance(feats, HDF5Array):
            # TODO(xiangsx): This is not memory efficient.
            # It will load all data into main memory.
            feats = feats.to_numpy()

        assert feats.dtype in [np.float64, np.float32, np.float16, np.int64, \
                              np.int32, np.int16, np.int8], \
            "Feature of NumericalMinMaxTransform must be floating points" \
            "or integers"
        assert len(feats.shape) <= 2, "Only support 1D fp feature or 2D fp feature"
        max_val = np.amax(feats, axis=0) if len(feats.shape) == 2 \
            else np.array([np.amax(feats, axis=0)])
        min_val = np.amin(feats, axis=0) if len(feats.shape) == 2 \
            else np.array([np.amin(feats, axis=0)])

        max_val[max_val > self._max_bound] = self._max_bound
        min_val[min_val < self._min_bound] = self._min_bound
        return {self.feat_name: (max_val, min_val)}

    def update_info(self, info):
        """ Store global information for the second phase data processing

        Parameters
        ----------
        info: list
            Information to be collected
        """
        max_vals = []
        min_vals = []
        for (max_val, min_val) in info:
            max_vals.append(max_val)
            min_vals.append(min_val)
        max_vals = np.stack(max_vals)
        min_vals = np.stack(min_vals)

        max_val = np.amax(max_vals, axis=0) if len(max_vals.shape) == 2 \
            else np.array([np.amax(max_vals, axis=0)])
        min_val = np.amin(min_vals, axis=0) if len(min_vals.shape) == 2 \
            else np.array([np.amin(min_vals, axis=0)])

        self._max_val = max_val
        self._min_val = min_val

    def __call__(self, feats):
        """ Do normalization for feats

        Parameters
        ----------
        feats : np array
            Data to be normalized

        Returns
        -------
        np.array
        """
        assert isinstance(feats, (np.ndarray, HDF5Array)), \
            "Feature of NumericalMinMaxTransform must be numpy array or HDF5Array"

        assert not np.any(self._max_val == self._min_val), \
            f"At least one element of Max Val {self._max_val} " \
            f"and Min Val {self._min_val} is equal. This will cause divide by zero error"

        if isinstance(feats, HDF5Array):
            # TODO(xiangsx): This is not memory efficient.
            # It will load all data into main memory.
            feats = feats.to_numpy()

        feats = (feats - self._min_val) / (self._max_val - self._min_val)
        feats[feats > 1] = 1 # any value > self._max_val is set to self._max_val
        feats[feats < 0] = 0 # any value < self._min_val is set to self._min_val
        feats = _feat_astype(feats, self._out_dtype)

        return {self.feat_name: feats}

class RankGaussTransform(GlobalProcessFeatTransform):
    """ Use Gauss rank transformation to transform input data

        The idea is from
        http://fastml.com/preparing-continuous-features-for-neural-networks-with-rankgauss/

    Parameters
    ----------
    col_name : str
        The name of the column that contains the feature.
    feat_name : str
        The feature name used in the constructed graph.
    out_dtype:
        The dtype of the transformed feature.
        Default: None, we will not do data type casting.
    epsilon: float
        Epsilon for normalization.
    """
    def __init__(self, col_name, feat_name, out_dtype=None, epsilon=None):
        self._epsilon = epsilon if epsilon is not None else 1e-6
        super(RankGaussTransform, self).__init__(col_name, feat_name, out_dtype)

    def __call__(self, feats):
        # do nothing. Rank Gauss is done after merging all arrays together.
        assert isinstance(feats, (np.ndarray, HDF5Array)), \
                f"The feature {self.feat_name} has to be NumPy array."
        assert np.issubdtype(feats.dtype, np.integer) \
                or np.issubdtype(feats.dtype, np.floating), \
                f"The feature {self.feat_name} has to be integers or floats."

        return {self.feat_name: feats}

    def after_merge_transform(self, feats):
        # The feats can be a numpy array or a numpy memmaped object
        # Get ranking information.
        feats = feats.argsort(axis=0).argsort(axis=0)
        feat_range = len(feats) - 1
        # norm to [-1, 1]
        feats = (feats / feat_range - 0.5) * 2
        feats = np.clip(feats, -1 + self._epsilon, 1 - self._epsilon)
        feats = erfinv(feats)

        feats = _feat_astype(feats, self._out_dtype)
        return feats

class Tokenizer(FeatTransform):
    """ A wrapper to a tokenizer.

    It is defined to process multiple strings.

    Parameters
    ----------
    col_name : str
        The name of the column that contains the text features
    feat_name : str
        The prefix of the tokenized data
    bert_model : str
        The name of the BERT model.
    max_seq_length : int
        The maximal length of the tokenization results.
    """
    def __init__(self, col_name, feat_name, bert_model, max_seq_length):
        super(Tokenizer, self).__init__(col_name, feat_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
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

class Text2BERT(FeatTransform):
    """ Compute BERT embeddings.

    It computes BERT embeddings.

    Parameters
    ----------
    col_name : str
        The name of the column that contains the text features
    feat_name : str
        The prefix of the tokenized data
    tokenizer : Tokenizer
        A tokenizer
    model_name : str
        The BERT model name.
    infer_batch_size : int
        The inference batch size.
    out_dtype:
        The dtype of the transformed feature.
        Default: None, we will not do data type casting.
    """
    def __init__(self, col_name, feat_name, tokenizer, model_name,
                 infer_batch_size=None, out_dtype=None):
        super(Text2BERT, self).__init__(col_name, feat_name, out_dtype)
        self.model_name = model_name
        self.lm_model = None
        self.tokenizer = tokenizer
        self.device = None
        self.infer_batch_size = infer_batch_size

    def _init(self):
        """ Initialize the BERT model.

        We should delay the BERT model initialization because we need to
        initialize the BERT model in the worker process instead of creating it
        in the master process and passing it to the worker process.
        """
        if th.cuda.is_available():
            gpu = int(os.environ['CUDA_VISIBLE_DEVICES']) \
                    if 'CUDA_VISIBLE_DEVICES' in os.environ else 0
            self.device = f"cuda:{gpu}"
        else:
            self.device = None

        if self.lm_model is None:
            config = BertConfig.from_pretrained(self.model_name)
            lm_model = BertModel.from_pretrained(self.model_name,
                                                 config=config)
            lm_model.eval()

            # We use the local GPU to compute BERT embeddings.
            if self.device is not None:
                lm_model = lm_model.to(self.device)
            self.lm_model = lm_model

    def __call__(self, strs):
        """ Compute BERT embeddings of the strings..

        Parameters
        ----------
        strs : list of strings.
            The text data.

        Returns
        -------
        dict: BERT embeddings.
        """
        self._init()
        outputs = self.tokenizer(strs)
        if self.infer_batch_size is not None:
            tokens_list = th.split(th.tensor(outputs['input_ids']), self.infer_batch_size)
            att_masks_list = th.split(th.tensor(outputs['attention_mask']),
                                      self.infer_batch_size)
            token_types_list = th.split(th.tensor(outputs['token_type_ids']),
                                        self.infer_batch_size)
        else:
            tokens_list = [th.tensor(outputs['input_ids'])]
            att_masks_list = [th.tensor(outputs['attention_mask'])]
            token_types_list = [th.tensor(outputs['token_type_ids'])]
        with th.no_grad():
            out_embs = []
            for tokens, att_masks, token_types in zip(tokens_list, att_masks_list,
                                                      token_types_list):
                if self.device is not None:
                    outputs = self.lm_model(tokens.to(self.device),
                                            attention_mask=att_masks.to(self.device).long(),
                                            token_type_ids=token_types.to(self.device).long())
                else:
                    outputs = self.lm_model(tokens,
                                            attention_mask=att_masks.long(),
                                            token_type_ids=token_types.long())
                out_embs.append(outputs.pooler_output.cpu().numpy())
        if len(out_embs) > 1:
            feats = np.concatenate(out_embs)
        else:
            feats = out_embs[0]

        feats = _feat_astype(feats, self._out_dtype)
        return {self.feat_name: feats}

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
        feats = _feat_astype(feats, self._out_dtype)
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
        out_dtype = _get_output_dtype(feat['out_dtype']) if 'out_dtype' in feat else None
        if 'transform' not in feat:
            transform = Noop(feat['feature_col'], feat_name, out_dtype=out_dtype)
        else:
            conf = feat['transform']
            assert 'name' in conf, "'name' must be defined in the transformation field."
            if conf['name'] == 'tokenize_hf':
                assert 'bert_model' in conf, \
                        "'tokenize_hf' needs to have the 'bert_model' field."
                assert 'max_seq_length' in conf, \
                        "'tokenize_hf' needs to have the 'max_seq_length' field."
                transform = Tokenizer(feat['feature_col'], feat_name, conf['bert_model'],
                                      int(conf['max_seq_length']))
            elif conf['name'] == 'bert_hf':
                assert 'bert_model' in conf, \
                        "'bert_hf' needs to have the 'bert_model' field."
                assert 'max_seq_length' in conf, \
                        "'bert_hf' needs to have the 'max_seq_length' field."
                infer_batch_size = int(conf['infer_batch_size']) \
                        if 'infer_batch_size' in conf else 1024
                transform = Text2BERT(feat['feature_col'], feat_name,
                                      Tokenizer(feat['feature_col'], feat_name,
                                                conf['bert_model'],
                                                int(conf['max_seq_length'])),
                                      conf['bert_model'],
                                      infer_batch_size=infer_batch_size,
                                      out_dtype=out_dtype)
            elif conf['name'] == 'max_min_norm':
                max_bound = conf['max_bound'] if 'max_bound' in conf else sys.float_info.max
                min_bound = conf['min_bound'] if 'min_bound' in conf else -sys.float_info.max
                transform = NumericalMinMaxTransform(feat['feature_col'],
                                                     feat_name,
                                                     max_bound,
                                                     min_bound,
                                                     out_dtype=out_dtype)
            elif conf['name'] == 'rank_gauss':
                epsilon = conf['epsilon'] if 'epsilon' in conf else None
                transform = RankGaussTransform(feat['feature_col'],
                                               feat_name,
                                               out_dtype=out_dtype,
                                               epsilon=epsilon)
            elif conf['name'] == 'to_categorical':
                separator = conf['separator'] if 'separator' in conf else None
                transform = CategoricalTransform(feat['feature_col'], feat_name,
                                                 separator=separator, transform_conf=conf)
            else:
                raise ValueError('Unknown operation: {}'.format(conf['name']))
        ops.append(transform)
    return ops

def preprocess_features(data, ops):
    """ Pre-process the data with the specified operations.

    This function runs the input pre-process operations on the corresponding data
    and returns the pre-process results. An example of preprocessing is getting
    the cardinality of a categorical feature.

    Parameters
    ----------
    data : dict
        The data stored as a dict.
    ops : list of FeatTransform
        The operations that transform features.

    Returns
    -------
    dict : the key is the data name, the value is the pre-processed data.
    """
    pre_data = {}
    for op in ops:
        res = op.pre_process(data[op.col_name])
        assert isinstance(res, dict)
        for key, val in res.items():
            pre_data[key] = val

    return pre_data

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
            # Check if has 1D features. If yes, convert to 2D features
            if len(val.shape) == 1:
                if isinstance(val, HDF5Array):
                    val = val.to_numpy().reshape(-1, 1)
                else:
                    val = val.reshape(-1, 1)
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
        if label.ndim == 1:
            return np.logical_not(np.isnan(label)).nonzero()[0]
        else:
            return np.nonzero(np.sum(np.isnan(label), axis=1) == 0)[0]
    elif np.issubdtype(label.dtype, np.integer):
        return np.arange(len(label))
    else:
        raise ValueError("GraphStorm only supports label data of integers and float." + \
                         f"This label data has data type of {label.dtype}.")

class CustomLabelProcessor:
    """ Process labels with custom data split.

    This allows users to define custom data split for training/validation/test.

    Parameters
    ----------
    col_name : str
        The column name for labels.
    label_name : str
        The label name.
    task_type : str
        The task type.
    train_idx : Numpy array
        The array that contains the index of training data points.
    val_idx : Numpy array
        The array that contains the index of validation data points.
    test_idx : Numpy array
        The array that contains the index of test data points.
    """
    def __init__(self, col_name, label_name, task_type,
                 train_idx=None, val_idx=None, test_idx=None):
        self._col_name = col_name
        self._label_name = label_name
        self._train_idx = train_idx
        self._val_idx = val_idx
        self._test_idx = test_idx
        self._task_type = task_type

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

    def data_split(self, num_samples):
        """ Split the data for training/validation/test.

        Parameters
        ----------
        num_samples : int
            The total number of data points.

        Returns
        -------
        dict of Numpy array
            The arrays for training/validation/test masks.
        """
        train_mask = np.zeros((num_samples,), dtype=np.int8)
        val_mask = np.zeros((num_samples,), dtype=np.int8)
        test_mask = np.zeros((num_samples,), dtype=np.int8)
        if self._train_idx is not None:
            train_mask[self._train_idx] = 1
        if self._val_idx is not None:
            val_mask[self._val_idx] = 1
        if self._test_idx is not None:
            test_mask[self._test_idx] = 1
        train_mask_name = 'train_mask'
        val_mask_name = 'val_mask'
        test_mask_name = 'test_mask'
        return {train_mask_name: train_mask,
                val_mask_name: val_mask,
                test_mask_name: test_mask}

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
        if self.col_name in data:
            label = data[self.col_name]
            num_samples = len(label)
        else:
            assert len(data) > 0, "The edge data is empty."
            label = None
            for val in data.values():
                num_samples = len(val)
                break
        res = self.data_split(num_samples)
        if label is not None and self._task_type == "classification":
            res[self.label_name] = np.int32(label)
        elif label is not None:
            res[self.label_name] = label
        return res

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
    if 'custom_split_filenames' in label_conf:
        custom_split = label_conf['custom_split_filenames']
        assert isinstance(custom_split, dict), \
                "Custom data split needs to provide train/val/test index."
        train_idx = read_index_json(custom_split['train']) if 'train' in custom_split else None
        val_idx = read_index_json(custom_split['valid']) if 'valid' in custom_split else None
        test_idx = read_index_json(custom_split['test']) if 'test' in custom_split else None
        label_col = label_conf['label_col'] if 'label_col' in label_conf else None
        return [CustomLabelProcessor(label_col, label_col, task_type,
                                     train_idx, val_idx, test_idx)]

    if 'split_pct' in label_conf:
        split_pct = label_conf['split_pct']
    else:
        logging.info("'split_pct' is not found. " + \
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
