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
import abc
import json
import warnings

import numpy as np
import torch as th

from scipy.special import erfinv # pylint: disable=no-name-in-module
from transformers import AutoTokenizer
from transformers import AutoModel, AutoConfig

from .file_io import read_index
from .utils import ExtMemArrayWrapper, ExtFeatureWrapper, generate_hash

LABEL_STATS_FIELD = "training_label_stats"
LABEL_STATS_FREQUENCY_COUNT = "frequency_cnt"

CLASSIFICATION_LABEL_STATS_TYPES = [LABEL_STATS_FREQUENCY_COUNT]

def _check_label_stats_type(task_type, label_stats_type):
    if task_type == "classification":
        if label_stats_type is not None:
            assert label_stats_type in CLASSIFICATION_LABEL_STATS_TYPES, \
                "GraphStorm only support collecting training label statistics in " \
                f"following format {CLASSIFICATION_LABEL_STATS_TYPES} for classification tasks."
        return label_stats_type

    return None

def collect_label_stats(feat_name, label_stats):
    """ Collect label stats according to different stats_type

    Parameters
    ----------
    feat_name: str
        Feature name that stores label stats. It composes of two parts:
        LABEL_STATS_FIELD+<label feature name>
    label_stats: list
        A list of stats created by differet works.

    Return:
    tuple: A tuple of
        1. Name of the corresponding label
        2. label stats type (Used for printing statistics)
        3. label stats
    """
    label_feat_name = feat_name[len(LABEL_STATS_FIELD):]
    stats_type = label_stats[0][0]
    if stats_type == LABEL_STATS_FREQUENCY_COUNT:
        label_frequency = {}
        for _, vals, counts in label_stats:
            for val, cnt in zip(vals, counts):
                if val not in label_frequency:
                    label_frequency[int(val)] = int(cnt)
                else:
                    label_frequency[int(val)] += int(cnt)
        return (label_feat_name, LABEL_STATS_FREQUENCY_COUNT, label_frequency)

    raise RuntimeError(f"Unknown label stats type {stats_type}")

def print_label_stats(stats):
    """ Print label stats

    Parameters
    ----------
    stats: tuple
        stats_type, stats
    """
    stats_type, stats = stats
    if stats_type == LABEL_STATS_FREQUENCY_COUNT:
        logging.debug("Counts of each label:")
        logging.debug("[Label Index] | Label Name | Counts")
        for i, label_name in enumerate(stats):
            logging.debug("[%d]\t%s: \t%d", i, label_name, stats[label_name])

def print_node_label_stats(ntype, label_name, stats):
    """ Print label stats of nodes

    Parameters
    ----------
    ntype: str
        Node type
    label_name: str
        Label name
    stats: tuple
        stats_type, stats
    """
    logging.debug("Label statistics of %s nodes with label name %s", ntype, label_name)
    print_label_stats(stats)

def print_edge_label_stats(etype, label_name, stats):
    """ Print label stats of nodes

    Parameters
    ----------
    etype: tuple
        Edge type
    label_name: str
        Label name
    stats: tuple
        stats_type, stats
    """
    logging.debug("Label statistics of %s edges with label name %s", etype, label_name)
    print_label_stats(stats)

def compress_label_stats(stats):
    """ Compress stats into a json object

    Parameters
    ----------
    stats: tuple
        stats_type, stats
    """
    stats_type, stats = stats
    if stats_type == LABEL_STATS_FREQUENCY_COUNT:
        info = {"stats_type": LABEL_STATS_FREQUENCY_COUNT,
                "info": stats}
        return info
    else:
        raise RuntimeError(f"Unknown label stats type {stats_type}")

def save_node_label_stats(output_dir, node_label_stats):
    """ Save node label stats into disk

    Parameters
    ----------
    output_dir: str
        Path to store node label stats
    node_label_stats: dict
        Node label stats to save
    """
    info = {}
    for ntype in node_label_stats:
        stats_summary = {}
        for label_name, stats in node_label_stats[ntype].items():
            stats_summary[label_name] = compress_label_stats(stats)
        info[ntype] = stats_summary
    with open(os.path.join(output_dir, 'node_label_stats.json'), 'w', encoding="utf8") as f:
        json.dump(info, f, indent=4)

def save_edge_label_stats(output_dir, edge_label_stats):
    """ Save edge label stats into disk

    Parameters
    ----------
    output_dir: str
        Path to store edge label stats
    edge_label_stats: dict
        Edge label stats to save
    """
    info = {}
    for etype in edge_label_stats:
        stats_summary = {}
        for label_name, stats in edge_label_stats[etype].items():
            stats_summary[label_name] = compress_label_stats(stats)
        info[",".join(etype)] = stats_summary
    with open(os.path.join(output_dir, 'edge_label_stats.json'), 'w', encoding="utf8") as f:
        json.dump(info, f, indent=4)

def _get_output_dtype(dtype_str):
    if dtype_str == 'float16':
        return np.float16
    elif dtype_str == 'float32':
        return np.float32
    elif dtype_str == 'float64':
        return np.float64
    elif dtype_str == 'int8':
        return np.int8 # for train, val, test mask
    else:
        assert False, f"Unknown dtype {dtype_str}, only support int8, float16, float32, " + \
                       "and float64."

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

    @property
    def out_dtype(self):
        """ Output feature dtype
        """
        return self._out_dtype

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
        feats = self.call(feats)
        return self.as_out_dtype(feats)

    @abc.abstractmethod
    def call(self, feats):
        """ This function implements the feature transformation logic

        Parameters
        ----------
        feats : Numpy array
            The feature data

        Returns
        -------
        dict : The key is the feature name, the value is the feature.
        """

    def as_out_dtype(self, feats):
        """ Convert feats into out_dtype
            By default (out_dtype is None), it does nothing.

        Parameters
        ----------
        feats: Numpy array or dict of Numpy array
            The feature data

        Returns
        -------
        Numpy array or dict: the output feature with dtype of out_dtype
        """
        if self.out_dtype is None:
            return feats

        if isinstance(feats, dict):
            return {key: feat.astype(self.out_dtype) \
                        for key, feat in feats.items()}
        else:
            return feats.astype(self.out_dtype)

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

        Return:
            np.array: processed feature
        """

    def call(self, feats):
        raise NotImplementedError

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

        Return:
            dict: {feature_name: feats_statistics}
        """

    def update_info(self, info):
        """ Store global information for the second phase data processing

        Parameters
        ----------
        info: list
            Information to be collected. A list of feats_statistics returned by pre_process
        """

    def call(self, feats):
        raise NotImplementedError

class BucketTransform(FeatTransform):
    """ Convert the numerical value into buckets.

    Parameters
    ----------
    col_name : str
        The name of the column that contains the feature.
    feat_name : str
        The feature name used in the constructed graph.
    bucket_cnt: num:
        The count of bucket lists used in the bucket feature transform
    bucket_range: list[num]:
        The range of bucket lists only defining the start and end point
    slide_window_size: int
        interval or range within which numeric values are grouped into buckets
    out_dtype:
        The dtype of the transformed feature.
        Default: None, we will not do data type casting.
    """
    def __init__(self, col_name, feat_name, bucket_cnt,
                 bucket_range, slide_window_size=0, out_dtype=None):
        assert bucket_cnt is not None, \
            f"bucket count must be provided for bucket feature transform of feature {feat_name}"
        assert bucket_range is not None and len(bucket_range) == 2, \
            f"bucket range must be provided for bucket feature transform of feature {feat_name}"
        self.bucket_cnt = bucket_cnt
        self.bucket_range = bucket_range
        self.slide_window_size = slide_window_size
        out_dtype = np.float32 if out_dtype is None else out_dtype
        super(BucketTransform, self).__init__(col_name, feat_name, out_dtype)

    def call(self, feats):
        """ This transforms the features.

        Parameters
        ----------
        feats : Numpy array
            The numerical feature data

        Returns
        -------
        dict : The key is the feature name, the value is the feature.
        """
        assert isinstance(feats, (np.ndarray, ExtMemArrayWrapper)), \
                f"The feature {self.feat_name} has to be NumPy array " \
                f"within numerical value."
        if isinstance(feats, ExtMemArrayWrapper):
            feats = feats.to_numpy()
        assert np.issubdtype(feats.dtype, np.integer) \
                or np.issubdtype(feats.dtype, np.floating), \
                f"The feature {self.feat_name} has to be integers or floats."

        encoding = np.zeros((len(feats), self.bucket_cnt), dtype=np.int8)
        max_val = max(self.bucket_range)
        min_val = min(self.bucket_range)
        bucket_size = (max_val - min_val) / self.bucket_cnt
        for i, f in enumerate(feats):
            high_val = min(f + (self.slide_window_size / 2), max_val)
            low_val = max(f - (self.slide_window_size / 2), min_val)

            # Determine upper and lower bucket membership
            low_val -= min_val
            high_val -= min_val
            low_idx = max(low_val // bucket_size, 0)
            high_idx = min(high_val // bucket_size + 1, self.bucket_cnt)

            idx = np.arange(start=low_idx, stop=high_idx, dtype=int)
            encoding[i][idx] = 1.0

            # Avoid edge case not in bucket
            if f >= max_val:
                encoding[i][-1] = 1.0
            if f <= min_val:
                encoding[i][0] = 1.0

        return {self.feat_name: encoding}

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
            # We assume the keys of a categorical mapping are strings.
            # But previously keys can be integers. So we convert them
            # into strings.
            self._val_dict = \
                {str(key): val for key, val in transform_conf['mapping'].items()}
            self._conf = transform_conf
        else:
            self._conf = transform_conf
        self._separator = separator
        super(CategoricalTransform, self).__init__(col_name, feat_name)

    def pre_process(self, feats):
        # If the mapping already exists, we don't need to do anything.
        if len(self._val_dict) > 0:
            return {}

        assert isinstance(feats, (np.ndarray, ExtMemArrayWrapper)), \
            f"Feature of CategoricalTransform must be a numpy " \
            f"array or ExtMemArray for feature {self.feat_name}"
        if isinstance(feats, ExtMemArrayWrapper):
            # TODO(xiangsx): This is not memory efficient.
            # It will load all data into main memory.
            feats = feats.to_numpy()

        feats = feats[feats != None] # pylint: disable=singleton-comparison
        if self._separator is None:
            return {self.feat_name: np.unique(feats.astype(str))}
        else:
            assert feats.dtype.type is np.str_, \
                "We can only convert strings to multiple categorical values with separaters." \
                f"for feature {self.feat_name}"
            vals = []
            for feat in feats:
                vals.extend(feat.split(self._separator))
            return {self.feat_name: np.unique(vals)}

    def update_info(self, info):
        # We already have the mapping.
        if len(self._val_dict) > 0:
            assert len(info) == 0
            return

        self._val_dict = {str(key): i for i, key in enumerate(np.unique(np.concatenate(info)))}
        # We need to save the mapping in the config object.
        if self._conf is not None:
            self._conf['mapping'] = self._val_dict

    def call(self, feats):
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
                if feat is None:
                    continue
                if str(feat) in self._val_dict:
                    encoding[i, self._val_dict[str(feat)]] = 1
                # if key does not exist, keep the feature as all zeros.
        else:
            for i, feat in enumerate(feats):
                if feat is None:
                    continue
                idx = [self._val_dict[val] for val in feat.split(self._separator) \
                       if val in self._val_dict]
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
        The maximum float value. Any number larger than max_bound will be set to max_bound.
    min_bound : float
        The minimum float value. Any number smaller than min_bound will be set to min_bound.
    max_val : list of float
        Define the value of `max` in the Min-Max normalization formula for each feature.
        If max_val is set, max_bound will be ignored.
    min_val : list of float
        Define the value of `min` in the Min-Max normalization formula for each feature.
        If min_val is set, min_bound will be ignored.
    out_dtype:
        The dtype of the transformed feature.
        Default: None, we will not do data type casting.
    transform_conf : dict
        The configuration for the feature transformation.
    """
    def __init__(self, col_name, feat_name,
                 max_bound=sys.float_info.max,
                 min_bound=-sys.float_info.max,
                 max_val=None, min_val=None,
                 out_dtype=None, transform_conf=None):
        self._max_val = np.array(max_val, dtype=np.float32) if max_val is not None else None
        self._min_val = np.array(min_val, dtype=np.float32) if min_val is not None else None
        self._conf = transform_conf
        if out_dtype in [np.float64, np.float32, np.float16, np.int64, \
                              np.int32, np.int16, np.int8]:
            fifo = np.finfo(out_dtype)
        else:
            fifo = np.finfo(np.float32)
        self._max_bound = fifo.max if max_bound >= fifo.max else max_bound
        self._min_bound = -fifo.max if min_bound <= -fifo.max else min_bound
        out_dtype = np.float32 if out_dtype is None else out_dtype
        super(NumericalMinMaxTransform, self).__init__(col_name, feat_name, out_dtype)

    def pre_process(self, feats):
        assert isinstance(feats, (np.ndarray, ExtMemArrayWrapper)), \
            f"Feature {self.feat_name} of NumericalMinMaxTransform " \
            "must be numpy array or ExtMemArray"

        # The max and min of $val = (val-min) / (max-min)$ is pre-defined
        # in the transform_conf, return max_val and min_val directly
        if self._max_val is not None and self._min_val is not None:
            return {self.feat_name: (self._max_val, self._min_val)}

        if isinstance(feats, ExtMemArrayWrapper):
            # TODO(xiangsx): This is not memory efficient.
            # It will load all data into main memory.
            feats = feats.to_numpy()

        if feats.dtype not in [np.float64, np.float32, np.float16, np.int64, \
                              np.int32, np.int16, np.int8]:
            logging.warning("The feature %s has to be floating points or integers,"
                            "but get %s. Try to cast it into float32",
                            self.feat_name, feats.dtype)
            try:
                # if input dtype is not float or integer, we need to cast the data
                # into float32
                feats = feats.astype(np.float32)
            except: # pylint: disable=bare-except
                raise ValueError(f"The feature {self.feat_name} has to be integers or floats.")
        assert len(feats.shape) <= 2, \
            "Only support 1D fp feature or 2D fp feature, " \
            f"but get {len(feats.shape)}D feature for {self.feat_name}"

        if self._max_val is None:
            max_val = np.amax(feats, axis=0) if len(feats.shape) == 2 \
                else np.array([np.amax(feats, axis=0)])
            max_val[max_val > self._max_bound] = self._max_bound
        else:
            max_val = self._max_val

        if self._min_val is None:
            min_val = np.amin(feats, axis=0) if len(feats.shape) == 2 \
                else np.array([np.amin(feats, axis=0)])
            min_val[min_val < self._min_bound] = self._min_bound
        else:
            min_val = self._min_val

        return {self.feat_name: (max_val, min_val)}

    def update_info(self, info):
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

        # We need to save the max_val and min_val in the config object.
        if self._conf is not None:
            self._conf['max_val'] = self._max_val.tolist()
            self._conf['min_val'] = self._min_val.tolist()

    def call(self, feats):
        """ Do normalization for feats

        Parameters
        ----------
        feats : np array
            Data to be normalized

        Returns
        -------
        np.array
        """
        assert isinstance(feats, (np.ndarray, ExtMemArrayWrapper)), \
            f"Feature {self._feat_name} of NumericalMinMaxTransform " \
            "must be numpy array or ExtMemArray"

        assert not np.any(self._max_val == self._min_val), \
            f"At least one element of Max Val {self._max_val} " \
            f"and Min Val {self._min_val} is equal for feature {self.feat_name}. " \
            "This will cause divide-by-zero error"

        if isinstance(feats, ExtMemArrayWrapper):
            # TODO(xiangsx): This is not memory efficient.
            # It will load all data into main memory.
            feats = feats.to_numpy()

        if feats.dtype not in [np.float64, np.float32, np.float16, np.int64, \
                              np.int32, np.int16, np.int8]:
            try:
                # if input dtype is not float or integer, we need to cast the data
                # into float32
                feats = feats.astype(np.float32)
            except: # pylint: disable=bare-except
                raise ValueError(f"The feature {self.feat_name} has to be integers or floats.")

        feats = (feats - self._min_val) / (self._max_val - self._min_val)
        feats[feats > 1] = 1 # any value > self._max_val is set to self._max_val
        feats[feats < 0] = 0 # any value < self._min_val is set to self._min_val

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
    uniquify: bool
        When uniquify is set to True, GraphStorm will
        deduplicate the input features before computing the
        rank gauss norm on the input features.
        Note: Set it to True will make feature processing slower.
        Default: False.
    """
    def __init__(self, col_name, feat_name, out_dtype=None, epsilon=None, uniquify=False):
        self._epsilon = epsilon if epsilon is not None else 1e-6
        self._uniquify = uniquify
        out_dtype = np.float32 if out_dtype is None else out_dtype
        super(RankGaussTransform, self).__init__(col_name, feat_name, out_dtype)

    def __call__(self, feats):
        # Overwrite __call__ to avoid cast the feature into out_dtype.
        # This is not the final output, we should not cast the feature into out_dtype
        # will cast the feature in after_merge_transform
        return self.call(feats)

    def call(self, feats):
        # do nothing. Rank Gauss is done after merging all arrays together.
        assert isinstance(feats, (np.ndarray, ExtMemArrayWrapper)), \
                f"The feature {self.feat_name} has to be NumPy array."

        if np.issubdtype(feats.dtype, np.integer) \
            or np.issubdtype(feats.dtype, np.floating): \
            return {self.feat_name: feats}
        else:
            logging.warning("The feature %s has to be floating points or integers,"
                            "but get %s. Try to cast it into float32",
                            self.feat_name, feats.dtype)
            try:
                # if input dtype is not float or integer, we need to cast the data
                # into float32
                feats = feats.astype(np.float32)
            except: # pylint: disable=bare-except
                raise ValueError(f"The feature {self.feat_name} has to be integers or floats.")

            return {self.feat_name: feats}

    def after_merge_transform(self, feats):
        # The feats can be a numpy array or a numpy memmaped object
        # Get ranking information.
        if isinstance(feats, ExtMemArrayWrapper):
            feats = feats.to_numpy()

        if self._uniquify:
            uni_feats, indices = np.unique(feats, axis=0, return_inverse=True)

            uni_feats = uni_feats.argsort(axis=0).argsort(axis=0)
            if len(uni_feats) == 1:
                logging.warning("features of %s are identical. Will return all 0s",
                                self.feat_name)
                return self.as_out_dtype(np.zeros(feats.shape))

            feat_range = len(uni_feats) - 1
            uni_feats = (uni_feats / feat_range - 0.5) * 2
            uni_feats = np.clip(uni_feats, -1 + self._epsilon, 1 - self._epsilon)
            uni_feats = erfinv(uni_feats)
            feats = uni_feats[indices]
        else:
            feats = feats.argsort(axis=0).argsort(axis=0)
            feat_range = len(feats) - 1
            # norm to [-1, 1]
            feats = (feats / feat_range - 0.5) * 2
            feats = np.clip(feats, -1 + self._epsilon, 1 - self._epsilon)
            feats = erfinv(feats)

        return self.as_out_dtype(feats)

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
        The name of the lm model. We keep the parameter name for backward compatibilities
    max_seq_length : int
        The maximal length of the tokenization results.
    """
    def __init__(self, col_name, feat_name, bert_model, max_seq_length):
        super(Tokenizer, self).__init__(col_name, feat_name)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.max_seq_length = max_seq_length

    def call(self, feats):
        """ Tokenization function.

        Parameters
        ----------
        strs : list of strings.
            The text data to be tokenized.

        Returns
        -------
        a dict of tokenization results.
        """
        strs = feats
        tokens = []
        att_masks = []
        type_ids = []
        for s in strs:
            assert isinstance(s, str), \
                "The input of the tokenizer has to be a string for feature {self.feat_name}."
            t = self.tokenizer(s, max_length=self.max_seq_length,
                               truncation=True, padding='max_length', return_tensors='pt')
            tokens.append(t['input_ids'])
            # The masks are small integers. We can use int4 or int8 to store them.
            # This can signficantly reduce memory consumption.
            att_masks.append(t['attention_mask'].to(th.int8))
            # Some tokenizer doesn't produce `token_type_ids`, replace w. zeros
            type_ids.append(t.get('token_type_ids', th.zeros_like(t['input_ids'])).to(th.int8))
        token_id_name = 'input_ids'
        atten_mask_name = 'attention_mask'
        token_type_id_name = 'token_type_ids'
        return {token_id_name: th.cat(tokens, dim=0).numpy(),
                atten_mask_name: th.cat(att_masks, dim=0).numpy(),
                token_type_id_name: th.cat(type_ids, dim=0).numpy()}

class Text2BERT(FeatTransform):
    """ Compute LM embeddings.

    It computes LM embeddings.

    Parameters
    ----------
    col_name : str
        The name of the column that contains the text features
    feat_name : str
        The prefix of the tokenized data
    tokenizer : Tokenizer
        A tokenizer
    model_name : str
        The LM model name.
    infer_batch_size : int
        The inference batch size.
    out_dtype:
        The dtype of the transformed feature.
        Default: None, we will not do data type casting.
    """
    def __init__(self, col_name, feat_name, tokenizer, model_name,
                 infer_batch_size=None, out_dtype=None):
        out_dtype = np.float32 if out_dtype is None else out_dtype
        super(Text2BERT, self).__init__(col_name, feat_name, out_dtype)
        self.model_name = model_name
        self.lm_model = None
        self.tokenizer = tokenizer
        self.device = None
        self.infer_batch_size = infer_batch_size

    def _init(self):
        """ Initialize the LM model.

        We should delay the LM model initialization because we need to
        initialize the LM model in the worker process instead of creating it
        in the master process and passing it to the worker process.
        """
        if th.cuda.is_available():
            gpu = int(os.environ['CUDA_VISIBLE_DEVICES']) \
                    if 'CUDA_VISIBLE_DEVICES' in os.environ else 0
            self.device = f"cuda:{gpu}"
        else:
            self.device = "cpu"

        if self.lm_model is None:
            config = AutoConfig.from_pretrained(self.model_name)
            lm_model = AutoModel.from_pretrained(self.model_name, config)
            lm_model.eval()

            # We use the local GPU to compute LM embeddings.
            if self.device is not None:
                lm_model = lm_model.to(self.device)
            self.lm_model = lm_model

    def call(self, feats):
        """ Compute LM embeddings of the strings..

        Parameters
        ----------
        strs : list of strings.
            The text data.

        Returns
        -------
        dict: LM embeddings.
        """
        self._init()
        strs = feats
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

        return {self.feat_name: feats}

class Noop(FeatTransform):
    """ This doesn't transform the feature.

    Parameters
    ----------
    col_name : str
        The name of the column that contains the feature.
    feat_name : str
        The feature name used in the constructed graph.
    out_dtype : str
        The dtype of the transformed feature.
        Default: None, we will not do data type casting.
    truncate_dim : int
        When provided, will truncate the output float-vector feature to the specified dimension.
        This is useful when the feature is a multi-dimensional vector and we only need
        a subset of the dimensions, e.g. for Matryoshka Representation Learning embeddings.
    """
    def __init__(self, col_name, feat_name, out_dtype=None, truncate_dim=None):
        out_dtype = np.float32 if out_dtype is None else out_dtype
        super(Noop, self).__init__(col_name, feat_name, out_dtype)
        self.truncate_dim = truncate_dim

    def call(self, feats):
        """ This transforms the features.

        Parameters
        ----------
        feats : Numpy array
            The feature data

        Returns
        -------
        dict : The key is the feature name, the value is the feature.
        """
        assert isinstance(feats, (np.ndarray, ExtMemArrayWrapper)), \
                f"The feature {self.feat_name} has to be NumPy array."
        assert np.issubdtype(feats.dtype, np.integer) \
                or np.issubdtype(feats.dtype, np.floating), \
                f"The feature {self.feat_name} has to be integers or floats."
        if self.truncate_dim is not None:
            if isinstance(feats, np.ndarray):
                feats = feats[:, :self.truncate_dim]
            else:
                assert isinstance(feats, ExtMemArrayWrapper)
                # Need to convert to in-memory array to make truncation possible
                feats = feats.to_numpy()[:, :self.truncate_dim]
        return {self.feat_name: feats}

class HardEdgeNegativeTransform(TwoPhaseFeatTransform):
    """ Translate input data into node ids for hard negative stored as edge features

    Parameters
    ----------
    col_name : str
        The name of the column that contains the feature.
    feat_name : str
        The feature name used in the constructed graph.
    separator : str
        The separator to split data into multiple node ids.
    """
    def __init__(self, col_name, feat_name, separator=None):
        self._target_ntype = None
        self._target_etype = None
        self._nid_map = None
        self._separator = separator
        super().__init__(col_name, feat_name, out_dtype=np.int64)

    def set_target_etype(self, etype):
        """ Set the etype of this hard edge negative transformation ops
            and associated hard negative information. For example,
            self._target_ntype.

        Parameters
        ----------
        etype : tuple of str
            The edge type the hard negatives belonging to.
        """
        raise NotImplementedError

    @property
    def target_etype(self):
        """ The the edge type of this hard negative transformation.
        """
        return self._target_etype

    @property
    def neg_ntype(self):
        """ Return the node type of hard negatives
        """
        return self._target_ntype

    def set_id_maps(self, id_maps):
        """ Set ID mapping for converting raw string ID to Graph ID
        """
        assert self._target_ntype is not None, \
            "The target node type should be set, it can be the source node type " \
            "or the destination node type depending on the hard negative case."
        assert self._target_ntype in id_maps, \
            f"The nid mapping should have the mapping for {self._target_ntype}. " \
            f"But only has {id_maps.keys()}"
        self._nid_map = id_maps

    def pre_process(self, feats):
        """ Pre-process input feats

            Not all the edges have the same number of hard negatives.
            Thus we need to know the maxmun number of hard negatives first.

        Parameters
        ----------
        feats:
            feats to be processed

        Return:
            dict: {feature_name: feats_statistics}
        """
        assert isinstance(feats, (np.ndarray, ExtMemArrayWrapper)), \
            f"Feature {self.feat_name} of HardEdgeNegativeTransform " \
            "must be numpy array or ExtMemArray"

        if self._separator is None:
            # It is possible that the input is a
            # np.array(np.array(), np.array(), ...)
            # when the input is a array of variable length list.
            if len(feats.shape) == 1:
                max_dim = max(len(feat) for feat in feats)
            else:
                max_dim = feats.shape[1]
        else:
            assert len(feats.shape) == 1 or feats.shape[1] == 1, \
                "When a separator is given, the input feats " \
                f"of {self.feat_name} must be a list of strings."

            feats = feats.astype(str)
            max_dim = 0
            for feat in feats:
                dim_size = len(feat.split(self._separator))
                max_dim = dim_size if dim_size > max_dim else max_dim
        return {self.feat_name: max_dim}

    def update_info(self, info):
        max_dim = max(info)
        self._max_dim = max_dim

    def call(self, feats):
        """ Parse hard negatives as features

        Hard negatives can be stored as string arrays where
        each string is a node id. For example:

        .. code::

            src | dst | hard_negs
            s_0 | d_0 | ["h_0", "h_1"]
            s_1 | d_1 | ["h_2", "h_3"]
            s_2 | d_2 | ["h_4", ""]
            s_3 | d_3 | ["h_5", "h_3"]
            ...

        Or strings with a delimeter to separate node ids.
        For example:

        .. code::

            src | dst | hard_negs
            s_0 | d_0 | "h_0;h_1"
            s_1 | d_1 | "h_2;h_3"
            s_2 | d_2 | "h_4"
            s_3 | d_3 | "h_5;h_3"
            ...

        Parameters
        ----------
        feats : np array
            Data with hard negatives.

        Returns
        -------
        np.array
        """
        assert self._target_ntype is not None, \
            "The target node type should be set, it can be the source node type " \
            "or the destination node type depending on the hard negative case."
        nid_map = self._nid_map[self._target_ntype]

        # It is possible that some edges do not
        # have enough pre-defined hard negatives.
        # In certain cases, GraphStorm will fill the
        # un-provided hard negatives with -1s.
        neg_ids = np.full((len(feats), self._max_dim), -1, dtype=np.int64)
        for i, feat in enumerate(feats):
            if feat is None:
                continue

            if self._separator is None:
                raw_ids = feat
            else:
                raw_ids = np.array(feat.split(self._separator))
            nids, _ = nid_map.map_id(raw_ids.astype(
                nid_map.map_key_dtype))

            # Write hard negative node ids into the hard
            # negative features.
            # When len(raw_ids) < self._max_dim (max negatives
            # per edge), GraphStorm fills the rest with -1.
            neg_ids[i][:nids.shape[0]] = nids

        return {self.feat_name: neg_ids}

class HardEdgeDstNegativeTransform(HardEdgeNegativeTransform):
    """ Translate input data (destination node raw id) into GraphStorm node ids
        for hard negative stored as edge features.
    """

    def set_target_etype(self, etype):
        self._target_etype = tuple(etype)
        # target node type is destination node type.
        self._target_ntype = etype[2]

def parse_feat_ops(confs, input_data_format=None):
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
    input_data_format: str
        Input data format, it can be parquet, csv, hdf5.

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
        assert (isinstance(feat['feature_col'], str) and feat['feature_col'] != "") \
               or (isinstance(feat['feature_col'], list) and len(feat['feature_col']) >= 1), \
            "feature column should not be empty"
        feat_name = feat['feature_name'] if 'feature_name' in feat else feat['feature_col']

        out_dtype = _get_output_dtype(feat['out_dtype']) if 'out_dtype' in feat else None
        if 'transform' not in feat:
            transform = Noop(
                feat['feature_col'],
                feat_name,
                out_dtype=out_dtype,
                truncate_dim=feat.get('truncate_dim', None)
            )
        else:
            conf = feat['transform']
            assert 'name' in conf, "'name' must be defined in the transformation field."
            if conf['name'] == 'tokenize_hf':
                assert 'bert_model' in conf, \
                        "'tokenize_hf' needs to have the 'bert_model' field."
                assert 'max_seq_length' in conf, \
                        "'tokenize_hf' needs to have the 'max_seq_length' field."
                if isinstance(feat['feature_col'], list) and len(feat['feature_col']) > 1:
                    raise RuntimeError("Not support multiple column for tokenize_hf transformation")
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
                # TODO: Not support max_min_norm feature transformation on multiple columns
                # without explicitly defining max_val and min_val.
                # Otherwise, the definition of max_val and min_val for each column is unclear.
                # define max_val and min_val for each column.
                if isinstance(feat['feature_col'], list) and len(feat['feature_col']) > 1:
                    assert 'max_val' in conf and 'min_val' in conf, \
                        "max_val and min_val for max_min_norm feature transformation is needed"
                    warnings.warn("The same max_val and min_val will apply to all columns")
                max_bound = conf['max_bound'] if 'max_bound' in conf else sys.float_info.max
                min_bound = conf['min_bound'] if 'min_bound' in conf else -sys.float_info.max
                max_val = conf['max_val'] if 'max_val' in conf else None
                min_val = conf['min_val'] if 'min_val' in conf else None
                transform = NumericalMinMaxTransform(feat['feature_col'],
                                                     feat_name,
                                                     max_bound,
                                                     min_bound,
                                                     max_val,
                                                     min_val,
                                                     out_dtype=out_dtype, transform_conf=conf)
            elif conf['name'] == 'rank_gauss':
                epsilon = conf['epsilon'] if 'epsilon' in conf else None
                uniquify = conf['uniquify'] if 'uniquify' in conf else False
                transform = RankGaussTransform(feat['feature_col'],
                                               feat_name,
                                               out_dtype=out_dtype,
                                               epsilon=epsilon,
                                               uniquify=uniquify)
            elif conf['name'] == 'to_categorical':
                separator = conf['separator'] if 'separator' in conf else None
                # TODO: Not support categorical feature transformation on multiple columns.
                # It is not clear to define category mapping for each column
                if isinstance(feat['feature_col'], list) and len(feat['feature_col']) > 1:
                    raise RuntimeError("Do not support categorical "
                                       "feature transformation on multiple columns")
                transform = CategoricalTransform(feat['feature_col'], feat_name,
                                                 separator=separator, transform_conf=conf)
            elif conf['name'] == 'bucket_numerical':
                assert 'bucket_cnt' in conf, \
                    "It is required to count of bucket information for bucket feature transform"
                assert 'range' in conf, \
                    "It is required to provide range information for bucket feature transform"
                if isinstance(feat['feature_col'], list) and len(feat['feature_col']) > 1:
                    warnings.warn("The same bucket range and count will be applied to all columns")
                bucket_cnt = conf['bucket_cnt']
                bucket_range = conf['range']
                if 'slide_window_size' in conf:
                    slide_window_size = conf['slide_window_size']
                else:
                    slide_window_size = 0
                transform = BucketTransform(feat['feature_col'],
                                               feat_name,
                                               bucket_cnt=bucket_cnt,
                                               bucket_range=bucket_range,
                                               slide_window_size=slide_window_size,
                                               out_dtype=out_dtype)
            elif conf['name'] == 'edge_dst_hard_negative':
                assert input_data_format not in ["hdf5"], \
                    "Edge_dst_hard_negative transformation does not work with hdf5 inputs."
                separator = conf['separator'] if 'separator' in conf else None
                transform = HardEdgeDstNegativeTransform(feat['feature_col'],
                                                         feat_name,
                                                         separator=separator)
            else:
                raise ValueError('Unknown operation: {}'.format(conf['name']))
        ops.append(transform)

    two_phase_feat_ops = []
    after_merge_feat_ops = {}
    hard_edge_neg_ops = []
    for op in ops:
        if isinstance(op, TwoPhaseFeatTransform):
            two_phase_feat_ops.append(op)
        if isinstance(op, GlobalProcessFeatTransform):
            after_merge_feat_ops[op.feat_name] = op
        if isinstance(op, HardEdgeNegativeTransform):
            hard_edge_neg_ops.append(op)

    return ops, two_phase_feat_ops, after_merge_feat_ops, hard_edge_neg_ops

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
        if isinstance(op.col_name, str):
            col_name = [op.col_name]
        else:
            col_name = op.col_name
        for col in col_name:
            res = op.pre_process(data[col])
            # Do not expect multiple keys for multiple columns, the expected output will only
            # have 1 key/val pair. But for single column, some feature transformations like
            # Tokenizer will return multiple key-val pairs, so do not check for single column
            if len(col_name) > 1:
                assert isinstance(res, dict) and len(res) == 1, \
                    f"It is expected only have one feature name after preprocessing features " \
                    f"for multiple column feature transformation, but get {len(res)}"
            for key, val in res.items():
                if key in pre_data:
                    assert pre_data[key] == val, f"It is expected same preprocessed value " \
                                                 f"for each column but get {pre_data[key]} " \
                                                 f"and {val}"
                pre_data[key] = val

    return pre_data

def process_features(data, ops, ext_mem_path=None):
    """ Process the data with the specified operations.

    This function runs the input operations on the corresponding data
    and returns the processed results.

    Parameters
    ----------
    data : dict
        The data stored as a dict.
    ops : list of FeatTransform
        The operations that transform features.
    ext_mem_path: str or None
        The path of external memory

    Returns
    -------
    dict : the key is the data name, the value is the processed data.
    """
    new_data = {}
    for op in ops:
        if isinstance(op.col_name, str):
            col_name = [op.col_name]
        else:
            col_name = op.col_name
        tmp_key, wrapper = "", ""
        # Create ExtFeatureWrapper for multiple columns on external memory
        if ext_mem_path is not None:
            hash_hex_feature_path = generate_hash()
            feature_path = 'feature_{}_{}'.format(op.feat_name, hash_hex_feature_path)
            feature_path = ext_mem_path + feature_path
            os.makedirs(feature_path)
            wrapper = ExtFeatureWrapper(feature_path)
        else:
            wrapper = None
        for col in col_name:
            res = op(data[col])
            # Do not expect multiple keys for multiple columns, the expected output will only
            # have 1 key/val pair. But for single column, some feature transformations like
            # Tokenizer will return multiple key-val pairs, so do not check for single column
            if len(col_name) > 1:
                assert isinstance(res, dict) and len(res) == 1, \
                    f"It is expected only have one feature name after the process_features " \
                    f"for multiple column feature transformation, but get {len(res)}"
            for key, val in res.items():
                # Check if it has 1D features. If yes, convert to 2D features
                if len(val.shape) == 1:
                    if isinstance(val, ExtMemArrayWrapper):
                        val = val.to_numpy().reshape(-1, 1)
                    else:
                        val = val.reshape(-1, 1)
                if len(col_name) == 1:
                    new_data[key] = val
                    continue
                tmp_key = key
                # Use external memory if it is required
                if ext_mem_path is not None:
                    wrapper.append(val)
                else:
                    val = np.column_stack((new_data[key], val)) \
                        if key in new_data else val
                    new_data[key] = val

        if len(col_name) > 1 and ext_mem_path is not None:
            new_data[tmp_key] = wrapper.merge()

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
    id_col : str or tuple
        The name of the ID column.
    task_type : str
        The task type.
    train_idx : Numpy array
        The array that contains the index of training data points.
    val_idx : Numpy array
        The array that contains the index of validation data points.
    test_idx : Numpy array
        The array that contains the index of test data points.
    stats_type: str
        Speicfy how to summarize label statistics
    mask_field_names: tuple of str
        Field name of train, validation and test masks
        Default: ("train_mask", "val_mask", "test_mask")
    """
    def __init__(self, col_name, label_name, id_col, task_type,
                 train_idx=None, val_idx=None, test_idx=None,
                 stats_type=None, mask_field_names=("train_mask", "val_mask", "test_mask")):
        self._id_col = id_col
        self._col_name = col_name
        self._label_name = label_name
        self._train_idx = set(train_idx) if train_idx is not None else None
        self._val_idx = set(val_idx) if val_idx is not None else None
        self._test_idx = set(test_idx) if test_idx is not None else None
        self._task_type = task_type
        self._stats_type = stats_type

        assert isinstance(mask_field_names, tuple) and len(mask_field_names) == 3, \
            "mask_field_names must be a tuple with three strings " \
            "for training mask, validation mask and test mask, respectively." \
            "For example ('tmask', 'vmask', 'tmask')."
        self._mask_field_names = mask_field_names

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

    @property
    def train_mask_name(self):
        """ The field name of the train mask
        """
        return self._mask_field_names[0]

    @property
    def val_mask_name(self):
        """ The field name of the validation mask
        """
        return self._mask_field_names[1]

    @property
    def test_mask_name(self):
        """ The field name of the test mask
        """
        return self._mask_field_names[2]

    def data_split(self, ids):
        """ Split the data for training/validation/test.

        Parameters
        ----------
        ids : numpy array
            The array of IDs.

        Returns
        -------
        dict of Numpy array
            The arrays for training/validation/test masks.
        """
        num_samples = len(ids)
        train_mask = np.zeros((num_samples,), dtype=np.int8)
        val_mask = np.zeros((num_samples,), dtype=np.int8)
        test_mask = np.zeros((num_samples,), dtype=np.int8)
        for i, idx in enumerate(ids):
            if self._train_idx is not None and idx in self._train_idx:
                train_mask[i] = 1
            elif self._val_idx is not None and idx in self._val_idx:
                val_mask[i] = 1
            elif self._test_idx is not None and idx in self._test_idx:
                test_mask[i] = 1
        train_mask_name = self.train_mask_name # default: 'train_mask'
        val_mask_name = self.val_mask_name # default: 'val_mask'
        test_mask_name = self.test_mask_name # default: 'test_mask'
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
        label = data[self.col_name] if self.col_name in data else None
        if isinstance(self._id_col, str):
            # For node label, the id_col is expected to be one single column
            assert self._id_col in data, \
                    f"The input data does not have ID column {self._id_col}."
        else:
            # For edge label, the id_col is expected a be a pair of (src_id_col, dest_id_col)
            assert self._id_col[0] and self._id_col[1] in data,\
                f"The input data does not have ID column {self._id_col[0]} and {self._id_col[1]}"

        if isinstance(self._id_col, str):
            res = self.data_split(data[self._id_col])
        else:
            res = self.data_split(list(zip(data[self._id_col[0]], data[self._id_col[1]])))
        if label is not None and self._task_type == "classification":
            res[self.label_name] = np.int32(label)
            if self._stats_type is not None:
                if self._stats_type == LABEL_STATS_FREQUENCY_COUNT:
                    # get train labels
                    train_labels = res[self.label_name][ \
                        res[self.train_mask_name].astype(np.bool_)]
                    vals, counts = np.unique(train_labels, return_counts=True)
                    res[LABEL_STATS_FIELD+self.label_name] = \
                        (LABEL_STATS_FREQUENCY_COUNT, vals, counts)
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
    stats_type: str
        Speicfy how to summarize label statistics
        Default: None
    mask_field_names: tuple of str
        Specify the field name of train, validation and test masks
        Default: ["train_mask", "val_mask", "test_mask"]
    """
    def __init__(self, col_name, label_name, split_pct,
                 stats_type=None, mask_field_names=("train_mask", "val_mask", "test_mask")):
        self._col_name = col_name
        self._label_name = label_name
        self._split_pct = split_pct
        self._stats_type = stats_type
        assert isinstance(mask_field_names, tuple) and len(mask_field_names) == 3, \
            "mask_field_names must be a tuple with three strings " \
            "for training mask, validation mask and test mask, respectively." \
            "For example ('tmask', 'vmask', 'tmask')."
        self._mask_field_names = mask_field_names

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

    @property
    def mask_field_names(self):
        """ The field names of train, validation and test masks
        """
        return self._mask_field_names

    @property
    def train_mask_name(self):
        """ The field name of the train mask
        """
        return self._mask_field_names[0]

    @property
    def val_mask_name(self):
        """ The field name of the validation mask
        """
        return self._mask_field_names[1]

    @property
    def test_mask_name(self):
        """ The field name of the test mask
        """
        return self._mask_field_names[2]

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
        if train_split == 0 and val_split == 0 and test_split == 0:
            # Train, val and test are all zero
            # Ignore the split
            return {}
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
        train_mask_name = self.train_mask_name # default: 'train_mask'
        val_mask_name = self.val_mask_name # default: 'val_mask'
        test_mask_name = self.test_mask_name # default: 'test_mask'
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

        if self._stats_type is not None:
            if self._stats_type == LABEL_STATS_FREQUENCY_COUNT:
                # get train labels
                train_labels = res[self.label_name][ \
                    res[self.train_mask_name].astype(np.bool_)]
                vals, counts = np.unique(train_labels, return_counts=True)
                res[LABEL_STATS_FIELD+self.label_name] = \
                    (LABEL_STATS_FREQUENCY_COUNT, vals, counts)
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
    A tuple of
        list of LabelProcessor : the label processors generated from the configurations.
    """
    label_confs = confs['labels']
    assert len(label_confs) >= 1, \
        "If a 'labels' field is defined for a node type or an edge type in the " \
        "configuration file, it should not be empty."

    def parse_label_conf(label_conf):
        assert 'task_type' in label_conf, "'task_type' must be defined in the label field."
        task_type = label_conf['task_type']
        label_stats_type = label_conf['label_stats_type'] \
            if 'label_stats_type' in label_conf else None
        label_stats_type = _check_label_stats_type(task_type, label_stats_type)

        # default mask names
        mask_field_names = ("train_mask", "val_mask", "test_mask")
        if 'mask_field_names' in label_conf:
            # User defined mask names
            assert isinstance(label_conf['mask_field_names'], list) and \
                len(label_conf['mask_field_names']) == 3, \
                "User defined mask_field_names must be a list of three strings." \
                f"But get {label_conf['mask_field_names']}"
            mask_field_names = tuple(label_conf['mask_field_names'])

        if 'custom_split_filenames' in label_conf:
            custom_split = label_conf['custom_split_filenames']
            assert isinstance(custom_split, dict), \
                    "Custom data split needs to provide train/val/test index."
            if "column" not in custom_split:
                custom_split["column"] = []
            # Treat all input as an input of list[str]
            if isinstance(custom_split['column'], str):
                custom_split["column"] = [custom_split["column"]]
            train_idx, val_idx, test_idx = read_index(custom_split)
            label_col = label_conf['label_col'] if 'label_col' in label_conf else None
            if "node_id_col" in confs:
                return CustomLabelProcessor(col_name=label_col, label_name=label_col,
                                            id_col=confs["node_id_col"],
                                            task_type=task_type,
                                            train_idx=train_idx,
                                            val_idx=val_idx,
                                            test_idx=test_idx,
                                            stats_type=label_stats_type,
                                            mask_field_names=mask_field_names)
            elif "source_id_col" in confs and "dest_id_col" in confs:
                return CustomLabelProcessor(col_name=label_col, label_name=label_col,
                                            id_col=(confs["source_id_col"],
                                                    confs["dest_id_col"]),
                                            task_type=task_type,
                                            train_idx=train_idx,
                                            val_idx=val_idx,
                                            test_idx=test_idx,
                                            stats_type=label_stats_type,
                                            mask_field_names=mask_field_names)
            else:
                raise AttributeError("Custom data segmentation should be "
                                    "applied to either node or edge tasks.")

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
            return ClassificationProcessor(label_col, label_col, split_pct,
                                            label_stats_type, mask_field_names)
        elif task_type == 'regression':
            assert 'label_col' in label_conf, \
                    "'label_col' must be defined in the label field."
            label_col = label_conf['label_col']
            return RegressionProcessor(label_col, label_col, split_pct,
                                        label_stats_type, mask_field_names)
        else:
            assert task_type == 'link_prediction', \
                    "The task type must be classification, regression or link_prediction."
            assert not is_node, "link_prediction task must be defined on edges."
            return LinkPredictionProcessor(None, None, split_pct,
                                            label_stats_type, mask_field_names)
    label_ops = []
    for label_conf in label_confs:
        label_ops.append(parse_label_conf(label_conf))

    if len(label_ops) > 1:
        # check whether train/val/test mask names are
        # different for different labels
        mask_names = []
        for ops in label_ops:
            mask_names.append(ops.train_mask_name)
            mask_names.append(ops.val_mask_name)
            mask_names.append(ops.test_mask_name)
        if len(mask_names) == len(set(mask_names)):
            # In multi-task learning, we expect each task has
            # its own train, validation and test mask fields.
            # But there can be exceptions as users want to
            # provide masks through node features or
            # some tasks are sharing the same mask.
            logging.warning("Some train/val/test mask field "
                            "names are duplicated, please check: %s."
                            "If you provide masks as node/edge features,"
                            "please ignore this warning."
                            "If you share train/val/test mask fields "
                            "across different tasks, please ignore this warning.",
                            mask_names)

    return label_ops

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
    assert len(label_processors) >= 1, \
        "Number of label_processors must be one or more."
    ret = {}
    for label_processor in label_processors:
        label_feats = label_processor(data)
        logging.debug("Label information: %s", label_feats)
        ret.update(label_feats)
    return ret

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
