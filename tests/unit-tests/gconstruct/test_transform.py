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
"""
import pytest
import inspect

import numpy as np
import torch as th
from numpy.testing import assert_equal, assert_almost_equal, assert_raises
from scipy.special import erfinv
from transformers import AutoTokenizer, AutoModel, AutoConfig

import graphstorm as gs
from graphstorm.gconstruct.transform import (
    parse_feat_ops,
    process_features,
    preprocess_features,
    update_ops
)
from graphstorm.gconstruct.transform import (_get_output_dtype,
                                             NumericalMinMaxTransform,
                                             NumericalStandardTransform,
                                             GlobalProcessFeatTransform,
                                             Noop,
                                             RankGaussTransform,
                                             CategoricalTransform,
                                             BucketTransform,
                                             HardEdgeDstNegativeTransform,
                                             Tokenizer,
                                             Text2BERT,
                                             TabularFMTransform)
from graphstorm.gconstruct.transform import (_check_label_stats_type,
                                             collect_label_stats,
                                             CustomLabelProcessor,
                                             ClassificationProcessor)
from graphstorm.gconstruct.transform import (LABEL_STATS_FIELD,
                                             LABEL_STATS_FREQUENCY_COUNT)
from graphstorm.gconstruct.id_map import IdMap


def test_get_output_dtype():
    assert _get_output_dtype("float16") == np.float16
    assert _get_output_dtype("float32") == np.float32
    assert _get_output_dtype("float64") == np.float64
    assert_raises(Exception, _get_output_dtype, "int32")

@pytest.mark.parametrize("input_dtype", [np.cfloat, np.float32, np.float16])
def test_fp_min_max_bound(input_dtype):
    feats = np.random.randn(100).astype(input_dtype)
    feats[0] = 10.
    feats[1] = -10.
    transform = NumericalMinMaxTransform("test", "test")
    max_val, min_val = transform.pre_process(feats)["test"]
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1

    # test invalid inputs
    feats[0] = np.nan
    with assert_raises(AssertionError):
        _ = transform.pre_process(feats.astype(input_dtype))

    feats[0] = np.inf
    with assert_raises(AssertionError):
        _ = transform.pre_process(feats.astype(input_dtype))

    # without predefined bound.
    feats = np.random.randn(100).astype(input_dtype)
    res_dtype = np.float32 if input_dtype == np.cfloat else input_dtype
    fifo = np.finfo(res_dtype)
    max_v = np.max(feats) + 10
    min_v= np.min(feats) - 10
    feats[0] = max_v
    feats[1] = min_v
    transform = NumericalMinMaxTransform("test", "test",
                                         out_dtype=input_dtype)
    max_val, min_val = transform.pre_process(feats)["test"]
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert_almost_equal(max_val[0], max_v, decimal=2)
    assert_almost_equal(min_val[0], min_v, decimal=2)

    # has predefined bound.
    feats = np.random.randn(100).astype(input_dtype)
    res_dtype = np.float32 if input_dtype == np.cfloat else input_dtype
    fifo = np.finfo(res_dtype)
    max_v = np.max(feats) + 10
    min_v= np.min(feats) - 10
    feats[0] = max_v
    feats[1] = min_v
    transform = NumericalMinMaxTransform("test", "test",
                                         max_bound=5,
                                         min_bound=-5,
                                         out_dtype=input_dtype)
    max_val, min_val = transform.pre_process(feats)["test"]
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert_almost_equal(max_val[0], 5, decimal=2)
    assert_almost_equal(min_val[0], -5, decimal=2)

    if input_dtype == np.float16:
        feats = np.random.randn(100).astype(input_dtype)
        fifo = np.finfo(np.float32)
        max_v = np.max(feats) + 10
        min_v= np.min(feats) - 10
        feats[0] = max_v
        feats[1] = min_v
        transform = NumericalMinMaxTransform("test", "test",
                                             out_dtype=input_dtype)
        max_val, min_val = transform.pre_process(feats)["test"]
        assert len(max_val.shape) == 1
        assert len(min_val.shape) == 1
        assert max_val[0].dtype == np.float16
        assert min_val[0].dtype == np.float16
        assert_almost_equal(max_val[0], max_v, decimal=2)
        assert_almost_equal(min_val[0], min_v, decimal=2)

        feats = np.random.randn(100).astype(input_dtype)
        fifo = np.finfo(np.float32)
        max_v = np.max(feats) + 10
        min_v= np.min(feats) - 10
        feats[0] = max_v
        feats[1] = min_v
        transform = NumericalMinMaxTransform("test", "test",
                                             max_bound=fifo.max,
                                             min_bound=-fifo.max,
                                             out_dtype=input_dtype)
        max_val, min_val = transform.pre_process(feats)["test"]
        assert transform._max_bound == np.finfo(np.float16).max
        assert transform._min_bound == -np.finfo(np.float16).max
        assert len(max_val.shape) == 1
        assert len(min_val.shape) == 1
        assert max_val[0].dtype == np.float16
        assert min_val[0].dtype == np.float16
        assert_almost_equal(max_val[0], max_v, decimal=2)
        assert_almost_equal(min_val[0], min_v, decimal=2)


@pytest.mark.parametrize("input_dtype", [np.cfloat, np.float32])
def test_fp_transform(input_dtype):
    # test NumericalMinMaxTransform pre-process
    transform = NumericalMinMaxTransform("test", "test")
    feats = np.random.randn(100).astype(input_dtype)

    max_val, min_val = transform.pre_process(feats)["test"]
    max_v = np.amax(feats).astype(np.float32)
    min_v = np.amin(feats).astype(np.float32)
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert_equal(max_val[0], max_v)
    assert_equal(min_val[0], min_v)

    feats = np.random.randn(100, 1).astype(input_dtype)
    max_val, min_val = transform.pre_process(feats)["test"]
    max_v = np.amax(feats).astype(np.float32)
    min_v = np.amin(feats).astype(np.float32)
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert_equal(max_val[0], max_v)
    assert_equal(min_val[0], min_v)

    feats = np.random.randn(100, 10).astype(input_dtype)
    max_val, min_val = transform.pre_process(feats)["test"]
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert len(max_val) == 10
    assert len(min_val) == 10
    for i in range(10):
        max_v = np.amax(feats[:,i]).astype(np.float32)
        min_v = np.amin(feats[:,i]).astype(np.float32)
        assert_equal(max_val[i], max_v)
        assert_equal(min_val[i], min_v)

    feats = np.random.randn(100).astype(input_dtype)
    feats[0] = 10.
    feats[1] = -10.
    transform = NumericalMinMaxTransform("test", "test", max_bound=5., min_bound=-5.)
    max_val, min_val = transform.pre_process(feats)["test"]
    max_v = np.amax(feats).astype(np.float32)
    min_v = np.amin(feats).astype(np.float32)
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert_equal(max_val[0], 5.)
    assert_equal(min_val[0], -5.)

    feats = np.random.randn(100, 1).astype(input_dtype)
    feats[0][0] = 10.
    feats[1][0] = -10.
    transform = NumericalMinMaxTransform("test", "test", max_bound=5., min_bound=-5.)
    max_val, min_val = transform.pre_process(feats)["test"]
    max_v = np.amax(feats).astype(np.float32)
    min_v = np.amin(feats).astype(np.float32)
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert_equal(max_val[0], 5.)
    assert_equal(min_val[0], -5.)

    feats = np.random.randn(100, 10).astype(input_dtype)
    feats[0] = 10.
    feats[1] = -10.
    transform = NumericalMinMaxTransform("test", "test", max_bound=5., min_bound=-5.)
    max_val, min_val = transform.pre_process(feats)["test"]
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert len(max_val) == 10
    assert len(min_val) == 10
    for i in range(10):
        max_v = np.amax(feats[:,i])
        min_v = np.amin(feats[:,i])
        assert_equal(max_val[i], 5.)
        assert_equal(min_val[i], -5.)

    # Test collect info
    transform_conf = {
        "name": "max_min_norm"
    }
    transform = NumericalMinMaxTransform("test", "test", transform_conf=transform_conf)
    info = [(np.array([1.]), np.array([-1.])),
            (np.array([2.]), np.array([-0.5])),
            (np.array([0.5]), np.array([-0.1]))]
    transform.update_info(info)
    assert len(transform._max_val) == 1
    assert len(transform._min_val) == 1
    assert_equal(transform._max_val[0], 2.)
    assert_equal(transform._min_val[0], -1.)
    assert 'max_val' in transform_conf
    assert 'min_val' in transform_conf
    assert_equal(np.array(transform_conf['max_val']), 2.)
    assert_equal(np.array(transform_conf['min_val']), -1.)

    info = [(np.array([1., 2., 3.]), np.array([-1., -2., 0.5])),
            (np.array([2., 1., 3.]), np.array([-0.5, -3., 0.1])),
            (np.array([0.5, 3., 1.]), np.array([-0.1, -2., 0.3]))]
    transform.update_info(info)
    assert len(transform._max_val) == 3
    assert len(transform._min_val) == 3
    assert_equal(transform._max_val[0], 2.)
    assert_equal(transform._min_val[0], -1.)
    assert 'max_val' in transform_conf
    assert 'min_val' in transform_conf
    assert_equal(np.array(transform_conf['max_val']),
                 np.array([2.,3.,3.]))
    assert_equal(np.array(transform_conf['min_val']),
                 np.array([-1.,-3.,0.1]))

    transform_conf = {
        "name": "max_min_norm",
        "max_val": [1.,1.,1.],
        "min_val": [-1.,-1.,-1.]
    }
    transform = NumericalMinMaxTransform("test", "test",
                                        max_val=transform_conf['max_val'],
                                        min_val=transform_conf['min_val'],
                                        transform_conf=transform_conf)
    feats = 2 * np.random.randn(10, 3).astype(input_dtype)
    feats[0][0] = 2
    feats[0][1] = -2
    info = transform.pre_process(feats)
    max_val = np.array(transform_conf['max_val'])
    min_val = np.array(transform_conf["min_val"])
    assert_equal(info["test"][0], max_val)
    assert_equal(info["test"][1], min_val)
    transform.update_info([info["test"]])
    assert_equal(np.array(transform_conf['max_val']),
                 np.array([1.,1.,1.]))
    assert_equal(np.array(transform_conf['min_val']),
                 np.array([-1.,-1.,-1.]))
    result = transform(feats)
    assert transform.feat_dim == (3,)
    true_result = (feats - min_val) / (max_val - min_val)
    true_result[true_result > 1] = 1
    true_result[true_result < 0] = 0
    assert_almost_equal(result["test"].astype(input_dtype), true_result)

    transform_conf = {
        "name": "max_min_norm",
        "min_val": [-1.,-1.,-1.]
    }
    transform = NumericalMinMaxTransform("test", "test",
                                        min_val=transform_conf['min_val'],
                                        transform_conf=transform_conf)
    info = transform.pre_process(feats)
    max_val = info["test"][0]
    min_val = np.array(transform_conf['min_val'])
    assert_equal(info["test"][0], max_val)
    transform.update_info([info["test"]])
    assert_equal(np.array(transform_conf['max_val']),
                 max_val)
    assert_equal(np.array(transform_conf['min_val']),
                 np.array([-1.,-1.,-1.]))
    result = transform(feats)
    assert transform.feat_dim == (3,)
    true_result = (feats - min_val) / (max_val - min_val)
    true_result[true_result > 1] = 1
    true_result[true_result < 0] = 0
    assert_almost_equal(result["test"].astype(input_dtype), true_result)

    transform_conf = {
        "name": "max_min_norm",
        "max_val": [1.,1.,1.]
    }
    transform = NumericalMinMaxTransform("test", "test",
                                        max_val=transform_conf['max_val'],
                                        transform_conf=transform_conf)
    info = transform.pre_process(feats)
    max_val = np.array(transform_conf['max_val'])
    min_val = info["test"][1]
    assert_equal(info["test"][0], max_val)
    transform.update_info([info["test"]])
    assert_equal(np.array(transform_conf['max_val']),
                 np.array([1.,1.,1.]))
    assert_equal(np.array(transform_conf['min_val']),
                 min_val)
    result = transform(feats)
    assert transform.feat_dim == (3,)
    true_result = (feats - min_val) / (max_val - min_val)
    true_result[true_result > 1] = 1
    true_result[true_result < 0] = 0
    assert_almost_equal(result["test"].astype(input_dtype), true_result)

@pytest.mark.parametrize("input_dtype", [np.cfloat, np.float32])
@pytest.mark.parametrize("out_dtype", [None, np.float16])
def test_fp_min_max_transform(input_dtype, out_dtype):
    transform = NumericalMinMaxTransform("test", "test", out_dtype=out_dtype)
    max_val = np.array([2.])
    min_val = np.array([-1.])
    transform._max_val = max_val
    transform._min_val = min_val
    feats = np.random.randn(100).astype(input_dtype)
    norm_feats = transform(feats)["test"]
    assert transform.feat_dim == (1,)
    if out_dtype is not None:
        assert norm_feats.dtype == np.float16
    else:
        assert norm_feats.dtype != np.float16
    feats[feats > max_val] = max_val
    feats[feats < min_val] = min_val
    feats = (feats-min_val)/(max_val-min_val)
    feats = feats if out_dtype is None else feats.astype(out_dtype)
    assert_almost_equal(norm_feats, feats, decimal=5)

    feats = np.random.randn(100, 1).astype(input_dtype)
    norm_feats = transform(feats)["test"]
    assert transform.feat_dim == (1,)
    if out_dtype is not None:
        assert norm_feats.dtype == np.float16
    else:
        assert norm_feats.dtype != np.float16
    feats[feats > max_val] = max_val
    feats[feats < min_val] = min_val
    feats = (feats-min_val)/(max_val-min_val)
    feats = feats if out_dtype is None else feats.astype(out_dtype)
    assert_almost_equal(norm_feats, feats, decimal=5)

    transform = NumericalMinMaxTransform("test", "test", out_dtype=out_dtype)
    max_val = np.array([2., 3., 0.])
    min_val = np.array([-1., 1., -0.5])
    transform._max_val = max_val
    transform._min_val = min_val
    feats = np.random.randn(10, 3).astype(input_dtype)
    norm_feats = transform(feats)["test"]
    assert transform.feat_dim == (3,)
    if out_dtype is not None:
        assert norm_feats.dtype == np.float16
    else:
        assert norm_feats.dtype != np.float16
    for i in range(3):
        new_feats = feats[:,i]
        new_feats[new_feats > max_val[i]] = max_val[i]
        new_feats[new_feats < min_val[i]] = min_val[i]
        new_feats = (new_feats-min_val[i])/(max_val[i]-min_val[i])
        new_feats = new_feats if out_dtype is None else new_feats.astype(out_dtype)
        assert_almost_equal(norm_feats[:,i], new_feats, decimal=5)


def test_categorize_transform():
    # Test a single categorical value.
    transform_conf = {
        "name": "to_categorical"
    }
    transform = CategoricalTransform("test1", "test", transform_conf=transform_conf)
    str_ids = [str(i) for i in np.random.randint(0, 10, 1000)]
    str_ids[0] = None
    str_ids[-1] = None # allow None data
    str_ids = str_ids + [str(i) for i in range(10)]
    res = transform.pre_process(np.array(str_ids))
    assert "test" in res
    assert len(res["test"]) == 10
    for i in range(10):
        assert str(i) in res["test"]

    info = [ np.array([str(i) for i in range(6)]),
            np.array([str(i) for i in range(4, 10)]) ]
    transform.update_info(info)
    feat = np.array([str(i) for i in np.random.randint(0, 10, 100)])
    cat_feat = transform(feat)
    assert "test" in cat_feat
    for feat, str_i in zip(cat_feat["test"], feat):
        # make sure one value is 1
        assert feat[int(str_i)] == 1
        # after we set the value to 0, the entire vector has 0 values.
        feat[int(str_i)] = 0
        assert np.all(feat == 0)
    assert "mapping" in transform_conf
    assert len(transform_conf["mapping"]) == 10
    feat = np.array([None, None]) # transform numpy array with None value.
    cat_feat = transform(feat)
    assert transform.feat_dim == (len(transform_conf["mapping"]),)
    assert "test" in cat_feat
    assert np.all(cat_feat["test"][0] == 0)
    assert np.all(cat_feat["test"][1] == 0)

    feat = [str(i) for i in np.random.randint(0, 10, 100)]
    feat_with_unknown = feat[:]
    feat_with_unknown[0] = "10"
    feat = np.array(feat)
    feat_with_unknown = np.array(feat_with_unknown)
    cat_feat = transform(feat)
    assert transform.feat_dim == (len(transform_conf["mapping"]),)
    assert "test" in cat_feat
    for i, (feat, str_i) in enumerate(zip(cat_feat["test"], feat)):
        if i == 0:
            continue
        # make sure one value is 1
        assert feat[int(str_i)] == 1
        # after we set the value to 0, the entire vector has 0 values.
        feat[int(str_i)] = 0
        assert np.all(feat == 0)

    # Test categorical values with empty strings.
    transform = CategoricalTransform("test1", "test", separator=',')
    str_ids = [f"{i},{i+1}" for i in np.random.randint(0, 9, 1000)] + [",0"]
    str_ids = str_ids + [str(i) for i in range(9)]
    res = transform.pre_process(np.array(str_ids))
    assert "test" in res
    assert len(res["test"]) == 11

    # Test multiple categorical values.
    transform = CategoricalTransform("test1", "test", separator=',')
    str_ids = [f"{i},{i+1}" for i in np.random.randint(0, 9, 1000)]
    str_ids = str_ids + [str(i) for i in range(9)]
    res = transform.pre_process(np.array(str_ids))
    assert "test" in res
    assert len(res["test"]) == 10
    for i in range(10):
        assert str(i) in res["test"]

    info = [ np.array([str(i) for i in range(6)]),
            np.array([str(i) for i in range(4, 10)]) ]
    transform.update_info(info)
    feat = np.array([f"{i},{i+1}" for i in np.random.randint(0, 9, 100)])
    cat_feat = transform(feat)
    assert transform.feat_dim == (len(transform_conf["mapping"]),)
    assert "test" in cat_feat
    for feat, str_feat in zip(cat_feat["test"], feat):
        # make sure two elements are 1
        i = str_feat.split(",")
        assert feat[int(i[0])] == 1
        assert feat[int(i[1])] == 1
        # after removing the elements, the vector has only 0 values.
        feat[int(i[0])] = 0
        feat[int(i[1])] = 0
        assert np.all(feat == 0)

    # feat contains unknown keys
    feat = [f"{i},{i+1}" for i in np.random.randint(0, 9, 100)]
    feat_with_unknown = feat[:]
    feat_with_unknown[0] = feat[0]+",10"
    feat = np.array(feat)
    feat_with_unknown = np.array(feat_with_unknown)
    cat_feat = transform(feat_with_unknown)
    assert transform.feat_dim == (len(transform_conf["mapping"]),)
    assert "test" in cat_feat
    for feat, str_feat in zip(cat_feat["test"], feat):
        # make sure two elements are 1
        i = str_feat.split(",")
        assert feat[int(i[0])] == 1
        assert feat[int(i[1])] == 1
        # after removing the elements, the vector has only 0 values.
        feat[int(i[0])] = 0
        feat[int(i[1])] = 0
        assert np.all(feat == 0)

    # Test transformation with existing mapping.
    transform = CategoricalTransform("test1", "test", transform_conf=transform_conf)
    str_ids = [str(i) for i in np.random.randint(0, 10, 1000)]
    str_ids = str_ids + [str(i) for i in range(10)]
    res = transform.pre_process(np.array(str_ids))
    assert len(res) == 0

    transform.update_info([])
    feat = np.array([str(i) for i in np.random.randint(0, 10, 100)])
    cat_feat = transform(feat)
    assert transform.feat_dim == (len(transform_conf["mapping"]),)
    assert "test" in cat_feat
    for feat, str_i in zip(cat_feat["test"], feat):
        # make sure one value is 1
        idx = transform_conf["mapping"][str_i]
        assert feat[idx] == 1
        # after we set the value to 0, the entire vector has 0 values.
        feat[idx] = 0
        assert np.all(feat == 0)

    # Test the case when the input ids are mixture of int and strings
    # This may happen when loading csv files.
    transform_conf = {
        "name": "to_categorical"
    }
    transform = CategoricalTransform("test1", "test", transform_conf=transform_conf)
    str_ids = [str(i) for i in np.random.randint(0, 10, 1000)]
    str_ids[0] = None
    str_ids[-1] = None # allow None data
    str_ids = str_ids + [i for i in range(10)] # mix strings with ints
    res = transform.pre_process(np.array(str_ids))
    assert "test" in res
    assert len(res["test"]) == 10
    for i in range(10):
        assert str(i) in res["test"]

    info = [np.array([str(i) for i in range(6)]),
            np.array([str(i) for i in range(4, 10)])]
    transform.update_info(info)
    feat = np.array([str(i) for i in np.random.randint(0, 10, 100)])
    feat[0] = int(feat[0])
    cat_feat = transform(feat)
    assert transform.feat_dim == (len(transform_conf["mapping"]),)
    assert "test" in cat_feat
    for feat, str_i in zip(cat_feat["test"], feat):
        # make sure one value is 1
        assert feat[int(str_i)] == 1
        # after we set the value to 0, the entire vector has 0 values.
        feat[int(str_i)] = 0
        assert np.all(feat == 0)
    assert "mapping" in transform_conf
    assert len(transform_conf["mapping"]) == 10

    # check update_info
    transform_conf = {
        "name": "to_categorical"
    }
    transform = CategoricalTransform("test1", "test", transform_conf=transform_conf)
    info = [np.array([i for i in range(6)]),
            np.array([i for i in range(4, 10)])]
    transform.update_info(info)
    cat_feat = transform(feat)
    assert transform.feat_dim == (len(transform_conf["mapping"]),)
    assert "test" in cat_feat
    for feat, str_i in zip(cat_feat["test"], feat):
        # make sure one value is 1
        assert feat[int(str_i)] == 1
        # after we set the value to 0, the entire vector has 0 values.
        feat[int(str_i)] = 0
        assert np.all(feat == 0)
    assert len(transform_conf["mapping"]) == 10

    # check backward compatible
     # check update_info
    transform_conf = {
        "name": "to_categorical",
        "mapping": {i: i for i in range(10)}
    }
    transform = CategoricalTransform("test1", "test", transform_conf=transform_conf)
    assert len(transform_conf["mapping"]) == 10
    cat_feat = transform(feat)
    assert transform.feat_dim == (len(transform_conf["mapping"]),)
    assert "test" in cat_feat
    for feat, str_i in zip(cat_feat["test"], feat):
        # make sure one value is 1
        assert feat[int(str_i)] == 1
        # after we set the value to 0, the entire vector has 0 values.
        feat[int(str_i)] = 0
        assert np.all(feat == 0)

@pytest.mark.parametrize("out_dtype", [None, np.float16, np.float64])
def test_noop_transform(out_dtype):
    transform = Noop("test", "test", out_dtype=out_dtype)
    feats = np.random.randn(100).astype(np.float32)
    norm_feats = transform(feats)
    if out_dtype is not None:
        assert norm_feats["test"].dtype == out_dtype
    elif out_dtype == np.float64:
        assert norm_feats["test"].dtype == np.float64
    else:
        assert norm_feats["test"].dtype == np.float32

    assert transform.feat_dim == (1,)
    # invalid input
    feats[0] = np.nan
    with assert_raises(AssertionError):
        _ = transform(feats)

    feats[0] = np.inf
    with assert_raises(AssertionError):
        _ = transform(feats)

def test_noop_truncate():
    transform = Noop("test", "test", truncate_dim=16)
    feats = np.random.randn(100, 32).astype(np.float32)
    trunc_feats = transform(feats)

    assert trunc_feats["test"].shape[1] == 16

def test_noop_str_vector():
    """Test conversion of delimited string array to vectors"""

    transform = Noop("test", "test", separator=";")
    # Create a numpy array with ;-delimited vector strings
    feats = np.array(["1;2;3", "4;5;6", "7;8;9"])
    vector_feats = transform(feats)

    assert transform.feat_dim == (3,)
    expected_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    assert_equal(vector_feats["test"], expected_array)

@pytest.mark.parametrize("input_dtype", [np.cfloat, np.float32])
@pytest.mark.parametrize("out_dtype", [None, np.float16])
def test_rank_gauss_transform(input_dtype, out_dtype):
    eps = 1e-6
    transform = RankGaussTransform("test", "test", out_dtype=out_dtype, epsilon=eps)
    feat_0 = np.random.randn(100,2).astype(input_dtype)
    feat_trans_0 = transform(feat_0)['test']
    assert transform.feat_dim == (2,)
    feat_1 = np.random.randn(100,2).astype(input_dtype)
    feat_trans_1 = transform(feat_1)['test']
    assert transform.feat_dim == (2,)
    assert feat_trans_0.dtype == np.float32
    assert feat_trans_1.dtype == np.float32
    def rank_gauss(feat):
        lower = -1 + eps
        upper = 1 - eps
        range = upper - lower
        i = np.argsort(feat, axis=0)
        j = np.argsort(i, axis=0)
        j_range = len(j) - 1
        divider = j_range / range
        feat = j / divider
        feat = feat - upper
        return erfinv(feat)

    feat = np.concatenate([feat_0, feat_1])
    feat = rank_gauss(feat)
    new_feat = np.concatenate([feat_trans_0, feat_trans_1])
    trans_feat = transform.after_merge_transform(new_feat)

    if out_dtype is not None:
        assert trans_feat.dtype == np.float16
        assert_almost_equal(feat.astype(np.float16), trans_feat, decimal=3)
    else:
        assert trans_feat.dtype != np.float16
        assert_almost_equal(feat, trans_feat, decimal=4)

    feat_0 = np.random.rand(100,2).astype(input_dtype)
    feat_trans_0 = transform(feat_0)['test']
    feat_1 = np.ones((100,2)).astype(input_dtype)
    feat_trans_1 = transform(feat_1)['test']
    assert feat_trans_0.dtype == np.float32
    assert feat_trans_1.dtype == np.float32
    transform = RankGaussTransform("test", "test", out_dtype=out_dtype, epsilon=eps, uniquify=True)

    def rank_gauss(feat):
        lower = -1 + eps
        upper = 1 - eps
        range = upper - lower
        i = np.argsort(feat, axis=0)
        j = np.argsort(i, axis=0)
        j_range = len(j) - 1
        divider = j_range / range
        feat = j / divider
        feat = feat - upper
        return erfinv(feat)

    feat = np.concatenate([feat_0, feat_1])
    feat = rank_gauss(feat[:101])
    feat = feat[np.concatenate([np.arange(100), np.array([100]*100)])]
    new_feat = np.concatenate([feat_trans_0, feat_trans_1])
    trans_feat = transform.after_merge_transform(new_feat)

    if out_dtype is not None:
        assert trans_feat.dtype == np.float16
        assert_almost_equal(feat.astype(np.float16), trans_feat, decimal=3)
    else:
        assert trans_feat.dtype != np.float16
        assert_almost_equal(feat, trans_feat, decimal=4)

    # inpuu is invalid
    feat_0[0] = np.nan
    with assert_raises(AssertionError):
        _ = transform(feat_0)

    feat_0[0] = np.inf
    with assert_raises(AssertionError):
        _ = transform(feat_0)

def test_custom_node_label_processor():
    train_idx = np.arange(0, 10)
    val_idx = np.arange(10, 15)
    test_idx = np.arange(15, 20)
    clp = CustomLabelProcessor(col_name="test_label", label_name="test", id_col="id",
                               task_type="classification",
                               train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, stats_type=None)

    split = clp.data_split(np.arange(20))
    assert "train_mask" in split
    assert "val_mask" in split
    assert "test_mask" in split
    assert_equal(np.squeeze(np.nonzero(split["train_mask"])), train_idx)
    assert_equal(np.squeeze(np.nonzero(split["val_mask"])), val_idx)
    assert_equal(np.squeeze(np.nonzero(split["test_mask"])), test_idx)

    split = clp.data_split(np.arange(24))
    assert "train_mask" in split
    assert "val_mask" in split
    assert "test_mask" in split
    assert_equal(np.squeeze(np.nonzero(split["train_mask"])), train_idx)
    assert_equal(np.squeeze(np.nonzero(split["val_mask"])), val_idx)
    assert_equal(np.squeeze(np.nonzero(split["test_mask"])), test_idx)
    assert len(split["train_mask"]) == 24
    assert len(split["val_mask"]) == 24
    assert len(split["test_mask"]) == 24

    # Test with customized mask names.
    try:
        clp = CustomLabelProcessor(col_name="test_label", label_name="test", id_col="id",
                               task_type="classification",
                               train_idx=train_idx,
                               val_idx=val_idx,
                               test_idx=test_idx,
                               stats_type=None,
                               mask_field_names="train_mask")
        assert False, \
                "Should raise an exception as mask_field_names is in the wrong format."
    except:
        pass
    try:
        clp = CustomLabelProcessor(col_name="test_label", label_name="test", id_col="id",
                               task_type="classification",
                               train_idx=train_idx,
                               val_idx=val_idx,
                               test_idx=test_idx,
                               stats_type=None,
                               mask_field_names=("tm", "vm"))
        assert False, \
                "Should raise an exception as mask_field_names is in the wrong format."
    except:
        pass
    clp = CustomLabelProcessor(col_name="test_label", label_name="test", id_col="id",
                            task_type="classification",
                            train_idx=train_idx,
                            val_idx=val_idx,
                            test_idx=test_idx,
                            stats_type=None,
                            mask_field_names=("tm", "vm", "tsm"))
    split = clp.data_split(np.arange(20))
    assert "tm" in split
    assert "vm" in split
    assert "tsm" in split
    assert_equal(np.squeeze(np.nonzero(split["tm"])), train_idx)
    assert_equal(np.squeeze(np.nonzero(split["vm"])), val_idx)
    assert_equal(np.squeeze(np.nonzero(split["tsm"])), test_idx)

    # there is no label
    clp = CustomLabelProcessor(col_name="test_label", label_name="test", id_col="id",
                            task_type="classification",
                            train_idx=train_idx,
                            val_idx=val_idx,
                            test_idx=test_idx,
                            stats_type=None)
    input_data = {
        "feat": np.random.rand(24),
        "id": np.arange(24),
    }
    ret = clp(input_data)
    assert "train_mask" in ret
    assert "val_mask" in ret
    assert "test_mask" in ret
    assert_equal(np.squeeze(np.nonzero(ret["train_mask"])), train_idx)
    assert_equal(np.squeeze(np.nonzero(ret["val_mask"])), val_idx)
    assert_equal(np.squeeze(np.nonzero(ret["test_mask"])), test_idx)

    # there are labels, but not classification
    input_data = {
        "test_label": np.random.randint(0, 5, (24,)),
        "id": np.arange(24),
    }
    ret = clp(input_data)
    assert "train_mask" in ret
    assert "val_mask" in ret
    assert "test_mask" in ret
    assert_equal(np.squeeze(np.nonzero(ret["train_mask"])), train_idx)
    assert_equal(np.squeeze(np.nonzero(ret["val_mask"])), val_idx)
    assert_equal(np.squeeze(np.nonzero(ret["test_mask"])), test_idx)
    assert_equal(ret["test"], input_data["test_label"])

    # there labels and _stats_type is frequency count
    input_data = {
        "test_label": np.random.randint(0, 5, (24,)),
        "id": np.arange(24),
    }
    clp = CustomLabelProcessor(col_name="test_label", label_name="test", id_col="id",
                               task_type="classification",
                               train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                               stats_type=LABEL_STATS_FREQUENCY_COUNT)
    ret = clp(input_data)
    assert "train_mask" in ret
    assert "val_mask" in ret
    assert "test_mask" in ret
    assert_equal(np.squeeze(np.nonzero(ret["train_mask"])), train_idx)
    assert_equal(np.squeeze(np.nonzero(ret["val_mask"])), val_idx)
    assert_equal(np.squeeze(np.nonzero(ret["test_mask"])), test_idx)
    assert_equal(ret["test"], input_data["test_label"])
    stats_info_key = LABEL_STATS_FIELD+"test"
    assert LABEL_STATS_FIELD+"test" in ret
    vals, counts = np.unique(input_data["test_label"][train_idx], return_counts=True)
    assert ret[stats_info_key][0] == LABEL_STATS_FREQUENCY_COUNT
    assert_equal(ret[stats_info_key][1], vals)
    assert_equal(ret[stats_info_key][2], counts)

    # Test with customized mask names, label status still works
    clp = CustomLabelProcessor(col_name="test_label", label_name="test", id_col="id",
                               task_type="classification",
                               train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                               stats_type=LABEL_STATS_FREQUENCY_COUNT,
                               mask_field_names=("tm", "vm", "tsm"))
    ret = clp(input_data)
    assert "tm" in ret
    assert "vm" in ret
    assert "tsm" in ret
    assert_equal(np.squeeze(np.nonzero(ret["tm"])), train_idx)
    assert_equal(np.squeeze(np.nonzero(ret["vm"])), val_idx)
    assert_equal(np.squeeze(np.nonzero(ret["tsm"])), test_idx)
    assert_equal(ret["test"], input_data["test_label"])
    stats_info_key = LABEL_STATS_FIELD+"test"
    assert LABEL_STATS_FIELD+"test" in ret
    vals, counts = np.unique(input_data["test_label"][train_idx], return_counts=True)
    assert ret[stats_info_key][0] == LABEL_STATS_FREQUENCY_COUNT
    assert_equal(ret[stats_info_key][1], vals)
    assert_equal(ret[stats_info_key][2], counts)

def test_custom_edge_label_processor():
    # test generating labels on link prediction
    train_idx = tuple((i, j) for i in range(1, 10) for j in range(1, 10))
    val_idx = tuple((i, j) for i in range(11, 14) for j in range(11, 14))
    test_idx = tuple((i, j) for i in range(15, 20) for j in range(15, 20))

    data = tuple((i, j) for i in range(1, 20) for j in range(1, 20))
    index_in_train_idx = np.array([np.where(np.all(np.array(data) == t, axis=1))[0][0] for t in train_idx])
    index_in_val_idx = np.array([np.where(np.all(np.array(data) == t, axis=1))[0][0] for t in val_idx])
    index_in_test_idx = np.array([np.where(np.all(np.array(data) == t, axis=1))[0][0] for t in test_idx])

    clp = CustomLabelProcessor(col_name="test_label", label_name="test", id_col="id",
                               task_type="link_prediction",
                               train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                               stats_type=None)

    split = clp.data_split(data)
    assert "train_mask" in split
    assert "val_mask" in split
    assert "test_mask" in split

    assert_equal(np.squeeze(np.nonzero(split["train_mask"])), index_in_train_idx)
    assert_equal(np.squeeze(np.nonzero(split["val_mask"])), index_in_val_idx)
    assert_equal(np.squeeze(np.nonzero(split["test_mask"])), index_in_test_idx)
    # the total mask length should be 19 * 19
    assert len(split["train_mask"]) == 361
    assert len(split["val_mask"]) == 361
    assert len(split["test_mask"]) == 361

    # Test with customized mask names.
    clp = CustomLabelProcessor(col_name="test_label", label_name="test", id_col="id",
                                task_type="link_prediction",
                                train_idx=train_idx,
                                val_idx=val_idx,
                                test_idx=test_idx,
                                stats_type=None,
                                mask_field_names=("tm", "vm", "tsm"))
    split = clp.data_split(data)
    assert "tm" in split
    assert "vm" in split
    assert "tsm" in split
    assert_equal(np.squeeze(np.nonzero(split["tm"])), index_in_train_idx)
    assert_equal(np.squeeze(np.nonzero(split["vm"])), index_in_val_idx)
    assert_equal(np.squeeze(np.nonzero(split["tsm"])), index_in_test_idx)
    # the total mask length should be 19 * 19
    assert len(split["tm"]) == 361
    assert len(split["vm"]) == 361
    assert len(split["tsm"]) == 361


    # test generating labels on classification
    # there labels and _stats_type is frequency count
    clp = CustomLabelProcessor(col_name="test_label", label_name="test", id_col="id",
                               task_type="link_prediction",
                               train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                               stats_type=None)
    input_data = {
        "test_label": np.random.randint(0, 5, (361,)),
        "id": data,
    }
    clp = CustomLabelProcessor(col_name="test_label", label_name="test", id_col="id",
                               task_type="classification",
                               train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                               stats_type=LABEL_STATS_FREQUENCY_COUNT)
    ret = clp(input_data)
    assert "train_mask" in ret
    assert "val_mask" in ret
    assert "test_mask" in ret

    assert_equal(np.squeeze(np.nonzero(ret["train_mask"])), index_in_train_idx)
    assert_equal(np.squeeze(np.nonzero(ret["val_mask"])), index_in_val_idx)
    assert_equal(np.squeeze(np.nonzero(ret["test_mask"])), index_in_test_idx)
    assert_equal(ret["test"], input_data["test_label"])

    stats_info_key = LABEL_STATS_FIELD + "test"
    assert LABEL_STATS_FIELD + "test" in ret
    vals, counts = np.unique(input_data["test_label"][index_in_train_idx], return_counts=True)
    assert ret[stats_info_key][0] == LABEL_STATS_FREQUENCY_COUNT
    assert_equal(ret[stats_info_key][1], vals)
    assert_equal(ret[stats_info_key][2], counts)

def test_check_label_stats_type():
    stats_type = _check_label_stats_type("regression", LABEL_STATS_FREQUENCY_COUNT)
    assert stats_type is None

    stats_type = _check_label_stats_type("classification", LABEL_STATS_FREQUENCY_COUNT)
    assert stats_type == LABEL_STATS_FREQUENCY_COUNT

    with pytest.raises(Exception):
        stats_type = _check_label_stats_type("classification", "unknown")

def test_collect_label_stats():
    feat_name = LABEL_STATS_FIELD+"test"
    label_stats = [(LABEL_STATS_FREQUENCY_COUNT, np.array([0,1,2,3]), np.array([1,3,5,7]))]
    label_name, stats_type, info = collect_label_stats(feat_name, label_stats)
    assert label_name == "test"
    assert stats_type == LABEL_STATS_FREQUENCY_COUNT
    assert info[0] == 1
    assert info[1] == 3
    assert info[2] == 5
    assert info[3] == 7

    label_stats = [(LABEL_STATS_FREQUENCY_COUNT, np.array([0,2]), np.array([3,4])),
                   (LABEL_STATS_FREQUENCY_COUNT, np.array([0,1,2,3]), np.array([1,3,5,7]))]
    label_name, stats_type, info = collect_label_stats(feat_name, label_stats)
    assert label_name == "test"
    assert stats_type == LABEL_STATS_FREQUENCY_COUNT
    assert info[0] == 4
    assert info[1] == 3
    assert info[2] == 9
    assert info[3] == 7

    with pytest.raises(Exception):
        label_stats = [("unknown", np.array[0,1,2,3], np.array[1,3,5,7])]
        label_name, stats_type, info = collect_label_stats(feat_name, label_stats)

def test_classification_processor():
    clp = ClassificationProcessor("test_label", "test", [0.8,0.1,0.1], LABEL_STATS_FREQUENCY_COUNT)

    # there is no label
    input_data = {
        "test_label": np.random.randint(0, 5, (24,))
    }
    ret = clp(input_data)
    stats_info_key = LABEL_STATS_FIELD+"test"
    assert "test" in ret
    assert "train_mask" in ret
    assert "val_mask" in ret
    assert "test_mask" in ret
    assert stats_info_key in ret
    vals, counts = np.unique(input_data["test_label"][ret["train_mask"].astype(np.bool_)],
                             return_counts=True)
    assert ret[stats_info_key][0] == LABEL_STATS_FREQUENCY_COUNT
    assert_equal(ret[stats_info_key][1], vals)
    assert_equal(ret[stats_info_key][2], counts)

    # Test with customized mask name.
    try:
        clp = ClassificationProcessor("test_label", "test", [0.8,0.1,0.1],
                                      LABEL_STATS_FREQUENCY_COUNT,
                                      mask_field_names="train_mask")
        assert False, \
            "Should raise an exception as mask_field_names is in the wrong format."
    except:
        pass

    try:
        clp = ClassificationProcessor("test_label", "test", [0.8,0.1,0.1],
                                      LABEL_STATS_FREQUENCY_COUNT,
                                      mask_field_names=("tm", "vm"))
        assert False, \
            "Should raise an exception as mask_field_names is in the wrong format."
    except:
        pass

    clp = ClassificationProcessor("test_label", "test", [0.8,0.1,0.1],
                                      LABEL_STATS_FREQUENCY_COUNT,
                                      mask_field_names=("tm", "vm", "tsm"))
    ret = clp(input_data)
    assert "test" in ret
    assert "tm" in ret
    assert "vm" in ret
    assert "tsm" in ret
    assert stats_info_key in ret
    vals, counts = np.unique(input_data["test_label"][ret["tm"].astype(np.bool_)],
                             return_counts=True)
    assert_equal(ret[stats_info_key][1], vals)
    assert_equal(ret[stats_info_key][2], counts)

@pytest.mark.parametrize("out_dtype", [None, np.float16])
def test_bucket_transform(out_dtype):
    bucket_range = [10, 30]
    transform = BucketTransform("test", "test", 2,
                 bucket_range=bucket_range, slide_window_size=0, out_dtype=out_dtype)
    feats = np.array([1, 11, 21, 31])
    bucket_feats = transform(feats)
    assert transform.feat_dim == (2,)
    if out_dtype is not None:
        assert bucket_feats['test'].dtype == np.float16
    else:
        assert bucket_feats['test'].dtype == np.float32

    feats_tar = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=out_dtype)
    assert_equal(bucket_feats['test'], feats_tar)

    # Case for float number
    bucket_range = [1.1, 3.1]
    feats = np.array([0.2, 1.2, 2.2, 3.2])
    transform = BucketTransform("test", "test", 2,
                 bucket_range=bucket_range, slide_window_size=0, out_dtype=out_dtype)
    bucket_feats = transform(feats)
    assert transform.feat_dim == (2,)
    if out_dtype is not None:
        assert bucket_feats['test'].dtype == np.float16
    else:
        assert bucket_feats['test'].dtype == np.float32

    feats_tar = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=out_dtype)
    assert_equal(bucket_feats['test'], feats_tar)

    # Case with sliding window
    bucket_range = [10, 30]
    transform = BucketTransform("test", "test", 2,
                 bucket_range=bucket_range, slide_window_size=10, out_dtype=out_dtype)
    feats = np.array([1, 11, 21, 31])
    bucket_feats = transform(feats)
    assert transform.feat_dim == (2,)
    if out_dtype is not None:
        assert bucket_feats['test'].dtype == np.float16
    else:
        assert bucket_feats['test'].dtype == np.float32

    feats_tar = np.array([[1, 0], [1, 0], [1, 1], [0, 1]], dtype=out_dtype)
    assert_equal(bucket_feats['test'], feats_tar)

    # Edge case for data on the bucket edge
    bucket_range = [10, 30]
    transform = BucketTransform("test", "test", 2,
                                bucket_range=bucket_range, out_dtype=out_dtype)
    feats = np.array([1, 10, 20, 30])
    bucket_feats = transform(feats)
    assert transform.feat_dim == (2,)
    if out_dtype is not None:
        assert bucket_feats['test'].dtype == np.float16
    else:
        assert bucket_feats['test'].dtype == np.float32

    feats_tar = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=out_dtype)
    assert_equal(bucket_feats['test'], feats_tar)

    # Edge case for large sliding window
    bucket_range = [10, 30]
    transform = BucketTransform("test", "test", 3,
                                bucket_range=bucket_range, slide_window_size=20,
                                out_dtype=out_dtype)
    feats = np.array([1, 10, 20, 30])
    bucket_feats = transform(feats)
    assert transform.feat_dim == (3,)
    if out_dtype is not None:
        assert bucket_feats['test'].dtype == np.float16
    else:
        assert bucket_feats['test'].dtype == np.float32

    feats_tar = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=out_dtype)
    assert_equal(bucket_feats['test'], feats_tar)

    # Edge case for super large sliding window
    bucket_range = [10, 30]
    transform = BucketTransform("test", "test", 2,
                                bucket_range=bucket_range, slide_window_size=100,
                                out_dtype=out_dtype)
    feats = np.array([1, 10, 20, 30])
    bucket_feats = transform(feats)
    assert transform.feat_dim == (2,)
    if out_dtype is not None:
        assert bucket_feats['test'].dtype == np.float16
    else:
        assert bucket_feats['test'].dtype == np.float32

    feats_tar = np.array([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=out_dtype)
    assert_equal(bucket_feats['test'], feats_tar)

    # invalid input
    feats = np.array([1, 10, 20, np.nan])
    with assert_raises(AssertionError):
        bucket_feats = transform(feats)

    feats = np.array([1, 10, 20, np.inf])
    with assert_raises(AssertionError):
        bucket_feats = transform(feats)

@pytest.mark.parametrize("out_dtype", [None, np.float16])
def test_tabular_fm_transform(out_dtype):
    # Create sample tabular data with more columns but fewer rows
    np.random.seed(42)
    sample_data = {
        'feature1': np.random.randn(20),
        'feature2': np.random.randn(20),
        'feature3': np.random.randint(0, 5, 20),
        'feature4': np.random.uniform(0, 1, 20),
        'feature5': np.random.exponential(2, 20),
        'feature6': np.random.randint(10, 100, 20),
        'feature7': np.random.normal(5, 2, 20),
        'feature8': np.random.beta(2, 3, 20),
        'feature9': np.random.poisson(3, 20),
        'feature10': np.random.gamma(2, 2, 20)
    }
    
    # Create TabularFMTransform instance
    transform = TabularFMTransform("test_col", "test_feat", 
                                   out_dtype=out_dtype, target_col='feature3')
    
    # Test the transform
    result = transform(sample_data)
    
    # Verify the result structure
    assert "test_feat" in result
    assert isinstance(result["test_feat"], np.ndarray)
    
    # Check output dtype
    if out_dtype is not None:
        assert result["test_feat"].dtype == out_dtype
    else:
        assert result["test_feat"].dtype == np.float32
    
    # Verify feature dimension is set
    assert hasattr(transform, 'feat_dim')
    assert isinstance(transform.feat_dim, tuple)


@pytest.mark.parametrize("id_dtype", [str, np.int64])
def test_hard_edge_dst_negative_transform(id_dtype):
    hard_neg_trasnform = HardEdgeDstNegativeTransform("hard_neg", "hard_neg")
    assert hard_neg_trasnform.col_name == "hard_neg"
    assert hard_neg_trasnform.feat_name == "hard_neg"
    assert hard_neg_trasnform.out_dtype == np.int64

    raw_ids = np.array([(99-i) for i in range(100)])
    id_maps = {"src": IdMap(raw_ids.astype(id_dtype))}
    pass_set_id_maps = False
    try:
        # set_id_maps will fail if target_ntype is None
        assert hard_neg_trasnform._target_ntype is None
        hard_neg_trasnform.set_id_maps(id_maps)
    except:
        pass_set_id_maps = True
    assert pass_set_id_maps

    hard_neg_trasnform.set_target_etype(("src", "rel", "dst"))
    assert hard_neg_trasnform.neg_ntype == "dst"
    try:
        # set_id_maps will fail as target_ntype is dst
        # but only src has id mapping.
        hard_neg_trasnform.set_id_maps(id_maps)
    except:
        pass_set_id_maps = True
    assert pass_set_id_maps

    id_maps = {"dst": IdMap(raw_ids.astype(id_dtype))}
    hard_neg_trasnform.set_id_maps(id_maps)

    input_feats0 = np.random.randint(0, 100, size=(20, 10), dtype=np.int64)
    input_id_feats0 = input_feats0.astype(id_dtype)
    info0 = hard_neg_trasnform.pre_process(input_id_feats0)
    assert info0["hard_neg"] == 10

    input_feats1 = np.random.randint(0, 100, size=(20, 20), dtype=np.int64)
    input_id_feats1 = input_feats1.astype(id_dtype)
    info1 = hard_neg_trasnform.pre_process(input_id_feats1)
    assert info1["hard_neg"] == 20

    hard_neg_trasnform.update_info([info0["hard_neg"], info1["hard_neg"]])
    assert hard_neg_trasnform._max_dim == 20

    neg0 = hard_neg_trasnform(input_id_feats0)
    assert hard_neg_trasnform.feat_dim == (hard_neg_trasnform._max_dim,)
    assert_equal(neg0["hard_neg"][:,:10], 99-input_feats0)
    assert_equal(neg0["hard_neg"][:,10:], np.full((20, 10), -1, dtype=np.int64))
    neg1 = hard_neg_trasnform(input_id_feats1)
    assert hard_neg_trasnform.feat_dim == (hard_neg_trasnform._max_dim,)
    assert_equal(neg1["hard_neg"], 99-input_feats1)

    hard_neg_trasnform = HardEdgeDstNegativeTransform("hard_neg", "hard_neg", separator=",")
    hard_neg_trasnform.set_target_etype(("src", "rel", "dst"))
    hard_neg_trasnform.set_id_maps(id_maps)

    input_feats0 = np.random.randint(0, 100, size=(20, 10), dtype=np.int64)
    input_id_feats0 = [",".join(feats) for feats in input_feats0.astype(str).tolist()]
    input_id_feats0.append(",".join([str(i) for i in range(15)]))
    input_id_feats0 = np.array(input_id_feats0)
    info0 = hard_neg_trasnform.pre_process(input_id_feats0)
    assert info0["hard_neg"] == 15

    input_feats1 = np.random.randint(0, 100, size=(20, 20), dtype=np.int64)
    input_id_feats1 = [",".join(feats) for feats in input_feats1.astype(str).tolist()]
    input_id_feats1.append(",".join([str(i) for i in range(15)]))
    input_id_feats1 = np.array(input_id_feats1)
    info1 = hard_neg_trasnform.pre_process(input_id_feats1)
    assert info1["hard_neg"] == 20

    hard_neg_trasnform.update_info([info0["hard_neg"], info1["hard_neg"]])
    assert hard_neg_trasnform._max_dim == 20

    neg0 = hard_neg_trasnform(input_id_feats0)
    assert hard_neg_trasnform.feat_dim == (hard_neg_trasnform._max_dim,)
    assert_equal(neg0["hard_neg"][:20,:10], 99-input_feats0)
    assert_equal(neg0["hard_neg"][:20,10:], np.full((20, 10), -1, dtype=np.int64))
    assert_equal(neg0["hard_neg"][20][:15], np.array([(99-i) for i in range(15)]))
    assert_equal(neg0["hard_neg"][20][15:], np.full((5,), -1, dtype=np.int64))
    neg1 = hard_neg_trasnform(input_id_feats1)
    assert hard_neg_trasnform.feat_dim == (hard_neg_trasnform._max_dim,)
    assert_equal(neg1["hard_neg"][:20], 99-input_feats1)
    assert_equal(neg1["hard_neg"][20][:15], np.array([(99-i) for i in range(15)]))
    assert_equal(neg1["hard_neg"][20][15:], np.full((5,), -1, dtype=np.int64))

    # nid map use int as key
    hard_neg_trasnform = HardEdgeDstNegativeTransform("hard_neg", "hard_neg")
    hard_neg_trasnform.set_target_etype(("src", "rel", "dst"))
    id_maps = {"dst": IdMap(raw_ids)}
    hard_neg_trasnform.set_id_maps(id_maps)

    input_feats = np.random.randint(0, 100, size=(20, 10), dtype=np.int64)
    input_id_feats = input_feats.astype(id_dtype)
    info = hard_neg_trasnform.pre_process(input_id_feats)
    assert info["hard_neg"] == 10

    hard_neg_trasnform.update_info([info["hard_neg"]])
    assert hard_neg_trasnform._max_dim == 10

    neg = hard_neg_trasnform(input_id_feats)
    assert hard_neg_trasnform.feat_dim == (hard_neg_trasnform._max_dim,)
    assert_equal(neg["hard_neg"], 99-input_feats)

    hard_neg_trasnform = HardEdgeDstNegativeTransform("hard_neg", "hard_neg", separator=",")
    hard_neg_trasnform.set_target_etype(("src", "rel", "dst"))
    hard_neg_trasnform.set_id_maps(id_maps)

    input_feats = np.random.randint(0, 100, size=(20, 10), dtype=np.int64)
    input_id_feats = [",".join(feats) for feats in input_feats.astype(str).tolist()]
    input_id_feats.append(",".join([str(i) for i in range(15)]))
    input_id_feats = np.array(input_id_feats)
    info = hard_neg_trasnform.pre_process(input_id_feats)
    assert info["hard_neg"] == 15

    hard_neg_trasnform.update_info([info["hard_neg"]])
    assert hard_neg_trasnform._max_dim == 15
    neg0 = hard_neg_trasnform(input_id_feats0)
    assert_equal(neg0["hard_neg"][:20,:10], 99-input_feats0)
    assert_equal(neg0["hard_neg"][20][:15], np.array([(99-i) for i in range(15)]))

    # test when there are empty string in input array
    hard_neg_trasnform = HardEdgeDstNegativeTransform("hard_neg", "hard_neg")
    hard_neg_trasnform.set_target_etype(("src", "rel", "dst"))
    id_maps = {"dst": IdMap(raw_ids.astype(id_dtype))}
    hard_neg_trasnform.set_id_maps(id_maps)

    input_feats = np.random.randint(0, 100, size=(20, 10), dtype=np.int64)
    input_id_feats = input_feats.tolist()
    input_id_feats[0] = input_id_feats[0][:-1]
    input_id_feats[1] = input_id_feats[1][:-2]
    input_id_feats = np.array([np.array(feat) for feat in input_id_feats], dtype=object)
    info = hard_neg_trasnform.pre_process(input_id_feats)
    assert info["hard_neg"] == 10

    hard_neg_trasnform.update_info([info["hard_neg"]])
    assert hard_neg_trasnform._max_dim == 10

    neg = hard_neg_trasnform(input_id_feats)
    assert hard_neg_trasnform.feat_dim == (hard_neg_trasnform._max_dim,)
    ground_truth = 99-input_feats
    ground_truth[0][-1] = -1
    ground_truth[1][-1] = -1
    ground_truth[1][-2] = -1
    assert_equal(neg["hard_neg"][:,:10], ground_truth)

@pytest.mark.parametrize("input_dtype", [np.cfloat, np.float32])
def test_standard_pre_process(input_dtype):
    np.random.seed(0)
    feats0 = np.random.randn(100).astype(input_dtype)
    feats0 = feats0 + 1
    feats1 = np.random.randn(100).astype(input_dtype)
    feats1 = feats1 + 1
    info = []
    transform = NumericalStandardTransform("test", "test")
    summation = transform.pre_process(feats0)["test"]
    info.append(summation)
    assert_almost_equal(summation, sum(feats0.tolist()), decimal=5)

    summation = transform.pre_process(feats1)["test"]
    info.append(summation)
    assert_almost_equal(summation, sum(feats1.tolist()), decimal=5)
    transform.update_info(info)
    assert_almost_equal(transform._summation,
                        info[0]+info[1])

    # Sum value is provided.
    conf = {"sum": 1.0}
    transform = NumericalStandardTransform("test", "test", np.array([2.0]), transform_conf=conf)
    summation = transform.pre_process(feats0)["test"]
    assert summation is None
    info = [None]
    transform.update_info(info)
    assert_equal(transform._summation, 2.0)
    assert_equal(conf["sum"], 1.0)

    # there are multiple columns of values
    feats0 = np.random.randn(100,3).astype(input_dtype)
    feats0 = feats0 + 1
    feats1 = np.random.randn(100,3).astype(input_dtype)
    feats1 = feats1 + 1

    conf = {}
    info = []
    transform = NumericalStandardTransform("test", "test", transform_conf=conf)
    summation = transform.pre_process(feats0)["test"]
    info.append(summation)
    assert len(summation) == 3
    assert_almost_equal(summation.tolist()[0],
                        sum(feats0[:,0].tolist()),
                        decimal=4)
    assert_almost_equal(summation.tolist()[1],
                        sum(feats0[:,1].tolist()),
                        decimal=4)
    assert_almost_equal(summation.tolist()[2],
                        sum(feats0[:,2].tolist()),
                        decimal=4)

    summation = transform.pre_process(feats1)["test"]
    info.append(summation)
    assert len(summation) == 3
    assert_almost_equal(summation.tolist()[0],
                        sum(feats1[:,0].tolist()),
                        decimal=4)
    assert_almost_equal(summation.tolist()[1],
                        sum(feats1[:,1].tolist()),
                        decimal=4)
    assert_almost_equal(summation.tolist()[2],
                        sum(feats1[:,2].tolist()),
                        decimal=4)
    transform.update_info(info)
    assert_almost_equal(transform._summation,
                        info[0]+info[1])
    assert_almost_equal(conf["sum"],
                        info[0]+info[1])

    # input is invalid
    feats0[0] = np.nan
    with assert_raises(AssertionError):
        _ = transform.pre_process(feats0)

    feats0[0] = np.inf
    with assert_raises(AssertionError):
        _ = transform.pre_process(feats0)

    # array sum is zero
    info = [np.array([-1]), np.array([1])]
    transform = NumericalStandardTransform("test", "test")
    with assert_raises(AssertionError):
        transform.update_info(info)

    # array sum has zero
    info = [np.array([1, 2, -1]), np.array([1, 1, 1])]
    transform = NumericalStandardTransform("test", "test")
    with assert_raises(AssertionError):
        transform.update_info(info)


@pytest.mark.parametrize("input_dtype", [np.cfloat, np.float32])
def test_standard_transform(input_dtype):
    np.random.seed(0)
    feats0 = np.random.randn(100).astype(input_dtype)
    feats0 = feats0 + 1

    transform = NumericalStandardTransform("test", "test")
    summation = np.sum(feats0, keepdims=True)
    transform._summation = summation.reshape((-1,))
    out = transform(feats0)["test"]
    assert transform.feat_dim == (1,)
    assert_almost_equal(out, feats0/summation)

    # given sum
    transform = NumericalStandardTransform("test", "test", 20.2)
    out = transform(feats0)["test"]
    assert transform.feat_dim == (1,)
    assert_almost_equal(out, feats0/20.2)

    # there are multiple columns of values
    feats0 = np.random.randn(100,3).astype(input_dtype)
    feats0 = feats0 + 1

    # there are multiple columns of values
    # summation is updated by update_info
    transform = NumericalStandardTransform("test", "test")
    summation = transform.pre_process(feats0)["test"]
    info = [summation, summation, summation]
    transform.update_info(info)
    out = transform(feats0)["test"]
    assert transform.feat_dim == (3,)
    assert_almost_equal(out, feats0/summation/3)


def test_hf_tokenizer(bert_model="bert-base-uncased"):
    max_seq_length = 16
    transform = Tokenizer("test", "test", bert_model, max_seq_length=max_seq_length)
    input_texts = "A Graph neural network (GNN) is a class of artificial neural networks for processing data that can be represented as graphs."
    tokenizer_result = transform(input_texts)
    assert transform.feat_dim == (768,)

    # Expected token
    tokens = []
    att_mask_list = []
    token_type_ids = []
    for text in input_texts:
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        t = tokenizer(text, max_length=max_seq_length,
                        truncation=True, padding='max_length', return_tensors='pt')
        tokens.append(t['input_ids'])
        att_mask_list.append(t['attention_mask'].to(th.int8))
        token_type_ids.append(t.get('token_type_ids', th.zeros_like(t['input_ids'])).to(th.int8))
    tokens = th.cat(tokens, dim=0).numpy()
    att_mask_list = th.cat(att_mask_list, dim=0).numpy()
    token_type_ids = th.cat(token_type_ids, dim=0).numpy()
    assert_equal(tokenizer_result["input_ids"], tokens)
    assert_equal(tokenizer_result["attention_mask"], att_mask_list)
    assert_equal(tokenizer_result["token_type_ids"], token_type_ids)


def test_hf_embedding(bert_model="bert-base-uncased"):
    max_seq_length = 16
    input_texts = "A Graph neural network (GNN) is a class of artificial neural networks for processing data that can be represented as graphs."
    transform = Text2BERT("test", "test",
              Tokenizer("test", "test", bert_model, max_seq_length),
              bert_model)
    hf_emb = transform(input_texts)['test']
    assert transform.feat_dim == (768,)

    # Tokenize the original text data for validation
    tokenizer = Tokenizer("test", "test", bert_model, max_seq_length)
    config = AutoConfig.from_pretrained(bert_model)
    lm_model = AutoModel.from_pretrained(bert_model, config)
    lm_model.eval()
    lm_model = lm_model.to("cpu")

    outputs = tokenizer(input_texts)
    tokens_list = [th.tensor(outputs['input_ids'])]
    att_masks_list = [th.tensor(outputs['attention_mask'])]
    token_types_list = [th.tensor(outputs['token_type_ids'])]
    with th.no_grad():
        out_embs = []
        for tokens, att_masks, token_types in zip(tokens_list, att_masks_list,
                                                  token_types_list):
            outputs = lm_model(tokens,
                                attention_mask=att_masks.long(),
                                token_type_ids=token_types.long())
            out_embs.append(outputs.pooler_output.cpu().numpy())
    if len(out_embs) > 1:
        feats = np.concatenate(out_embs)
    else:
        feats = out_embs[0]

    expected_output = feats
    for idx, _ in enumerate(hf_emb):
        np.testing.assert_almost_equal(
            hf_emb[idx], expected_output[idx], decimal=3, err_msg=f"Row {idx} is not equal"
        )
        

def test_update_ops():
    """ test the update operation function
    """
    np.random.seed(0)
    feats0 = np.random.randn(100)
    feats0 = feats0 + 1
    feats1 = np.random.randn(100)
    feats1 = feats1 + 1
    feats2 = np.random.randn(100)
    feats2 = feats2 + 1
    
    # only feats0 and feats2 are in the input feature dict
    input_feats = {
        'feat0': feats0,
        'feat2': feats2
    }
    # all three features have transforms in construction json
    ops = [
        NumericalStandardTransform("feat0", "feat0"),
        GlobalProcessFeatTransform("feat1", "feat1", np.float32),
        NumericalMinMaxTransform("feat2", "feat2")
    ]

    # run update_ops
    new_feat_ops, two_phase_feat_ops, after_merge_feat_ops, hard_edge_neg_ops = \
        update_ops(ops, input_feats)
    
    # new ops list should only have two ops
    assert len(new_feat_ops) == 2
    assert len(two_phase_feat_ops) == 2
    assert len(after_merge_feat_ops) == 0
    assert len(hard_edge_neg_ops) == 0
    # new ops list should only have feat0 and feat2
    assert all([op.feat_name in ['feat0', 'feat2'] for op in new_feat_ops])

if __name__ == '__main__':
    test_update_ops()
