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
import random
import os
import tempfile
import pytest
import decimal
import pyarrow.parquet as pq
import numpy as np
import dgl
import torch as th
import pandas as pd
import copy

from functools import partial
from numpy.testing import assert_equal, assert_almost_equal

from graphstorm.gconstruct.construct_graph import parse_edge_data, verify_confs, is_homogeneous
from graphstorm.gconstruct.file_io import write_data_parquet, read_data_parquet
from graphstorm.gconstruct.file_io import write_data_json, read_data_json
from graphstorm.gconstruct.file_io import write_data_csv, read_data_csv
from graphstorm.gconstruct.file_io import write_data_hdf5, read_data_hdf5, HDF5Array
from graphstorm.gconstruct.file_io import write_index_json
from graphstorm.gconstruct.transform import parse_feat_ops, process_features, preprocess_features
from graphstorm.gconstruct.transform import parse_label_ops, process_labels
from graphstorm.gconstruct.transform import Noop, do_multiprocess_transform, LinkPredictionProcessor
from graphstorm.gconstruct.id_map import IdMap, IdReverseMap, map_node_ids
from graphstorm.gconstruct.transform import (BucketTransform, RankGaussTransform,
                                             Text2BERT, NumericalMinMaxTransform)
from graphstorm.gconstruct.utils import (ExtMemArrayMerger,
                                         ExtMemArrayWrapper,
                                         partition_graph,
                                         update_two_phase_feat_ops,
                                         HDF5Array, ExtFeatureWrapper)

def test_parquet():
    handle, tmpfile = tempfile.mkstemp()
    os.close(handle)

    data = {}
    data["data1"] = np.random.rand(10, 3)
    data["data2"] = np.random.rand(10)
    write_data_parquet(data, tmpfile)
    data1 = read_data_parquet(tmpfile)
    assert len(data1) == 2
    assert "data1" in data1
    assert "data2" in data1
    np.testing.assert_array_equal(data1['data1'], data['data1'])
    np.testing.assert_array_equal(data1['data2'], data['data2'])

    data1 = read_data_parquet(tmpfile, data_fields=['data1'])
    assert len(data1) == 1
    assert "data1" in data1
    assert "data2" not in data1
    np.testing.assert_array_equal(data1['data1'], data['data1'])

    # verify if a field does not exist.
    try:
        data1 = read_data_parquet(tmpfile, data_fields=['data1', 'data3'])
        assert False, "This shouldn't happen."
    except:
        pass
    os.remove(tmpfile)

    # create a parquet with variable length of data
    data = {
        "data": [[95, 85], [88], [76, 89, 92]]
    }
    df = pd.DataFrame(data)
    df.to_parquet(tmpfile)

    data1 = read_data_parquet(tmpfile, data_fields=['data'])
    assert len(data1) == 1
    assert "data" in data1
    data1 = data1["data"]
    assert len(data1) == 3
    np.testing.assert_equal(data1[0], np.array(data["data"][0]))
    np.testing.assert_equal(data1[1], np.array(data["data"][1]))
    np.testing.assert_equal(data1[2], np.array(data["data"][2]))

    os.remove(tmpfile)

@pytest.mark.parametrize("delimiter", [None, ",", "\t"])
def test_csv(delimiter):
    data = {
            "t1": np.random.uniform(size=(10,)),
            "t2": np.random.uniform(size=(10,)),
    }
    with tempfile.TemporaryDirectory() as tmpdirname:
        if not delimiter:
            write_data_csv(data, os.path.join(tmpdirname, 'test.csv'))
            data1 = read_data_csv(os.path.join(tmpdirname, 'test.csv'))
        else:
            write_data_csv(data, os.path.join(tmpdirname, 'test.csv'), delimiter=delimiter)
            data1 = read_data_csv(os.path.join(tmpdirname, 'test.csv'), delimiter=delimiter)

        for key, val in data.items():
            assert key in data1
            np.testing.assert_almost_equal(data1[key], data[key])

        data1 = read_data_csv(os.path.join(tmpdirname, 'test.csv'), data_fields=['t1']) if not delimiter \
            else read_data_csv(os.path.join(tmpdirname, 'test.csv'), delimiter=delimiter, data_fields=['t1'])
        assert 't1' in data1
        np.testing.assert_almost_equal(data1['t1'], data['t1'])

def check_json_same_len(tmpfile):
    data = {}
    data["data1"] = np.random.rand(10, 3)
    data["data2"] = np.random.rand(10)
    write_data_json(data, tmpfile)
    data1 = read_data_json(tmpfile, ["data1", "data2"])
    assert len(data1) == 2
    assert "data1" in data1
    assert "data2" in data1
    assert data['data2'].dtype == np.float64
    assert np.all(data1['data1'] == data['data1'])
    assert np.all(data1['data2'] == data['data2'])

    # Test the case that some field doesn't exist.
    try:
        data1 = read_data_json(tmpfile, ["data1", "data3"])
        assert False, "This shouldn't happen"
    except:
        pass

def check_json_unequal_len(tmpfile):
    # Here we write arrays of different lengths to a JSON file.
    data = {
            "data": [np.random.rand(size) for size in [10, 5, 2, 5]]
    }
    write_data_json(data, tmpfile)
    data1 = read_data_json(tmpfile, ["data"])
    assert len(data1) == 1
    assert "data" in data1
    assert data1['data'].shape == (4, 10)
    for i, arr in enumerate(data["data"]):
        assert np.all(data1["data"][i][:len(arr)] == arr)
        assert np.all(data1["data"][i][len(arr):] == 0)

def test_json():
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_json_same_len(os.path.join(tmpdirname, "same_len.json"))
        check_json_unequal_len(os.path.join(tmpdirname, "unequal_len.json"))

def test_hdf5():
    handle, tmpfile = tempfile.mkstemp()
    os.close(handle)

    data = {}
    data["data1"] = np.random.rand(10, 3)
    data["data2"] = np.random.rand(10)
    write_data_hdf5(data, tmpfile)
    data1 = read_data_hdf5(tmpfile)
    assert len(data1) == 2
    assert "data1" in data1
    assert "data2" in data1
    np.testing.assert_array_equal(data1['data1'], data['data1'])
    np.testing.assert_array_equal(data1['data2'], data['data2'])

    data1 = read_data_hdf5(tmpfile, data_fields=['data1'])
    assert len(data1) == 1
    assert "data1" in data1
    assert "data2" not in data1
    np.testing.assert_array_equal(data1['data1'], data['data1'])

    try:
        data1 = read_data_hdf5(tmpfile, data_fields=['data1', "data3"])
        assert False, "This should not happen."
    except:
        pass

    # Test HDF5Array
    data1 = read_data_hdf5(tmpfile, data_fields=['data1'], in_mem=False)
    assert isinstance(data1['data1'], HDF5Array)
    np.testing.assert_array_equal(data1['data1'][:], data['data1'][:])
    idx = np.arange(0, len(data1['data1']), 2)
    np.testing.assert_array_equal(data1['data1'][idx], data['data1'][idx])
    idx = th.randint(0, len(data1['data1']), size=(100,))
    np.testing.assert_array_equal(data1['data1'][idx], data['data1'][idx])
    idx = np.random.randint(0, len(data1['data1']), size=(100,))
    np.testing.assert_array_equal(data1['data1'][idx], data['data1'][idx])

    os.remove(tmpfile)

def check_feat_ops_noop():
    # Just get the features without transformation.
    feat_op1 = [{
        "feature_col": "test1",
        "feature_name": "test2",
    }]
    (res, _, _, _) = parse_feat_ops(feat_op1)
    assert len(res) == 1
    assert res[0].col_name == feat_op1[0]["feature_col"]
    assert res[0].feat_name == feat_op1[0]["feature_name"]
    assert isinstance(res[0], Noop)

    data = {
        "test1": np.random.rand(4, 2),
    }
    proc_res = process_features(data, res)
    assert "test2" in proc_res
    assert proc_res["test2"].dtype == np.float32
    np.testing.assert_allclose(proc_res["test2"], data["test1"])

    # When the feature name is not specified.
    feat_op1 = [{
        "feature_col": "test1",
    }]
    (res, _, _, _) = parse_feat_ops(feat_op1)
    assert len(res) == 1
    assert res[0].col_name == feat_op1[0]["feature_col"]
    assert res[0].feat_name == feat_op1[0]["feature_col"]
    assert isinstance(res[0], Noop)

def check_feat_ops_tokenize():
    feat_op2 = [
        {
            "feature_col": "test1",
            "feature_name": "test2",
        },
        {
            "feature_col": "test3",
            "feature_name": "test4",
            "transform": {"name": 'tokenize_hf',
                'bert_model': 'bert-base-uncased',
                'max_seq_length': 16
            },
        },
    ]
    (res, _, _, _) = parse_feat_ops(feat_op2)
    assert len(res) == 2
    assert res[1].col_name == feat_op2[1]["feature_col"]
    assert res[1].feat_name == feat_op2[1]["feature_name"]
    op = res[1]
    tokens = op(["hello world", "hello world"])
    assert len(tokens) == 3
    assert tokens['input_ids'].shape == (2, 16)
    assert tokens['attention_mask'].shape == (2, 16)
    assert tokens['token_type_ids'].shape == (2, 16)
    np.testing.assert_array_equal(tokens['input_ids'][0],
                                  tokens['input_ids'][1])
    np.testing.assert_array_equal(tokens['attention_mask'][0],
                                  tokens['attention_mask'][1])
    np.testing.assert_array_equal(tokens['token_type_ids'][0],
                                  tokens['token_type_ids'][1])

    data = {
        "test1": np.random.rand(2, 4).astype(np.float32),
        "test3": ["hello world", "hello world"],
    }
    proc_res = process_features(data, res)
    np.testing.assert_array_equal(data['test1'], proc_res['test2'])
    assert "input_ids" in proc_res
    assert "attention_mask" in proc_res
    assert "token_type_ids" in proc_res

def check_feat_ops_bert():
    feat_op3 = [
        {
            "feature_col": "test3",
            "feature_name": "test4",
            "transform": {"name": 'bert_hf',
                'bert_model': 'bert-base-uncased',
                'max_seq_length': 16
            },
        },
    ]
    (res, _, _, _) = parse_feat_ops(feat_op3)
    assert len(res) == 1
    assert res[0].col_name == feat_op3[0]["feature_col"]
    assert res[0].feat_name == feat_op3[0]["feature_name"]
    data = {
        "test1": np.random.rand(2, 4).astype(np.float32),
        "test3": ["hello world", "hello world"],
    }
    proc_res = process_features(data, res)
    assert "test4" in proc_res
    assert len(proc_res['test4']) == 2
    # There are two text strings and both of them are "hello world".
    # The BERT embeddings should be the same.
    np.testing.assert_array_equal(proc_res['test4'][0], proc_res['test4'][1])
    # Compute BERT embeddings with multiple mini-batches.
    feat_op4 = [
        {
            "feature_col": "test3",
            "feature_name": "test4",
            "transform": {"name": 'bert_hf',
                'bert_model': 'bert-base-uncased',
                'max_seq_length': 16,
                'infer_batch_size': 1,
            },
        },
    ]
    (res2, _, _, _)  = parse_feat_ops(feat_op4)
    assert len(res2) == 1
    assert res2[0].col_name == feat_op4[0]["feature_col"]
    assert res2[0].feat_name == feat_op4[0]["feature_name"]
    proc_res2 = process_features(data, res2)
    assert "test4" in proc_res2
    assert len(proc_res2['test4']) == 2
    np.testing.assert_allclose(proc_res['test4'], proc_res2['test4'], rtol=1e-3)

def check_feat_ops_maxmin():
    data0 = {
        "test1": np.random.rand(4, 2),
    }
    data1 = {
        "test1": np.random.rand(4, 2),
    }
    feat_op5 = [
        {
            "feature_col": "test1",
            "feature_name": "test5",
            "transform": {"name": 'max_min_norm',
            },
        },
    ]
    (res, _, _, _) = parse_feat_ops(feat_op5)
    assert len(res) == 1
    assert res[0].col_name == feat_op5[0]["feature_col"]
    assert res[0].feat_name == feat_op5[0]["feature_name"]
    preproc_res0 = preprocess_features(data0, res)
    preproc_res1 = preprocess_features(data1, res)
    assert "test5" in preproc_res0
    assert isinstance(preproc_res0["test5"], tuple)
    assert "test5" in preproc_res1
    assert isinstance(preproc_res1["test5"], tuple)
    return_dict = {
        0: preproc_res0,
        1: preproc_res1
    }
    update_two_phase_feat_ops(return_dict, res)
    data_col0 = data0["test1"][:,0].tolist() + data1["test1"][:,0].tolist()
    data_col1 = data0["test1"][:,1].tolist() + data1["test1"][:,1].tolist()
    max0 = max(data_col0)
    min0 = min(data_col0)
    max1 = max(data_col1)
    min1 = min(data_col1)

    proc_res3 = process_features(data0, res)
    assert "test5" in proc_res3
    proc_res4 = process_features(data1, res)
    assert "test5" in proc_res4
    proc_res5 = np.concatenate([proc_res3["test5"], proc_res4["test5"]], axis=0)
    assert proc_res5.dtype == np.float32
    data_col0 = (np.array(data_col0) - min0) / (max0 - min0)
    data_col1 = (np.array(data_col1) - min1) / (max1 - min1)
    assert_almost_equal(proc_res5[:,0], data_col0)
    assert_almost_equal(proc_res5[:,1], data_col1)

    feat_op6 = [
        {
            "feature_col": "test1",
            "feature_name": "test6",
            "transform": {"name": 'max_min_norm',
                          "max_bound": 2.,
                          "min_bound": -2.
            },
        },
    ]
    (res2, _, _, _)  = parse_feat_ops(feat_op6)
    assert len(res2) == 1
    assert res2[0].col_name == feat_op6[0]["feature_col"]
    assert res2[0].feat_name == feat_op6[0]["feature_name"]
    preproc_res0 = preprocess_features(data0, res2)
    preproc_res1 = preprocess_features(data1, res2)
    assert "test6" in preproc_res0
    assert isinstance(preproc_res0["test6"], tuple)
    assert "test6" in preproc_res1
    assert isinstance(preproc_res1["test6"], tuple)
    return_dict = {
        0: preproc_res0,
        1: preproc_res1
    }
    update_two_phase_feat_ops(return_dict, res2)
    data_col0 = data0["test1"][:,0].tolist() + data1["test1"][:,0].tolist()
    data_col1 = data0["test1"][:,1].tolist() + data1["test1"][:,1].tolist()
    max0 = max(data_col0)
    min0 = min(data_col0)
    max1 = max(data_col1)
    min1 = min(data_col1)
    data_col0 = [val if val < max0 else max0 for val in data_col0]
    data_col0 = [val if val > min0 else min0 for val in data_col0]
    data_col1 = [val if val < max1 else max1 for val in data_col1]
    data_col1 = [val if val > min1 else min1 for val in data_col1]

    proc_res3 = process_features(data0, res2)
    assert "test6" in proc_res3
    proc_res4 = process_features(data1, res2)
    assert "test6" in proc_res4
    proc_res6 = np.concatenate([proc_res3["test6"], proc_res4["test6"]], axis=0)
    assert proc_res6.dtype == np.float32
    data_col0 = (np.array(data_col0) - min0) / (max0 - min0)
    data_col1 = (np.array(data_col1) - min1) / (max1 - min1)
    assert_almost_equal(proc_res6[:,0], data_col0)
    assert_almost_equal(proc_res6[:,1], data_col1)

def check_feat_ops_rank_gauss():
    data7_0 = {
        "test1": np.random.randn(100,2).astype(np.float32)
    }
    data7_1 = {
        "test1": np.random.randn(100,2).astype(np.float32)
    }

    feat_op7 = [
        {
            "feature_col": "test1",
            "feature_name": "test7",
            "transform": {
                "name": 'rank_gauss'
            },
        },
    ]
    (res, _, _, _) = parse_feat_ops(feat_op7)
    assert len(res) == 1
    assert res[0].col_name == feat_op7[0]["feature_col"]
    assert res[0].feat_name == feat_op7[0]["feature_name"]
    proc_res7_0 = process_features(data7_0, res)
    proc_res7_1 = process_features(data7_1, res)
    new_feat = np.concatenate([proc_res7_0["test7"], proc_res7_1["test7"]])
    trans_feat = res[0].after_merge_transform(new_feat)
    assert trans_feat.dtype == np.float32
    # sum of gauss rank should be zero
    trans_feat = np.sort(trans_feat, axis=0)
    rev_trans_feat = np.flip(trans_feat, axis=0)
    assert np.all(trans_feat + rev_trans_feat == 0)
    num_unique_feat = np.unique(trans_feat, axis=0)
    assert len(num_unique_feat) == 200

    data7_0 = {
        "test1": np.zeros((100,2)).astype(np.float32)
    }
    data7_1 = {
        "test1": np.ones((100,2)).astype(np.float32)
    }
    feat_op7 = [
        {
            "feature_col": "test1",
            "feature_name": "test7",
            "transform": {
                "name": 'rank_gauss',
                "epsilon": 1e-5,
                "uniquify": True
            },
        },
    ]
    (res, _, _, _) = parse_feat_ops(feat_op7)
    assert len(res) == 1
    assert res[0].col_name == feat_op7[0]["feature_col"]
    assert res[0].feat_name == feat_op7[0]["feature_name"]
    proc_res7_0 = process_features(data7_0, res)
    proc_res7_1 = process_features(data7_1, res)
    new_feat = np.concatenate([proc_res7_0["test7"], proc_res7_1["test7"]])
    trans_feat = res[0].after_merge_transform(new_feat)
    assert trans_feat.dtype == np.float32
    # sum of gauss rank should be zero
    trans_feat = np.sort(trans_feat, axis=0)
    rev_trans_feat = np.flip(trans_feat, axis=0)
    assert np.all(trans_feat + rev_trans_feat == 0)
    num_unique_feat = np.unique(trans_feat, axis=0)
    assert len(num_unique_feat) == 2

def check_feat_ops_categorical():
    feat_op7 = [
        {
            "feature_col": "test1",
            "feature_name": "test7",
            "transform": {"name": 'to_categorical'},
        },
    ]
    (res, _, _, _) = parse_feat_ops(feat_op7)
    data0 = {
        "test1": np.array([str(i) for i in np.random.randint(0, 10, size=10)]
            + [str(i) for i in range(10)]),
    }
    data1 = {
        "test1": np.array([str(i) for i in np.random.randint(0, 10, size=10)]),
    }
    preproc_res0 = preprocess_features(data0, res)
    preproc_res1 = preprocess_features(data1, res)
    assert "test7" in preproc_res0
    assert "test7" in preproc_res1
    return_dict = {
        0: preproc_res0,
        1: preproc_res1
    }
    update_two_phase_feat_ops(return_dict, res)
    proc_res3 = process_features(data0, res)
    assert "test7" in proc_res3
    assert 'mapping' in feat_op7[0]["transform"]
    for one_hot, str_i in zip(proc_res3["test7"], data0["test1"]):
        idx = feat_op7[0]["transform"]['mapping'][str_i]
        assert one_hot[idx] == 1

    feat_op8 = [
        {
            "feature_col": "test1",
            "feature_name": "test8",
            "transform": {"name": 'to_categorical', "separator": ","},
        },
    ]
    (res2, _, _, _)  = parse_feat_ops(feat_op8)
    data0 = {
        "test1": np.array([f"{i},{i+1}" for i in np.random.randint(0, 9, size=10)]
            + [str(i) for i in range(9)]),
    }
    data1 = {
        "test1": np.array([str(i) for i in np.random.randint(0, 10, size=10)]),
    }
    preproc_res0 = preprocess_features(data0, res2)
    preproc_res1 = preprocess_features(data1, res2)
    assert "test8" in preproc_res0
    assert "test8" in preproc_res1
    return_dict = {
        0: preproc_res0,
        1: preproc_res1
    }
    update_two_phase_feat_ops(return_dict, res2)
    proc_res3 = process_features(data0, res2)
    assert "test8" in proc_res3
    # We only need to test the first 10
    for multi_hot, str_i in zip(proc_res3["test8"][:10], data0["test1"][:10]):
        str_i1, str_i2 = str_i.split(",")
        assert multi_hot[int(str_i1)] == 1
        assert multi_hot[int(str_i2)] == 1
    assert 'mapping' in feat_op8[0]["transform"]

def test_feat_ops():
    check_feat_ops_noop()
    check_feat_ops_tokenize()
    check_feat_ops_bert()
    check_feat_ops_maxmin()
    check_feat_ops_categorical()
    check_feat_ops_rank_gauss()

def test_process_features_fp16():
    # Just get the features without transformation.
    data = {}
    data["test1"] = np.random.rand(10, 3)
    data["test2"] = np.random.rand(10)
    data["test3"] = np.random.rand(10) * 5
    handle, tmpfile = tempfile.mkstemp()
    os.close(handle)
    write_data_hdf5(data, tmpfile)

    feat_op1 = [{
        "feature_col": "test1",
        "feature_name": "test1",
        "out_dtype": "float16",
    },{
        "feature_col": "test2",
        "feature_name": "test2",
        "out_dtype": "float16",
    },{
        "feature_col": "test3",
        "feature_name": "test3",
        "out_dtype": "int8",
    }]
    (ops_rst, _, _, _) = parse_feat_ops(feat_op1)
    rst = process_features(data, ops_rst)
    assert len(rst) == 3
    assert 'test1' in rst
    assert 'test2' in rst
    assert 'test3' in rst
    assert (len(rst['test1'].shape)) == 2
    assert (len(rst['test2'].shape)) == 2
    assert (len(rst['test3'].shape)) == 2
    assert rst['test1'].dtype == np.float16
    assert rst['test2'].dtype == np.float16
    assert rst['test3'].dtype == np.int8
    assert_almost_equal(rst['test1'], data['test1'], decimal=3)
    assert_almost_equal(rst['test2'], data['test2'].reshape(-1, 1), decimal=3)
    assert_almost_equal(rst['test3'], data['test3'].reshape(-1, 1).astype(np.int8))

    data1 = read_data_hdf5(tmpfile, ['test1', 'test2', 'test3'], in_mem=False)
    rst2 = process_features(data1, ops_rst)
    assert isinstance(rst2["test1"], HDF5Array)
    assert len(rst2) == 3
    assert 'test1' in rst2
    assert 'test2' in rst2
    assert 'test3' in rst
    assert (len(rst2['test1'].to_tensor().shape)) == 2
    assert (len(rst2['test2'].shape)) == 2
    assert (len(rst['test3'].shape)) == 2
    assert rst2['test1'].to_tensor().dtype == th.float16
    assert rst2['test2'].dtype == np.float16
    assert rst['test3'].dtype == np.int8
    data_slice = rst2['test1'][np.arange(10)]
    assert data_slice.dtype == np.float16
    assert_almost_equal(rst2['test1'].to_tensor().numpy(), data['test1'], decimal=3)
    assert_almost_equal(rst2['test1'].to_tensor().numpy(), data_slice)
    assert_almost_equal(rst2['test2'], data['test2'].reshape(-1, 1), decimal=3)
    assert_almost_equal(rst2['test3'], data['test3'].reshape(-1, 1).astype(np.int8))

def test_process_features():
    # Just get the features without transformation.
    data = {}
    data["test1"] = np.random.rand(10, 3).astype(np.float32)
    data["test2"] = np.random.rand(10).astype(np.float32)

    feat_op1 = [{
        "feature_col": "test1",
        "feature_name": "test1",
    },{
        "feature_col": "test2",
        "feature_name": "test2",
    }]
    (ops_rst, _, _, _) = parse_feat_ops(feat_op1)
    rst = process_features(data, ops_rst)
    assert len(rst) == 2
    assert 'test1' in rst
    assert 'test2' in rst
    assert (len(rst['test1'].shape)) == 2
    assert (len(rst['test2'].shape)) == 2
    np.testing.assert_array_equal(rst['test1'], data['test1'])
    np.testing.assert_array_equal(rst['test2'], data['test2'].reshape(-1, 1))

def test_label():
    def check_split(res, train_mask, val_mask, test_mask):
        assert len(res) == 4
        assert 'label' in res
        assert train_mask in res
        assert val_mask in res
        assert test_mask in res
        assert res[train_mask].shape == (len(data['label']),)
        assert res[val_mask].shape == (len(data['label']),)
        assert res[test_mask].shape == (len(data['label']),)
        assert np.sum(res[train_mask]) == 8
        assert np.sum(res[val_mask]) == 1
        assert np.sum(res[test_mask]) == 1
        assert np.sum(res[train_mask] + res[val_mask] + res[test_mask]) == 10

    def check_no_split(res, train_mask, val_mask, test_mask):
        assert len(res) == 1
        assert 'label' in res
        assert train_mask not in res
        assert val_mask not in res
        assert test_mask not in res

    def check_integer(label, res, train_mask, val_mask, test_mask):
        train_idx = res[train_mask] == 1
        val_idx = res[val_mask] == 1
        test_idx = res[test_mask] == 1
        assert np.all(np.logical_and(label[train_idx] >= 0, label[train_idx] <= 10))
        assert np.all(np.logical_and(label[val_idx] >= 0, label[val_idx] <= 10))
        assert np.all(np.logical_and(label[test_idx] >= 0, label[test_idx] <= 10))

    # Check classification
    def check_classification(res, mask_field_names=[("train_mask", "val_mask", "test_mask")]):
        # default masks are train_mask, val_mask and test_mask
        check_split(res, mask_field_names[0][0], mask_field_names[0][1], mask_field_names[0][2])
        assert np.issubdtype(res['label'].dtype, np.integer)
        check_integer(res['label'], res, mask_field_names[0][0], mask_field_names[0][1], mask_field_names[0][2])

    def check_classification_no_split(res, mask_field_names=[("train_mask", "val_mask", "test_mask")]):
        # default masks are train_mask, val_mask and test_mask
        check_no_split(res, mask_field_names[0][0], mask_field_names[0][1], mask_field_names[0][2])
        assert np.issubdtype(res['label'].dtype, np.integer)

    conf = {
            "labels": [
                {'task_type': 'classification',
                 'label_col': 'label',
                 'split_pct': [0.8, 0.1, 0.1]}
            ]
    }
    ops = parse_label_ops(conf, True)
    data = {'label' : np.random.uniform(size=10) * 10}
    assert ops[0].train_mask_name == "train_mask"
    assert ops[0].val_mask_name == "val_mask"
    assert ops[0].test_mask_name == "test_mask"
    res = process_labels(data, ops)
    check_classification(res)
    ops = parse_label_ops(conf, False)
    assert ops[0].train_mask_name == "train_mask"
    assert ops[0].val_mask_name == "val_mask"
    assert ops[0].test_mask_name == "test_mask"
    res = process_labels(data, ops)
    check_classification(res)

    # Check classification with invalid labels.
    data = {'label' : np.random.uniform(size=13) * 10}
    data['label'][[0, 3, 4]] = np.NAN
    ops = parse_label_ops(conf, True)
    res = process_labels(data, ops)
    check_classification(res)

    # Check multi-label classification with invalid labels.
    data = {'label' : np.random.randint(2, size=(15,5)).astype(np.float32)}
    data['label'][[0,3,4]] = np.NAN
    data['label'][[1,2], [3,4]] = np.NAN
    ops = parse_label_ops(conf, True)
    res = process_labels(data, ops)
    check_classification(res)

    # Check classification with integer labels.
    data = {'label' : np.random.randint(10, size=10)}
    ops = parse_label_ops(conf, True)
    res = process_labels(data, ops)
    check_classification(res)

    # Check classification with integer labels.
    # Data split doesn't use all labeled samples.
    conf = {
            "labels": [
                {'task_type': 'classification',
                 'label_col': 'label',
                 'split_pct': [0.4, 0.05, 0.05]}
            ]
    }
    ops = parse_label_ops(conf, True)
    data = {'label' : np.random.randint(3, size=20)}
    res = process_labels(data, ops)
    check_classification(res)

    # Check classification with user defined mask names.
    conf = {
            "labels": [
                {'task_type': 'classification',
                 'label_col': 'label',
                 'split_pct': [0.4, 0.05, 0.05],
                 'mask_field_names': ['train_mask_c',
                                      'val_mask_c',
                                      'test_mask_c']}
            ]
    }
    ops = parse_label_ops(conf, True)
    assert ops[0].train_mask_name == "train_mask_c"
    assert ops[0].val_mask_name == "val_mask_c"
    assert ops[0].test_mask_name == "test_mask_c"
    data = {'label' : np.random.randint(3, size=20)}
    res = process_labels(data, ops)
    check_classification(res, [("train_mask_c", "val_mask_c", "test_mask_c")])

    # split_pct is not specified.
    conf = {
            "labels": [
                {'task_type': 'classification',
                 'label_col': 'label'}
            ]
    }
    ops = parse_label_ops(conf, True)
    data = {'label' : np.random.randint(3, size=10)}
    res = process_labels(data, ops)
    assert np.sum(res['train_mask']) == 8
    assert np.sum(res['val_mask']) == 1
    assert np.sum(res['test_mask']) == 1

    # split_pct is not specified and customized mask names.
    conf = {
            "labels": [
                {'task_type': 'classification',
                 'label_col': 'label',
                 'mask_field_names': ['train_mask_c',
                                      'val_mask_c',
                                      'test_mask_c']}
            ]
    }
    ops = parse_label_ops(conf, True)
    assert ops[0].train_mask_name == "train_mask_c"
    assert ops[0].val_mask_name == "val_mask_c"
    assert ops[0].test_mask_name == "test_mask_c"
    data = {'label' : np.random.randint(3, size=10)}
    res = process_labels(data, ops)
    assert np.sum(res['train_mask_c']) == 8
    assert np.sum(res['val_mask_c']) == 1
    assert np.sum(res['test_mask_c']) == 1

    # Check custom data split for classification on node.
    data = {
            "id": np.arange(10),
            'label' : np.random.randint(3, size=10)
    }
    write_index_json(np.arange(8), "/tmp/train_idx.json")
    write_index_json(np.arange(8, 9), "/tmp/val_idx.json")
    write_index_json(np.arange(9, 10), "/tmp/test_idx.json")
    conf = {
            "node_id_col": "id",
            "labels": [
                {'task_type': 'classification',
                 'label_col': 'label',
                 'custom_split_filenames': {"train": "/tmp/train_idx.json",
                                            "valid": "/tmp/val_idx.json",
                                            "test": "/tmp/test_idx.json"}}
            ]
    }
    ops = parse_label_ops(conf, True)
    res = process_labels(data, ops)
    check_classification(res)

    # Check custom data split for classification on node with customized mask names.
    conf = {
            "node_id_col": "id",
            "labels": [
                {'task_type': 'classification',
                 'label_col': 'label',
                 'custom_split_filenames': {"train": "/tmp/train_idx.json",
                                            "valid": "/tmp/val_idx.json",
                                            "test": "/tmp/test_idx.json"},
                 'mask_field_names': ['train_mask_c',
                                      'val_mask_c',
                                      'test_mask_c']}
            ]
    }
    ops = parse_label_ops(conf, True)
    assert ops[0].train_mask_name == "train_mask_c"
    assert ops[0].val_mask_name == "val_mask_c"
    assert ops[0].test_mask_name == "test_mask_c"
    res = process_labels(data, ops)
    check_classification(res, [("train_mask_c", "val_mask_c", "test_mask_c")])

    # Check custom data split for classification on edge.
    data = {
            "src": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
            "dest": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            'label': np.random.randint(3, size=10)
    }
    write_index_json([(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2)],
                     "/tmp/train_idx.json")
    write_index_json([(3, 3)], "/tmp/val_idx.json")
    write_index_json([(4, 1)], "/tmp/test_idx.json")
    conf = {
            "source_id_col": "src",
            "dest_id_col": "dest",
            "labels": [
                {'task_type': 'classification',
                 'label_col': 'label',
                 'custom_split_filenames': {"train": "/tmp/train_idx.json",
                                            "valid": "/tmp/val_idx.json",
                                            "test": "/tmp/test_idx.json"}}
            ]
    }
    ops = parse_label_ops(conf, False)
    res = process_labels(data, ops)
    check_classification(res)

    # Check custom data split for classification on edge with customized mask names
    conf = {
            "source_id_col": "src",
            "dest_id_col": "dest",
            "labels": [
                {'task_type': 'classification',
                 'label_col': 'label',
                 'custom_split_filenames': {"train": "/tmp/train_idx.json",
                                            "valid": "/tmp/val_idx.json",
                                            "test": "/tmp/test_idx.json"},
                 'mask_field_names': ['train_mask_ec',
                                      'val_mask_ec',
                                      'test_mask_ec']}
            ]
    }
    ops = parse_label_ops(conf, False)
    assert ops[0].train_mask_name == "train_mask_ec"
    assert ops[0].val_mask_name == "val_mask_ec"
    assert ops[0].test_mask_name == "test_mask_ec"
    res = process_labels(data, ops)
    check_classification(res, [("train_mask_ec", "val_mask_ec", "test_mask_ec")])

    # Check custom data split with only training set.
    data = {
            "id": np.arange(10),
            'label' : np.random.randint(3, size=10)
    }
    write_index_json(np.arange(8), "/tmp/train_idx.json")
    conf = {
            "node_id_col": "id",
            "labels": [
                {'task_type': 'classification',
                 'label_col': 'label',
                 'custom_split_filenames': {"train": "/tmp/train_idx.json"} }
            ]
    }
    ops = parse_label_ops(conf, True)
    res = process_labels(data, ops)
    assert "train_mask" in res
    assert np.sum(res["train_mask"]) == 8
    assert "val_mask" in res
    assert np.sum(res["val_mask"]) == 0
    assert "test_mask" in res
    assert np.sum(res["test_mask"]) == 0

    # Check custom data split with only training set and customized mask names
    conf = {
            "node_id_col": "id",
            "labels": [
                {'task_type': 'classification',
                 'label_col': 'label',
                 'custom_split_filenames': {"train": "/tmp/train_idx.json"},
                 'mask_field_names': ['train_mask_c',
                                      'val_mask_c',
                                      'test_mask_c'] }
            ]
    }
    ops = parse_label_ops(conf, True)
    assert ops[0].train_mask_name == "train_mask_c"
    assert ops[0].val_mask_name == "val_mask_c"
    assert ops[0].test_mask_name == "test_mask_c"
    res = process_labels(data, ops)
    assert "train_mask_c" in res
    assert np.sum(res["train_mask_c"]) == 8
    assert "val_mask_c" in res
    assert np.sum(res["val_mask_c"]) == 0
    assert "test_mask_c" in res
    assert np.sum(res["test_mask_c"]) == 0

    # Check custom data split without providing split
    conf = {
            "labels": [
                {'task_type': 'classification',
                 'label_col': 'label',
                 'split_pct': [0, 0, 0]}
            ]
    }
    ops = parse_label_ops(conf, True)
    data = {'label' : np.random.uniform(size=10) * 10}
    res = process_labels(data, ops)
    check_classification_no_split(res)

    # Check custom data split without providing split with customized mask names
    conf = {
            "labels": [
                {'task_type': 'classification',
                 'label_col': 'label',
                 'split_pct': [0, 0, 0],
                 'mask_field_names': ['train_mask_c',
                                      'val_mask_c',
                                      'test_mask_c']}
            ]
    }
    ops = parse_label_ops(conf, True)
    assert ops[0].train_mask_name == "train_mask_c"
    assert ops[0].val_mask_name == "val_mask_c"
    assert ops[0].test_mask_name == "test_mask_c"
    data = {'label' : np.random.uniform(size=10) * 10}
    res = process_labels(data, ops)
    check_classification_no_split(res, [("train_mask_c", "val_mask_c", "test_mask_c")])

    # Check regression
    def check_regression(res, mask_field_names=[("train_mask", "val_mask", "test_mask")]):
        check_split(res, mask_field_names[0][0], mask_field_names[0][1], mask_field_names[0][2])
    def check_regression_no_split(res, mask_field_names=[("train_mask", "val_mask", "test_mask")]):
        check_no_split(res, mask_field_names[0][0], mask_field_names[0][1], mask_field_names[0][2])

    conf = {
            "labels": [
                {'task_type': 'regression',
                 'label_col': 'label',
                 'split_pct': [0.8, 0.1, 0.1],
                 'mask_field_names': ['train_mask_l',
                                      'val_mask_l',
                                      'test_mask_l']}
            ]
    }
    ops = parse_label_ops(conf, True)
    assert ops[0].train_mask_name == "train_mask_l"
    assert ops[0].val_mask_name == "val_mask_l"
    assert ops[0].test_mask_name == "test_mask_l"
    data = {'label' : np.random.uniform(size=10) * 10}
    res = process_labels(data, ops)
    check_regression(res, [("train_mask_l", "val_mask_l", "test_mask_l")])
    ops = parse_label_ops(conf, False)
    assert ops[0].train_mask_name == "train_mask_l"
    assert ops[0].val_mask_name == "val_mask_l"
    assert ops[0].test_mask_name == "test_mask_l"
    res = process_labels(data, ops)
    check_regression(res, [("train_mask_l", "val_mask_l", "test_mask_l")])

    # Check regression with customized mask names
    conf = {
            "labels": [
                {'task_type': 'regression',
                 'label_col': 'label',
                 'split_pct': [0.8, 0.1, 0.1],
                 'mask_field_names': ['train_mask_r',
                                      'val_mask_r',
                                      'test_mask_r']}
            ]
    }
    ops = parse_label_ops(conf, True)
    assert ops[0].train_mask_name == "train_mask_r"
    assert ops[0].val_mask_name == "val_mask_r"
    assert ops[0].test_mask_name == "test_mask_r"
    data = {'label' : np.random.uniform(size=10) * 10}
    res = process_labels(data, ops)
    check_regression(res, [("train_mask_r", "val_mask_r", "test_mask_r")])
    ops = parse_label_ops(conf, False)
    assert ops[0].train_mask_name == "train_mask_r"
    assert ops[0].val_mask_name == "val_mask_r"
    assert ops[0].test_mask_name == "test_mask_r"
    res = process_labels(data, ops)
    check_regression(res, [("train_mask_r", "val_mask_r", "test_mask_r")])

    conf = {
            "labels": [
                {'task_type': 'regression',
                 'label_col': 'label',
                 'split_pct': [0.8, 0.1, 0.1]
                }
            ]
    }
    ops = parse_label_ops(conf, True)
    assert ops[0].train_mask_name == "train_mask"
    assert ops[0].val_mask_name == "val_mask"
    assert ops[0].test_mask_name == "test_mask"
    data = {'label' : np.random.uniform(size=10) * 10}
    res = process_labels(data, ops)
    check_regression(res)
    ops = parse_label_ops(conf, False)
    res = process_labels(data, ops)
    check_regression(res)

    # Check regression with invalid labels.
    data = {'label' : np.random.uniform(size=13) * 10}
    data['label'][[0, 3, 4]] = np.NAN
    ops = parse_label_ops(conf, True)
    res = process_labels(data, ops)
    check_regression(res)

    # Check custom data split for regression.
    data = {
            "id": np.arange(10),
            'label' : np.random.uniform(size=10) * 10
    }
    write_index_json(np.arange(8), "/tmp/train_idx.json")
    write_index_json(np.arange(8, 9), "/tmp/val_idx.json")
    write_index_json(np.arange(9, 10), "/tmp/test_idx.json")
    conf = {
            "node_id_col": "id",
            "labels": [
                {'task_type': 'regression',
                 'label_col': 'label',
                 'custom_split_filenames': {"train": "/tmp/train_idx.json",
                                            "valid": "/tmp/val_idx.json",
                                            "test": "/tmp/test_idx.json"} }
            ]
    }
    ops = parse_label_ops(conf, True)
    res = process_labels(data, ops)
    check_regression(res)

    # Check custom data split without providing split
    conf = {
            "labels": [
                {'task_type': 'regression',
                 'label_col': 'label',
                 'split_pct': [0, 0, 0]}
            ]
    }
    data = {'label' : np.random.uniform(size=10) * 10}
    ops = parse_label_ops(conf, True)
    res = process_labels(data, ops)
    check_regression_no_split(res)

    # Check link prediction
    conf = {
            "labels": [
                {'task_type': 'link_prediction',
                 'split_pct': [0.8, 0.1, 0.1]}
            ]
    }
    ops = parse_label_ops(conf, False)
    assert ops[0].train_mask_name == "train_mask"
    assert ops[0].val_mask_name == "val_mask"
    assert ops[0].test_mask_name == "test_mask"
    data = {'label' : np.random.uniform(size=10) * 10}
    res = process_labels(data, ops)
    assert len(res) == 3
    assert 'train_mask' in res
    assert 'val_mask' in res
    assert 'test_mask' in res
    assert np.sum(res['train_mask']) == 8
    assert np.sum(res['val_mask']) == 1
    assert np.sum(res['test_mask']) == 1

    # Check link prediction with customized mask names
    conf = {
            "labels": [
                {'task_type': 'link_prediction',
                 'split_pct': [0.8, 0.1, 0.1],
                 'mask_field_names': ['train_mask_l',
                                      'val_mask_l',
                                      'test_mask_l']}
            ]
    }
    ops = parse_label_ops(conf, False)
    assert ops[0].train_mask_name == "train_mask_l"
    assert ops[0].val_mask_name == "val_mask_l"
    assert ops[0].test_mask_name == "test_mask_l"
    data = {'label' : np.random.uniform(size=10) * 10}
    res = process_labels(data, ops)
    assert len(res) == 3
    assert 'train_mask_l' in res
    assert 'val_mask_l' in res
    assert 'test_mask_l' in res
    assert np.sum(res['train_mask_l']) == 8
    assert np.sum(res['val_mask_l']) == 1
    assert np.sum(res['test_mask_l']) == 1

    # Check custom data split without providing split
    conf = {
            "labels": [
                {'task_type': 'link_prediction',
                 'split_pct': [0, 0, 0]}
            ]
    }
    ops = parse_label_ops(conf, False)
    data = {'label' : np.random.uniform(size=10) * 10}
    res = process_labels(data, ops)
    assert len(res) == 0
    assert 'train_mask' not in res
    assert 'val_mask' not in res
    assert 'test_mask' not in res

def test_multitask_label():
    def check_split(res, train_mask, val_mask, test_mask, label_field):
        assert label_field in res
        assert train_mask in res
        assert val_mask in res
        assert test_mask in res
        assert res[train_mask].shape == (len(data[label_field]),)
        assert res[val_mask].shape == (len(data[label_field]),)
        assert res[test_mask].shape == (len(data[label_field]),)
        assert np.sum(res[train_mask]) == 8
        assert np.sum(res[val_mask]) == 1
        assert np.sum(res[test_mask]) == 1
        assert np.sum(res[train_mask] + res[val_mask] + res[test_mask]) == 10

    def check_integer(label, res, train_mask, val_mask, test_mask):
        train_idx = res[train_mask] == 1
        val_idx = res[val_mask] == 1
        test_idx = res[test_mask] == 1
        assert np.all(np.logical_and(label[train_idx] >= 0, label[train_idx] <= 10))
        assert np.all(np.logical_and(label[val_idx] >= 0, label[val_idx] <= 10))
        assert np.all(np.logical_and(label[test_idx] >= 0, label[test_idx] <= 10))

    conf = {
            "labels": [
                {'task_type': 'link_prediction'},
                {'task_type': 'link_prediction'}
            ]
    }
    try:
        ops = parse_label_ops(conf, False)
        assert False, "Should fail as both two labels are using the same training mask"
    except:
        pass
    try:
        ops = parse_label_ops(conf, True)
        assert False, "Should fail as both two labels are using the same training mask"
    except:
        pass

    write_index_json(np.arange(8), "/tmp/train_idx.json")
    write_index_json(np.arange(8, 9), "/tmp/val_idx.json")
    write_index_json(np.arange(9, 10), "/tmp/test_idx.json")
    conf = {
            "node_id_col": "id",
            "labels": [
                {'task_type': 'classification',
                 'label_col': 'label_1',
                 'split_pct': [0.8, 0.1, 0.1]},
                {'task_type': 'classification',
                 'label_col': 'label_2',
                 'split_pct': [0.4, 0.05, 0.05],
                 'mask_field_names': ['train_mask_c1',
                                      'val_mask_c1',
                                      'test_mask_c1']},
                {'task_type': 'classification',
                 'label_col': 'label_3',
                 'split_pct': [0, 0, 0],
                 'mask_field_names': ['train_mask_c2',
                                      'val_mask_c2',
                                      'test_mask_c2']},
                {'task_type': 'regression',
                 'label_col': 'label_4',
                 'mask_field_names': ['train_mask_r1',
                                      'val_mask_r1',
                                      'test_mask_r1'],
                 'custom_split_filenames': {"train": "/tmp/train_idx.json",
                                            "valid": "/tmp/val_idx.json",
                                            "test": "/tmp/test_idx.json"}},
                {'task_type': 'regression',
                 'label_col': 'label_5',
                 'mask_field_names': ['train_mask_r2',
                                      'val_mask_r2',
                                      'test_mask_r2'],
                 'custom_split_filenames': {"train": "/tmp/train_idx.json"},}
            ]
    }
    ops = parse_label_ops(conf, True)
    assert len(ops) == 5
    assert ops[0].train_mask_name == "train_mask"
    assert ops[0].val_mask_name == "val_mask"
    assert ops[0].test_mask_name == "test_mask"
    assert ops[1].train_mask_name == "train_mask_c1"
    assert ops[1].val_mask_name == "val_mask_c1"
    assert ops[1].test_mask_name == "test_mask_c1"
    assert ops[3].train_mask_name == "train_mask_r1"
    assert ops[3].val_mask_name == "val_mask_r1"
    assert ops[3].test_mask_name == "test_mask_r1"
    assert ops[4].train_mask_name == "train_mask_r2"
    assert ops[4].val_mask_name == "val_mask_r2"
    assert ops[4].test_mask_name == "test_mask_r2"

    data = {'id': np.arange(10),
            'label_1' : np.random.randint(3, size=10),
            'label_2' : np.random.randint(3, size=20),
            'label_3' : np.random.randint(3, size=10),
            'label_4' : np.random.uniform(size=10) * 10,
            'label_5' : np.random.uniform(size=10) * 10}
    res = process_labels(data, ops)
    assert len(res) == 17
    check_split(res, "train_mask", "val_mask", "test_mask", "label_1")
    check_integer(res['label_1'], res, "train_mask", "val_mask", "test_mask")
    check_split(res, "train_mask_c1", "val_mask_c1", "test_mask_c1", "label_2")
    check_integer(res['label_2'], res, "train_mask_c1", "val_mask_c1", "test_mask_c1")
    assert "label_3" in res
    assert "label_4" in res
    check_split(res, "train_mask_r1", "val_mask_r1", "test_mask_r1", "label_4")
    assert "label_5" in res
    assert "train_mask_r2" in res
    assert np.sum(res["train_mask_r2"]) == 8
    assert "val_mask_r2" in res
    assert np.sum(res["val_mask_r2"]) == 0
    assert "test_mask_r2" in res
    assert np.sum(res["test_mask_r2"]) == 0

    write_index_json([(a,a) for a in range(8)], "/tmp/train_idx.json")
    write_index_json([(8, 8)], "/tmp/val_idx.json")
    write_index_json([(9, 9)], "/tmp/test_idx.json")
    # Check edge tasks
    conf = {
            "source_id_col": "src",
            "dest_id_col": "dst",
            "labels": [
                {'task_type': 'link_prediction',
                 'split_pct': [0.8, 0.1, 0.1],
                 'mask_field_names': ['train_mask_l',
                                      'val_mask_l',
                                      'test_mask_l']},
                {'task_type': 'regression',
                 'label_col': 'label1',
                 'split_pct': [0.8, 0.1, 0.1],
                 'mask_field_names': ['train_mask_r',
                                      'val_mask_r',
                                      'test_mask_r']
                },
                {'task_type': 'classification',
                 'label_col': 'label2',
                 'custom_split_filenames': {"train": "/tmp/train_idx.json",
                                            "valid": "/tmp/val_idx.json",
                                            "test": "/tmp/test_idx.json"},
                 'mask_field_names': ['train_mask_ec',
                                      'val_mask_ec',
                                      'test_mask_ec']},
            ]
    }
    ops = parse_label_ops(conf, False)
    assert len(ops) == 3
    assert ops[0].train_mask_name == "train_mask_l"
    assert ops[0].val_mask_name == "val_mask_l"
    assert ops[0].test_mask_name == "test_mask_l"
    assert ops[1].train_mask_name == "train_mask_r"
    assert ops[1].val_mask_name == "val_mask_r"
    assert ops[1].test_mask_name == "test_mask_r"
    assert ops[2].train_mask_name == "train_mask_ec"
    assert ops[2].val_mask_name == "val_mask_ec"
    assert ops[2].test_mask_name == "test_mask_ec"

    data = {"src": np.arange(10),
            "dst": np.arange(10),
            'label1' : np.random.randint(3, size=10),
            'label2' : np.random.randint(3, size=10)}
    res = process_labels(data, ops)
    assert 'train_mask_l' in res
    assert 'val_mask_l' in res
    assert 'test_mask_l' in res
    assert np.sum(res['train_mask_l']) == 8
    assert np.sum(res['val_mask_l']) == 1
    assert np.sum(res['test_mask_l']) == 1
    check_split(res, "train_mask_r", "val_mask_r", "test_mask_r", "label1")
    check_split(res, "train_mask_ec", "val_mask_ec", "test_mask_ec", "label2")

def check_link_prediction_custom_label():
    # Check link prediction on customized masks
    data = {
            "src": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
            "dest": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            "dest_2": [4, 5, 6, 6, 5, 4, 5, 6, 4, 6],
            'label': np.random.randint(3, size=10)
    }
    write_index_json([(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2)],
                     "/tmp/train_idx.json")
    write_index_json([(3, 3)], "/tmp/val_idx.json")
    write_index_json([(4, 1)], "/tmp/test_idx.json")
    conf = {
            "source_id_col": "src",
            "dest_id_col": "dest",
            "labels": [
                {'task_type': 'link_prediction',
                 'custom_split_filenames': {"train": "/tmp/train_idx.json",
                                            "valid": "/tmp/val_idx.json",
                                            "test": "/tmp/test_idx.json"}
                 }
            ]
    }
    ops = parse_label_ops(conf, False)
    assert ops[0].train_mask_name == "train_mask"
    assert ops[0].val_mask_name == "val_mask"
    assert ops[0].test_mask_name == "test_mask"
    res = process_labels(data, ops)
    assert len(res) == 3
    assert 'train_mask' in res
    assert 'val_mask' in res
    assert 'test_mask' in res
    assert np.sum(res['train_mask']) == 8
    assert np.sum(res['val_mask']) == 1
    assert np.sum(res['test_mask']) == 1

    # With customized mask names
    conf = {
            "source_id_col": "src",
            "dest_id_col": "dest",
            "labels": [
                {'task_type': 'link_prediction',
                 'custom_split_filenames': {"train": "/tmp/train_idx.json",
                                            "valid": "/tmp/val_idx.json",
                                            "test": "/tmp/test_idx.json"},
                 'mask_field_names': ['train_mask_l',
                                      'val_mask_l',
                                      'test_mask_l']
                 }
            ]
    }
    ops = parse_label_ops(conf, False)
    assert ops[0].train_mask_name == "train_mask_l"
    assert ops[0].val_mask_name == "val_mask_l"
    assert ops[0].test_mask_name == "test_mask_l"
    res = process_labels(data, ops)
    assert len(res) == 3
    assert 'train_mask_l' in res
    assert 'val_mask_l' in res
    assert 'test_mask_l' in res
    assert np.sum(res['train_mask_l']) == 8
    assert np.sum(res['val_mask_l']) == 1
    assert np.sum(res['test_mask_l']) == 1

def check_id_map_exist(id_map, input_ids):
    # Test the case that all Ids exist in the map.
    rand_ids = np.array([input_ids[random.randint(0, len(input_ids)) % len(input_ids)] for _ in range(5)])
    remap_ids, idx = id_map.map_id(rand_ids)
    assert len(idx) == len(rand_ids)
    assert np.issubdtype(remap_ids.dtype, np.integer)
    assert len(remap_ids) == len(rand_ids)
    for id1, id2 in zip(remap_ids, rand_ids):
        assert id1 == int(id2)

def check_id_map_not_exist(id_map, input_ids, out_range_ids):
    # Test the case that some of the Ids don't exist.
    rand_ids = np.array([input_ids[random.randint(0, len(input_ids)) % len(input_ids)] for _ in range(5)])
    rand_ids1 = np.concatenate([rand_ids, out_range_ids])
    remap_ids, idx = id_map.map_id(rand_ids1)
    assert len(remap_ids) == len(rand_ids)
    assert len(remap_ids) == len(idx)
    assert np.sum(idx >= len(rand_ids)) == 0
    for id1, id2 in zip(remap_ids, rand_ids):
        assert id1 == int(id2)

def check_id_map_dtype_not_match(id_map, str_ids):
    # Test the case that the ID array of integer type
    try:
        rand_ids = np.random.randint(10, size=5)
        remap_ids, idx = id_map.map_id(rand_ids)
        raise ValueError("fails")
    except:
        pass

    # Test the case that the ID map has integer keys.
    str_ids = np.array([i for i in range(10)])
    id_map = IdMap(str_ids)
    try:
        rand_ids = np.array([str(random.randint(0, len(str_ids))) for _ in range(5)])
        remap_ids, idx = id_map.map_id(rand_ids)
        raise ValueError("fails")
    except:
        pass

def test_id_map():
    # This tests all cases in IdMap.
    str_ids = np.array([str(i) for i in range(10)])
    id_map = IdMap(str_ids)

    check_id_map_exist(id_map, str_ids)
    check_id_map_not_exist(id_map, str_ids, np.array(["11", "15", "20"]))
    check_id_map_dtype_not_match(id_map, str_ids)

    # Test saving ID map with random IDs.
    ids = np.random.permutation(100)
    str_ids = np.array([str(i) for i in ids])
    id_map = IdMap(str_ids)
    id_map.save("/tmp/id_map.parquet")

    # Reconstruct the ID map from the parquet file.
    table = pq.read_table("/tmp/id_map.parquet")
    df_table = table.to_pandas()
    keys = np.array(df_table['orig'])
    vals = np.array(df_table['new'])
    new_id_map = {key: val for key, val in zip(keys, vals)}

    assert len(new_id_map) == len(id_map)
    new_ids1, _ = id_map.map_id(str_ids)
    new_ids2 = np.array([new_id_map[i] for i in str_ids])
    assert np.all(new_ids1 == new_ids2)

    # Test id map as other types such as decimal.Decimal (e.g., UUID)
    decial_ids = np.array([decimal.Decimal(i) for i in range(10)])
    id_map = IdMap(decial_ids)
    check_id_map_exist(id_map, decial_ids)
    check_id_map_not_exist(id_map, decial_ids, np.array([decimal.Decimal(11),
                                                         decimal.Decimal(15),
                                                         decimal.Decimal(20)]))

def test_id_reverse_map():
    with tempfile.TemporaryDirectory() as tmpdirname:
        str_ids = np.array([str(i) for i in range(10)])
        id_map = IdMap(str_ids)
        id_map.save(os.path.join(tmpdirname, "id_map.parquet"))
        id_reverse_map = IdReverseMap(os.path.join(tmpdirname, "id_map.parquet"))
        assert len(id_reverse_map) == len(id_map)

        test_ids = np.random.permutation(10)
        test_reverse = id_reverse_map.map_id(test_ids)
        new_ids = np.array([id_map._ids[str(key)] for key in test_reverse])
        assert_equal(test_ids, new_ids)

        test_reverse = id_reverse_map.map_range(1,9)
        new_ids = np.array([id_map._ids[str(key)] for key in test_reverse])
        assert_equal(np.arange(1, 9), new_ids)

def check_map_node_ids_exist(str_src_ids, str_dst_ids, id_map):
    # Test the case that both source node IDs and destination node IDs exist.
    src_ids = np.array([str(random.randint(0, len(str_src_ids) - 1)) for _ in range(15)])
    dst_ids = np.array([str(random.randint(0, len(str_dst_ids) - 1)) for _ in range(15)])
    new_src_ids, new_dst_ids, src_exist_locs, dst_exist_locs = \
        map_node_ids(src_ids, dst_ids, ("src", None, "dst"),
                                            id_map, False)
    assert len(new_src_ids) == len(src_ids)
    assert len(new_dst_ids) == len(dst_ids)
    assert src_exist_locs is None
    assert dst_exist_locs is None
    for src_id1, src_id2 in zip(new_src_ids, src_ids):
        assert src_id1 == int(src_id2)
    for dst_id1, dst_id2 in zip(new_dst_ids, dst_ids):
        assert dst_id1 == int(dst_id2)

def check_map_node_ids_src_not_exist(str_src_ids, str_dst_ids, id_map):
    # Test the case that source node IDs don't exist.
    src_ids = np.array([str(random.randint(0, 20)) for _ in range(15)])
    dst_ids = np.array([str(random.randint(0, len(str_dst_ids) - 1)) for _ in range(15)])
    try:
        new_src_ids, new_dst_ids, _, _ = \
            map_node_ids(src_ids, dst_ids, ("src", None, "dst"),
                                                id_map, False)
        raise ValueError("fail")
    except:
        pass

    # Test the case that source node IDs don't exist and we skip non exist edges.
    new_src_ids, new_dst_ids, src_exist_locs, dst_exist_locs \
        = map_node_ids(src_ids, dst_ids, ("src", None, "dst"),
                                            id_map, True)
    num_valid = sum([int(id_) < len(str_src_ids) for id_ in src_ids])
    assert len(new_src_ids) == num_valid
    assert len(new_dst_ids) == num_valid
    assert src_exist_locs is not None
    assert_equal(src_ids[src_exist_locs].astype(np.int64), new_src_ids)
    assert_equal(dst_ids[src_exist_locs].astype(np.int64), new_dst_ids)
    assert dst_exist_locs is None

    # Test the case that none of the source node IDs exists and we skip non exist edges.
    src_ids = np.array([str(random.randint(20, 100)) for _ in range(15)])
    new_src_ids, new_dst_ids, _, _ = map_node_ids(src_ids, dst_ids, ("src", None, "dst"),
                                            id_map, True)
    assert len(new_src_ids) == 0
    assert len(new_dst_ids) == 0

def check_map_node_ids_dst_not_exist(str_src_ids, str_dst_ids, id_map):
    # Test the case that destination node IDs don't exist.
    src_ids = np.array([str(random.randint(0, len(str_src_ids) - 1)) for _ in range(15)])
    dst_ids = np.array([str(random.randint(0, 20)) for _ in range(15)])
    try:
        new_src_ids, new_dst_ids, _, _ = \
            map_node_ids(src_ids, dst_ids, ("src", None, "dst"),
                                                id_map, False)
        raise ValueError("fail")
    except:
        pass

    # Test the case that destination node IDs don't exist and we skip non exist edges.
    new_src_ids, new_dst_ids, src_exist_locs, dst_exist_locs = map_node_ids(src_ids, dst_ids, ("src", None, "dst"),
                                            id_map, True)
    num_valid = sum([int(id_) < len(str_dst_ids) for id_ in dst_ids])
    assert len(new_src_ids) == num_valid
    assert len(new_dst_ids) == num_valid
    assert src_exist_locs is None
    assert_equal(src_ids[dst_exist_locs].astype(np.int64).reshape((-1,)), new_src_ids)
    assert_equal(dst_ids[dst_exist_locs].astype(np.int64).reshape((-1,)), new_dst_ids)
    assert dst_exist_locs is not None

    # Test the case that none of the destination node IDs exists and we skip non exist edges.
    dst_ids = np.array([str(random.randint(20, 100)) for _ in range(15)])
    new_src_ids, new_dst_ids, _, _ = map_node_ids(src_ids, dst_ids, ("src", None, "dst"),
                                            id_map, True)
    assert len(new_src_ids) == 0
    assert len(new_dst_ids) == 0

def test_map_node_ids():
    # This tests all cases in map_node_ids.
    str_src_ids = np.array([str(i) for i in range(10)])
    str_dst_ids = np.array([str(i) for i in range(15)])
    id_map = {"src": IdMap(str_src_ids),
              "dst": IdMap(str_dst_ids)}
    check_map_node_ids_exist(str_src_ids, str_dst_ids, id_map)
    check_map_node_ids_src_not_exist(str_src_ids, str_dst_ids, id_map)
    check_map_node_ids_dst_not_exist(str_src_ids, str_dst_ids, id_map)

def test_merge_arrays():
    # This is to verify the correctness of ExtMemArrayMerger
    converters = [ExtMemArrayMerger(None, 0),
                  ExtMemArrayMerger("/tmp", 2)]
    for converter in converters:
        # Input are HDF5 arrays.
        data = {}
        handle, tmpfile = tempfile.mkstemp()
        os.close(handle)
        data["data1"] = np.random.rand(10, 3)
        data["data2"] = np.random.rand(9, 3)
        write_data_hdf5(data, tmpfile)
        data1 = read_data_hdf5(tmpfile, in_mem=False)
        arrs = [data1['data1'], data1['data2']]
        res = converter(arrs, "test1")
        assert isinstance(res, (np.ndarray, ExtMemArrayWrapper))
        np.testing.assert_array_equal(res, np.concatenate([data["data1"],
                                                           data["data2"]]))

        # One HDF5 array
        res = converter([data1['data1']], "test1.5")
        assert isinstance(res, (np.ndarray, ExtMemArrayWrapper))
        np.testing.assert_array_equal(res, data['data1'])

        os.remove(tmpfile)

        # Merge two arrays whose feature dimension is larger than 2.
        data1 = np.random.uniform(size=(1000, 10))
        data2 = np.random.uniform(size=(900, 10))
        em_arr = converter([data1, data2], "test2")
        assert isinstance(em_arr, (np.ndarray, ExtMemArrayWrapper))
        np.testing.assert_array_equal(np.concatenate([data1, data2]), em_arr)

        # Merge two arrays whose feature dimension is smaller than 2.
        data1 = np.random.uniform(size=(1000,))
        data2 = np.random.uniform(size=(900,))
        em_arr = converter([data1, data2], "test3")
        assert isinstance(em_arr, (np.ndarray, ExtMemArrayWrapper))
        np.testing.assert_array_equal(np.concatenate([data1, data2]), em_arr)

        # Input is an array whose feature dimension is larger than 2.
        data1 = np.random.uniform(size=(1000, 10))
        em_arr = converter([data1], "test4")
        assert isinstance(em_arr, (np.ndarray, ExtMemArrayWrapper))
        np.testing.assert_array_equal(data1, em_arr)

        # Input is an array whose feature dimension is smaller than 2.
        data1 = np.random.uniform(size=(1000,))
        em_arr = converter([data1], "test5")
        assert isinstance(em_arr, (np.ndarray, ExtMemArrayWrapper))
        np.testing.assert_array_equal(data1, em_arr)

@pytest.mark.parametrize("num_parts", [1, 2])
def test_partition_graph(num_parts):
    # This is to verify the correctness of partition_graph.
    # This function does some manual node/edge feature constructions for each partition.
    num_nodes = {'node1': 100,
                 'node2': 100,
                 'node3': 100}
    rel1_src = [np.random.randint(0, num_nodes['node1']//2, size=2500),
                np.random.randint(num_nodes['node1']//2, num_nodes['node1'], size=2500)]
    rel1_dst = [np.random.randint(0, num_nodes['node2']//2, size=2500),
                np.random.randint(num_nodes['node2']//2, num_nodes['node2'], size=2500)]
    rel2_src = [np.random.randint(0, num_nodes['node1']//2, size=2500),
                np.random.randint(num_nodes['node1']//2, num_nodes['node1'], size=2500)]
    rel2_dst = [np.random.randint(0, num_nodes['node3']//2, size=2500),
                np.random.randint(num_nodes['node3']//2, num_nodes['node3'], size=2500)]
    edges = {('node1', 'rel1', 'node2'): (np.concatenate(rel1_src), np.concatenate(rel1_dst)),
             ('node1', 'rel2', 'node3'): (np.concatenate(rel2_src), np.concatenate(rel2_dst))}
    g = dgl.heterograph(edges, num_nodes_dict=num_nodes)
    src, dst = g.edges(etype=('node1', 'rel1', 'node2'), order='srcdst')
    assert np.all(dst[src < 50].numpy() < 50)
    assert np.all(dst[src >= 50].numpy() >= 50)

    edge_data = {('node1', 'rel1', 'node2'): {'feat': np.random.uniform(size=(g.number_of_edges('rel1'), 10))}}
    node_data = {
            'node1': {'feat': np.random.uniform(size=(num_nodes['node1'], 10)), },
            'node2': { 'feat': np.random.uniform(size=(num_nodes['node2'],)), }
    }

    # Normal case: there are training nodes and test nodes.
    train_mask1 = np.zeros((num_nodes['node1'],), dtype=np.int8)
    train_mask1[0:num_nodes['node1']//3] = 1
    test_mask1 = np.zeros((num_nodes['node1'],), dtype=np.int8)
    test_mask1[num_nodes['node1']//3:num_nodes['node1']//2] = 1
    train_mask2 = np.zeros((num_nodes['node2'],), dtype=np.int8)
    train_mask2[0:num_nodes['node2']//3] = 1
    test_mask2 = np.zeros((num_nodes['node2'],), dtype=np.int8)
    test_mask2[num_nodes['node2']//3:num_nodes['node2']//2] = 1
    node_data['node1']['train_mask'] = train_mask1
    node_data['node1']['test_mask'] = test_mask1
    node_data['node2']['train_mask'] = train_mask2
    node_data['node2']['test_mask'] = test_mask2
    # verify that the graph partitioning can balance the training set.
    with tempfile.TemporaryDirectory() as tmpdirname:
        partition_graph(g, node_data, edge_data, 'test', num_parts, tmpdirname,
                        part_method="metis", save_mapping=True)
        for i in range(num_parts):
            part_dir = os.path.join(tmpdirname, "part" + str(i))
            node_data1 = dgl.data.utils.load_tensors(os.path.join(part_dir, 'node_feat.dgl'))
            assert th.sum(node_data1['node1/train_mask']) > 0
            assert th.sum(node_data1['node2/train_mask']) > 0
            assert th.sum(node_data1['node1/test_mask']) > 0
            assert th.sum(node_data1['node2/test_mask']) > 0

    # case 2: all are training nodes.
    train_mask1 = np.ones((num_nodes['node1'],), dtype=np.int8)
    test_mask1 = np.zeros((num_nodes['node1'],), dtype=np.int8)
    node_data['node1']['train_mask'] = train_mask1
    node_data['node1']['test_mask'] = test_mask1
    # verify that the graph partitioning can balance the training set.
    with tempfile.TemporaryDirectory() as tmpdirname:
        partition_graph(g, node_data, edge_data, 'test', num_parts, tmpdirname,
                        part_method="metis", save_mapping=True)
        for i in range(num_parts):
            part_dir = os.path.join(tmpdirname, "part" + str(i))
            node_data1 = dgl.data.utils.load_tensors(os.path.join(part_dir, 'node_feat.dgl'))
            assert th.sum(node_data1['node1/train_mask']) > 0
            assert th.sum(node_data1['node1/test_mask']) == 0

    # case 3: subset are training nodes but not test nodes.
    train_mask1 = np.zeros((num_nodes['node1'],), dtype=np.int8)
    train_mask1[0:num_nodes['node1']//3] = 1
    test_mask1 = np.zeros((num_nodes['node1'],), dtype=np.int8)
    node_data['node1']['train_mask'] = train_mask1
    node_data['node1']['test_mask'] = test_mask1
    # verify that the graph partitioning can balance the training set.
    with tempfile.TemporaryDirectory() as tmpdirname:
        partition_graph(g, node_data, edge_data, 'test', num_parts, tmpdirname,
                        part_method="metis", save_mapping=True)
        for i in range(num_parts):
            part_dir = os.path.join(tmpdirname, "part" + str(i))
            node_data1 = dgl.data.utils.load_tensors(os.path.join(part_dir, 'node_feat.dgl'))
            assert th.sum(node_data1['node1/train_mask']) > 0
            assert th.sum(node_data1['node1/test_mask']) == 0

    # case 4: subset are test nodes but not training nodes.
    train_mask1 = np.zeros((num_nodes['node1'],), dtype=np.int8)
    test_mask1 = np.zeros((num_nodes['node1'],), dtype=np.int8)
    test_mask1[0:num_nodes['node1']//3] = 1
    node_data['node1']['train_mask'] = train_mask1
    node_data['node1']['test_mask'] = test_mask1
    # verify that the graph partitioning can balance the training set.
    with tempfile.TemporaryDirectory() as tmpdirname:
        partition_graph(g, node_data, edge_data, 'test', num_parts, tmpdirname,
                        part_method="metis", save_mapping=True)
        for i in range(num_parts):
            part_dir = os.path.join(tmpdirname, "part" + str(i))
            node_data1 = dgl.data.utils.load_tensors(os.path.join(part_dir, 'node_feat.dgl'))
            assert th.sum(node_data1['node1/train_mask']) == 0
            assert th.sum(node_data1['node1/test_mask']) > 0

    # case 5: all nodes are either training nodes or test nodes.
    train_mask1 = np.zeros((num_nodes['node1'],), dtype=np.int8)
    test_mask1 = np.zeros((num_nodes['node1'],), dtype=np.int8)
    train_mask1[0:num_nodes['node1']//2] = 1
    test_mask1[num_nodes['node1']//2:] = 1
    node_data['node1']['train_mask'] = train_mask1
    node_data['node1']['test_mask'] = test_mask1
    # verify that the graph partitioning can balance the training set.
    with tempfile.TemporaryDirectory() as tmpdirname:
        partition_graph(g, node_data, edge_data, 'test', num_parts, tmpdirname,
                        part_method="metis", save_mapping=True)
        for i in range(num_parts):
            part_dir = os.path.join(tmpdirname, "part" + str(i))
            node_data1 = dgl.data.utils.load_tensors(os.path.join(part_dir, 'node_feat.dgl'))
            assert th.sum(node_data1['node1/train_mask']) > 0
            assert th.sum(node_data1['node1/test_mask']) > 0

    # case 6: all nodes are training nodes and test nodes.
    train_mask1 = np.ones((num_nodes['node1'],), dtype=np.int8)
    test_mask1 = np.ones((num_nodes['node1'],), dtype=np.int8)
    node_data['node1']['train_mask'] = train_mask1
    node_data['node1']['test_mask'] = test_mask1
    # verify that the graph partitioning will fail.
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            partition_graph(g, node_data, edge_data, 'test', num_parts, tmpdirname,
                            part_method="metis", save_mapping=True)
            assert False, "This should not work"
        except:
            pass

    # Partition the graph with our own partition_graph.
    dgl.random.seed(0)
    node_data1 = []
    edge_data1 = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        partition_graph(g, node_data, edge_data, 'test', num_parts, tmpdirname,
                        part_method="random", save_mapping=True)
        for i in range(num_parts):
            part_dir = os.path.join(tmpdirname, "part" + str(i))
            node_data1.append(dgl.data.utils.load_tensors(os.path.join(part_dir,
                                                                       'node_feat.dgl')))
            edge_data1.append(dgl.data.utils.load_tensors(os.path.join(part_dir,
                                                                       'edge_feat.dgl')))

        # Check saved node ID and edge ID mapping
        tmp_node_map_file = os.path.join(tmpdirname, f"node_mapping.pt")
        tmp_edge_map_file = os.path.join(tmpdirname, f"edge_mapping.pt")
        assert os.path.exists(tmp_node_map_file)
        assert os.path.exists(tmp_edge_map_file)
        node_id_map = th.load(tmp_node_map_file)
        edge_id_map = th.load(tmp_edge_map_file)
        assert len(node_id_map) == len(num_nodes)
        assert len(edge_id_map) == len(edges)
        for node_type, num_node in num_nodes.items():
            assert node_id_map[node_type].shape[0] == num_node
        for edge_type, edge in edges.items():
            assert edge_id_map[edge_type].shape[0] == edge[0].shape[0]

    # Partition the graph with DGL's partition_graph.
    g = dgl.heterograph(edges, num_nodes_dict=num_nodes)
    dgl.random.seed(0)
    node_data2 = []
    edge_data2 = []
    for ntype in node_data:
        for name in node_data[ntype]:
            g.nodes[ntype].data[name] = th.tensor(node_data[ntype][name])
    for etype in edge_data:
        for name in edge_data[etype]:
            g.edges[etype].data[name] = th.tensor(edge_data[etype][name])
    with tempfile.TemporaryDirectory() as tmpdirname:
        dgl.distributed.partition_graph(g, 'test', num_parts, out_path=tmpdirname,
                                        part_method='random')
        for i in range(num_parts):
            part_dir = os.path.join(tmpdirname, "part" + str(i))
            node_data2.append(dgl.data.utils.load_tensors(os.path.join(part_dir,
                                                                       'node_feat.dgl')))
            edge_data2.append(dgl.data.utils.load_tensors(os.path.join(part_dir,
                                                                       'edge_feat.dgl')))

    # Verify the correctness.
    for ndata1, ndata2 in zip(node_data1, node_data2):
        assert len(ndata1) == len(ndata2)
        for name in ndata1:
            assert name in ndata2
            np.testing.assert_array_equal(ndata1[name].numpy(), ndata2[name].numpy())
    for edata1, edata2 in zip(edge_data1, edge_data2):
        assert len(edata1) == len(edata2)
        for name in edata1:
            assert name in edata2
            np.testing.assert_array_equal(edata1[name].numpy(), edata2[name].numpy())

def test_multiprocessing_checks():
    # If the data are stored in multiple HDF5 files and there are
    # features and labels for processing.
    conf = {
        "format": {"name": "hdf5"},
        "features":     [
            {
                "feature_col":  "feat",
                "transform": {"name": 'tokenize_hf',
                    'bert_model': 'bert-base-uncased',
                    'max_seq_length': 16
                },
            },
        ],
        "labels":       [
            {
                "label_col":    "label",
                "task_type":    "classification",
            },
        ],
    }
    in_files = ["/tmp/test1", "/tmp/test2"]
    (feat_ops, _, _, _) = parse_feat_ops(conf['features'])
    label_ops = parse_label_ops(conf, is_node=True)
    multiprocessing = do_multiprocess_transform(conf, feat_ops, label_ops, in_files)
    assert multiprocessing == True

    # If the data are stored in multiple HDF5 files and there are
    # labels for processing.
    conf = {
        "format": {"name": "hdf5"},
        "labels":       [
            {
                "label_col":    "label",
                "task_type":    "classification",
            },
        ],
    }
    in_files = ["/tmp/test1", "/tmp/test2"]
    feat_ops = None
    label_ops = parse_label_ops(conf, is_node=True)
    multiprocessing = do_multiprocess_transform(conf, feat_ops, label_ops, in_files)
    assert multiprocessing == True

    # If the data are stored in multiple HDF5 files and there are
    # features for processing.
    conf = {
        "format": {"name": "hdf5"},
        "features":     [
            {
                "feature_col":  "feat",
                "transform": {"name": 'tokenize_hf',
                    'bert_model': 'bert-base-uncased',
                    'max_seq_length': 16
                },
            },
        ],
    }
    in_files = ["/tmp/test1", "/tmp/test2"]
    (feat_ops, _, _, _) = parse_feat_ops(conf['features'])
    label_ops = None
    multiprocessing = do_multiprocess_transform(conf, feat_ops, label_ops, in_files)
    assert multiprocessing == True

    # If the data are stored in a single HDF5 file and there are
    # features for processing.
    in_files = ["/tmp/test1"]
    (feat_ops, _, _, _) = parse_feat_ops(conf['features'])
    label_ops = None
    multiprocessing = do_multiprocess_transform(conf, feat_ops, label_ops, in_files)
    assert multiprocessing == False

    # If the data are stored in multiple HDF5 files and there are
    # features that don't require processing.
    conf = {
        "format": {"name": "hdf5"},
        "features":     [
            {
                "feature_col":  "feat",
            },
        ],
    }
    in_files = ["/tmp/test1", "/tmp/test2"]
    (feat_ops, _, _, _) = parse_feat_ops(conf['features'])
    label_ops = None
    multiprocessing = do_multiprocess_transform(conf, feat_ops, label_ops, in_files)
    assert multiprocessing == False

    # If the data are stored in multiple parquet files and there are
    # features that don't require processing.
    conf = {
        "format": {"name": "parquet"},
        "features":     [
            {
                "feature_col":  "feat",
            },
        ],
    }
    in_files = ["/tmp/test1", "/tmp/test2"]
    (feat_ops, _, _, _) = parse_feat_ops(conf['features'])
    label_ops = None
    multiprocessing = do_multiprocess_transform(conf, feat_ops, label_ops, in_files)
    assert multiprocessing == True

    # If the data are stored in a single parquet file and there are
    # features that don't require processing.
    in_files = ["/tmp/test1"]
    (feat_ops, _, _, _) = parse_feat_ops(conf['features'])
    label_ops = None
    multiprocessing = do_multiprocess_transform(conf, feat_ops, label_ops, in_files)
    assert multiprocessing == False

def test_parse_edge_data():
    with tempfile.TemporaryDirectory() as tmpdirname:
        str_src_ids = np.array([str(i) for i in range(10)])
        str_dst_ids = np.array([str(i) for i in range(15)])
        node_id_map = {"src": IdMap(str_src_ids),
                       "dst": IdMap(str_dst_ids)}

        src_ids = np.array([str(random.randint(0, 20)) for _ in range(15)])
        dst_ids = np.array([str(random.randint(0, 25)) for _ in range(15)])
        feat = np.random.rand(15, 10)
        data = {
            "src_id": src_ids,
            "dst_id": dst_ids,
            "feat": feat,
        }

        feat_ops = [Noop("feat", "feat", None)]
        label_ops = [
            LinkPredictionProcessor(None, None, [0.7,0.1,0.2], None)]
        data_file = os.path.join(tmpdirname, "data.parquet")
        write_data_parquet(data, data_file)

        conf = {
            "source_id_col": "src_id",
            "dest_id_col": "dst_id",
            "relation": ("src", "rel", "dst")
        }
        keys = ["src_id", "dst_id", "feat"]
        src_ids, dst_ids, feat_data = \
            parse_edge_data(data_file, feat_ops, label_ops, node_id_map,
                            partial(read_data_parquet, data_fields=keys),
                            conf, skip_nonexist_edges=True)
        for _, val in feat_data.items():
            assert len(src_ids) == len(val)
            assert len(dst_ids) == len(val)

        assert "feat" in feat_data
        assert "train_mask" in feat_data
        assert "val_mask" in feat_data
        assert "test_mask" in feat_data

        # Test with customized mask names.
        try:
            label_ops = [
            LinkPredictionProcessor(None, None, [0.7,0.1,0.2], None,
                                    mask_field_names="train_mask")]
            assert False, \
                "Should raise an exception as mask_field_names is in the wrong format."
        except:
            pass

        try:
            label_ops = [
            LinkPredictionProcessor(None, None, [0.7,0.1,0.2], None,
                                    mask_field_names=("tm", "vm"))]
            assert False, \
                "Should raise an exception as mask_field_names is in the wrong format."
        except:
            pass

        label_ops = [
            LinkPredictionProcessor(None, None, [0.7,0.1,0.2], None,
                                    mask_field_names=("tm", "vm", "tsm"))]

        conf = {
            "source_id_col": "src_id",
            "dest_id_col": "dst_id",
            "relation": ("src", "rel", "dst")
        }
        keys = ["src_id", "dst_id", "feat"]
        src_ids, dst_ids, feat_data = \
            parse_edge_data(data_file, feat_ops, label_ops, node_id_map,
                            partial(read_data_parquet, data_fields=keys),
                            conf, skip_nonexist_edges=True)
        for _, val in feat_data.items():
            assert len(src_ids) == len(val)
            assert len(dst_ids) == len(val)

        assert "feat" in feat_data
        assert "tm" in feat_data
        assert "vm" in feat_data
        assert "tsm" in feat_data

@pytest.mark.parametrize("ext_mem_path", [None, "/tmp/"])
def test_multicolumn(ext_mem_path):
    # Just get the features without transformation.
    feat_op1 = [{
        "feature_col": ["test1", "test2"],
        "feature_name": "test3",
    }]
    (res, _, _, _) = parse_feat_ops(feat_op1)
    assert len(res) == 1
    assert res[0].col_name == feat_op1[0]["feature_col"]
    assert res[0].feat_name == feat_op1[0]["feature_name"]
    assert isinstance(res[0], Noop)

    data = {
        "test1": np.random.rand(4, 2),
        "test2": np.random.rand(4, 2)
    }
    data["test3"] = np.column_stack((data['test1'], data['test2']))
    proc_res = process_features(data, res, ext_mem_path=ext_mem_path)
    assert "test3" in proc_res
    assert proc_res["test3"].dtype == np.float32
    if isinstance(proc_res, ExtMemArrayWrapper):
        proc_res = proc_res.to_numpy()
    np.testing.assert_allclose(proc_res["test3"], data["test3"])

    # test on multiple column bucket feature transformation
    feat_op2 = [{
        "feature_col": ["test1", "test2"],
        "feature_name": "test3",
        "transform":{
            "name": "bucket_numerical",
            "range": [10, 200],
            "bucket_cnt": 10
        }
    }]
    data = {
        "test1": np.random.randint(0, 100, 3),
        "test2": np.random.randint(0, 100, 3)
    }
    (res, _, _, _) = parse_feat_ops(feat_op2)
    assert len(res) == 1
    assert res[0].col_name == feat_op2[0]["feature_col"]
    assert res[0].feat_name == feat_op2[0]["feature_name"]
    assert isinstance(res[0], BucketTransform)
    bucket_feats = process_features(data, res, ext_mem_path=ext_mem_path)
    assert "test3" in proc_res
    assert proc_res["test3"].dtype == np.float32

    data_bucket1 = {
        "test1": data["test1"]
    }
    feat_bucket_single1 = [{
        "feature_col": "test1",
        "feature_name": "test3",
        "transform":{
            "name": "bucket_numerical",
            "range": [10, 200],
            "bucket_cnt": 10
        }
    }]
    (res, _, _, _) = parse_feat_ops(feat_bucket_single1)
    bucket_feat_single1 = process_features(data_bucket1, res)

    data_bucket2 = {
        "test2": data["test2"]
    }
    feat_bucket_single2 = [{
        "feature_col": "test2",
        "feature_name": "test3",
        "transform":{
            "name": "bucket_numerical",
            "range": [10, 200],
            "bucket_cnt": 10
        }
    }]
    (res, _, _, _) = parse_feat_ops(feat_bucket_single2)
    bucket_feat_single2 = process_features(data_bucket2, res)
    bucket_expec = np.column_stack((bucket_feat_single1["test3"],
                                    bucket_feat_single2["test3"]))
    if isinstance(proc_res, ExtMemArrayWrapper):
        bucket_feats = bucket_feats.to_numpy()
    assert_equal(bucket_feats["test3"], bucket_expec)

    # test on multiple column rank-gauss feature transformation
    feat_op3 = [{
        "feature_col": ["test1", "test2"],
        "feature_name": "test3",
        "transform":{
            "name": "rank_gauss"
        }
    }]
    data = {
        "test1": np.random.randint(0, 100, 3),
        "test2": np.random.randint(0, 100, 3)
    }
    (res, _, _, _) = parse_feat_ops(feat_op3)
    assert len(res) == 1
    assert res[0].col_name == feat_op3[0]["feature_col"]
    assert res[0].feat_name == feat_op3[0]["feature_name"]
    assert isinstance(res[0], RankGaussTransform)
    rg_feats = process_features(data, res, ext_mem_path=ext_mem_path)
    assert "test3" in proc_res
    assert proc_res["test3"].dtype == np.float32

    data_rg1 = {
        "test1": data["test1"]
    }
    feat_rg_single1 = [{
        "feature_col": "test1",
        "feature_name": "test3",
        "transform":{
            "name": "rank_gauss",
            "epsilon": 1e-7
        }
    }]
    (res, _, _, _) = parse_feat_ops(feat_rg_single1)
    rg_feat_single1 = process_features(data_rg1, res)

    data_rg2 = {
        "test2": data["test2"]
    }
    feat_rg_single2 = [{
        "feature_col": "test2",
        "feature_name": "test3",
        "transform":{
            "name": "rank_gauss",
            "uniquify": True
        }
    }]
    (res, _, _, _) = parse_feat_ops(feat_rg_single2)
    rg_feat_single2 = process_features(data_rg2, res)
    rg_expec = np.column_stack((rg_feat_single1["test3"],
                                rg_feat_single2["test3"]))
    if isinstance(rg_feats, ExtMemArrayWrapper):
        rg_feats = rg_feats.to_numpy()
    assert_equal(rg_feats["test3"], rg_expec)

    # test on multiple column bert_hf feature transformation
    feat_op4 = [{
        "feature_col": ["test1", "test2"],
        "feature_name": "test3",
        "transform": {"name": 'bert_hf',
                      'bert_model': 'bert-base-uncased',
                      'max_seq_length': 16
                      },
    }]

    data = {
        "test1": ["test", "haha", "failure"],
        "test2": ["never", "pass", "lint"]
    }
    (res, _, _, _) = parse_feat_ops(feat_op4)
    assert len(res) == 1
    assert res[0].col_name == feat_op4[0]["feature_col"]
    assert res[0].feat_name == feat_op4[0]["feature_name"]
    assert isinstance(res[0], Text2BERT)
    bert_feats = process_features(data, res, ext_mem_path=ext_mem_path)
    assert "test3" in proc_res

    data_bert1 = {
        "test1": data["test1"]
    }
    feat_bert_single1 = [{
        "feature_col": "test1",
        "feature_name": "test3",
        "transform": {"name": 'bert_hf',
                      'bert_model': 'bert-base-uncased',
                      'max_seq_length': 16
                      },
    }]
    (res, _, _, _) = parse_feat_ops(feat_bert_single1)
    bert_feat_single1 = process_features(data_bert1, res)

    data_bert2 = {
        "test2": data["test2"]
    }
    feat_bert_single2 = [{
        "feature_col": "test2",
        "feature_name": "test3",
        "transform": {"name": 'bert_hf',
                      'bert_model': 'bert-base-uncased',
                      'max_seq_length': 16
                      },
    }]
    (res, _, _, _) = parse_feat_ops(feat_bert_single2)
    bert_feat_single2 = process_features(data_bert2, res)
    bert_expec = np.column_stack((bert_feat_single1["test3"],
                                bert_feat_single2["test3"]))
    if isinstance(bert_feats, ExtMemArrayWrapper):
        bert_feats = bert_feats.to_numpy()
    assert_equal(bert_feats["test3"], bert_expec)

    # test on max_min_norm with specifying max_val and min_val
    feat_op5 = [{
        "feature_col": ["test1", "test2"],
        "feature_name": "test3",
        "transform":{
            "name": "max_min_norm",
            "max_val": 100,
            "min_val": 0
        }
    }]
    data = {
        "test1": np.random.randint(0, 100, 3),
        "test2": np.random.randint(0, 100, 3)
    }
    (res, _, _, _) = parse_feat_ops(feat_op5)
    assert len(res) == 1
    assert res[0].col_name == feat_op5[0]["feature_col"]
    assert res[0].feat_name == feat_op5[0]["feature_name"]
    assert isinstance(res[0], NumericalMinMaxTransform)
    maxmin_feats = process_features(data, res, ext_mem_path=ext_mem_path)
    assert "test3" in proc_res
    assert proc_res["test3"].dtype == np.float32

    data_maxmin1 = {
        "test1": data["test1"]
    }
    feat_maxmin_single1 = [{
        "feature_col": "test1",
        "feature_name": "test3",
        "transform":{
            "name": "max_min_norm",
            "max_val": 100,
            "min_val": 0
        }
    }]
    (res, _, _, _) = parse_feat_ops(feat_maxmin_single1)
    maxmin_feat_single1 = process_features(data_maxmin1, res)

    data_maxmin2 = {
        "test2": data["test2"]
    }
    feat_maxmin_single2 = [{
        "feature_col": "test2",
        "feature_name": "test3",
        "transform":{
            "name": "max_min_norm",
            "max_val": 100,
            "min_val": 0
        }
    }]
    (res, _, _, _) = parse_feat_ops(feat_maxmin_single2)
    maxmin_feat_single2 = process_features(data_maxmin2, res)
    maxmin_expec = np.column_stack((maxmin_feat_single1["test3"],
                                maxmin_feat_single2["test3"]))
    if isinstance(maxmin_feats, ExtMemArrayWrapper):
        maxmin_feats = maxmin_feats.to_numpy()
    assert_equal(maxmin_feats["test3"], maxmin_expec)

    # test on max_min_norm without specifying max_val and min_val,
    # expect to throw error
    feat_op6 = [{
        "feature_col": ["test1", "test2"],
        "feature_name": "test3",
        "transform":{
            "name": "max_min_norm",
        }
    }]
    try:
        (res, _, _, _) = parse_feat_ops(feat_op6)
        assert False, "expected Error raised for not " \
                      "specifying max_val and min_val"
    except AssertionError as e:
        assert str(e) == "max_val and min_val for max_min_norm " \
                         "feature transformation is needed"

    # test on to_categorical, expect to throw error
    feat_op7 = [{
        "feature_col": ["test1", "test2"],
        "feature_name": "test3",
        "transform":{
            "name": "to_categorical",
        }
    }]
    try:
        (res, _, _, _) = parse_feat_ops(feat_op7)
        assert False, "expected Error raised for multi column on" \
                      " categorical feature transformation"
    except RuntimeError as e:
        assert str(e) == "Do not support categorical feature " \
                         "transformation on multiple columns"

    # test on no input
    feat_op8 = [{
        "feature_col": [],
        "feature_name": "test3"
    }]
    try:
        (res, _, _, _) = parse_feat_ops(feat_op8)
        assert False, "expected Error raised for invalid feature column []"
    except AssertionError as e:
        assert str(e) == "feature column should not be empty"

    # test on no input column
    feat_op9 = [{
        "feature_col": "",
        "feature_name": "test3"
    }]
    try:
        (res, _, _, _) = parse_feat_ops(feat_op9)
        assert False, "expected Error not raised for invalid feature column """
    except AssertionError as e:
        assert str(e) == "feature column should not be empty"

    # tests on tokenize_hf, expect to throw error
    feat_op10 = [{
        "feature_col": ["test1", "test2"],
        "feature_name": "test3",
        "transform": {
            "name": 'tokenize_hf',
            "bert_model": 'bert-base-uncased',
            "max_seq_length": 16
            },
    }]
    try:
        (res, _, _, _) = parse_feat_ops(feat_op10)
        assert False, "expected Error raised for multi column on" \
                      " tokenize_hf feature transformation"
    except RuntimeError as e:
        assert str(e) == "Not support multiple column for tokenize_hf transformation"

    feat_op11 = [{
        "feature_col": ["test1", "test2", "test3"],
        "feature_name": "test4",
    }]
    (res, _, _, _) = parse_feat_ops(feat_op11)
    assert len(res) == 1
    assert res[0].col_name == feat_op11[0]["feature_col"]
    assert res[0].feat_name == feat_op11[0]["feature_name"]
    assert isinstance(res[0], Noop)

    # tests on more than 2 columns
    data = {
        "test1": np.random.rand(4, 2),
        "test2": np.random.rand(4, 2),
        "test3": np.random.rand(4, 2)
    }
    data["test4"] = np.column_stack((data['test1'], data['test2'], data['test3']))
    proc_res = process_features(data, res, ext_mem_path=ext_mem_path)
    assert "test4" in proc_res
    assert proc_res["test4"].dtype == np.float32
    if isinstance(proc_res, ExtMemArrayWrapper):
        proc_res = proc_res.to_numpy()
    np.testing.assert_allclose(proc_res["test4"], data["test4"])


def test_feature_wrapper():
    # same array size
    data = {
        "test1": np.random.randint(0, 100, (3, 1)),
        "test2": np.random.randint(0, 100, (3, 1))
    }
    if not os.path.exists("/tmp/featurewrapper"):
        os.makedirs("/tmp/featurewrapper")
    test_extfeature_wrapper = ExtFeatureWrapper("/tmp/featurewrapper")
    try:
        a = test_extfeature_wrapper[:]
    except RuntimeError as e:
        assert str(e) == "Call ExtFeatureWrapper.merge() first before calling __getitem__"

    try:
        a = test_extfeature_wrapper.to_numpy()
    except RuntimeError as e:
        assert str(e) == "Call ExtFeatureWrapper.merge() first before calling to_numpy()"

    test_extfeature_wrapper.append(data["test1"])
    file_count = sum(1 for name in os.listdir("/tmp/featurewrapper")
                     if os.path.isfile(os.path.join("/tmp/featurewrapper", name)))
    assert file_count == 1
    test_extfeature_wrapper.append(data["test2"])
    file_count = sum(1 for name in os.listdir("/tmp/featurewrapper")
                     if os.path.isfile(os.path.join("/tmp/featurewrapper", name)))
    assert file_count == 2
    test_extfeature_wrapper.merge()
    data["test3"] = np.column_stack((data['test1'], data['test2']))
    file_count = sum(1 for name in os.listdir("/tmp/featurewrapper")
                     if os.path.isfile(os.path.join("/tmp/featurewrapper", name)))
    assert file_count == 3
    np.testing.assert_allclose(test_extfeature_wrapper[:], data["test3"])

    # different array size
    data = {
        "test1": np.random.randint(0, 100, (3, 7)),
        "test2": np.random.randint(0, 100, (3, 1))
    }
    if not os.path.exists("/tmp/featurewrapper2"):
        os.makedirs("/tmp/featurewrapper2")
    test_extfeature_wrapper = ExtFeatureWrapper("/tmp/featurewrapper2")

    test_extfeature_wrapper.append(data["test1"])
    file_count = sum(1 for name in os.listdir("/tmp/featurewrapper2")
                     if os.path.isfile(os.path.join("/tmp/featurewrapper2", name)))
    assert file_count == 1
    test_extfeature_wrapper.append(data["test2"])
    file_count = sum(1 for name in os.listdir("/tmp/featurewrapper2")
                     if os.path.isfile(os.path.join("/tmp/featurewrapper2", name)))
    assert file_count == 2
    test_extfeature_wrapper.merge()
    file_count = sum(1 for name in os.listdir("/tmp/featurewrapper2")
                     if os.path.isfile(os.path.join("/tmp/featurewrapper2", name)))
    assert file_count == 3
    data["test3"] = np.column_stack((data['test1'], data['test2']))
    np.testing.assert_allclose(test_extfeature_wrapper[:], data["test3"])
    test_extfeature_wrapper = None
    test_gc()


def test_gc():
    assert not os.path.isdir("/tmp/featurewrapper"), \
        "Directory /tmp/featurewrapper should not exist after gc"
    assert not os.path.isdir("/tmp/featurewrapper2"), \
        "Directory /tmp/featurewrapper2 should not exist after gc"

def test_parse_feat_ops_data_format():
    conf = {
        "format": {"name": "hdf5"},
        "features":     [
            {
                "feature_col":  "feat",
                "transform": {"name": 'edge_dst_hard_negative'},
            },
            {
                "feature_col":  "feat2",
                "transform": {"name": 'max_min_norm'},
            },
            {
                "feature_col":  "feat3",
                "transform": {"name": 'rank_gauss'},
            },
        ],
    }
    try:
        (feat_ops, _, _, _) = parse_feat_ops(conf['features'], conf['format']['name'])
        assert False, "edge_dst_hard_negative can not work with hdf5."
    except AssertionError as e:
        assert str(e) == "Edge_dst_hard_negative transformation does not work with hdf5 inputs."

    conf['format']['name'] = "parquet"
    (feat_ops, two_phase_feat_ops, after_merge_feat_ops, hard_edge_neg_ops) = \
        parse_feat_ops(conf['features'], conf['format']['name'])
    assert len(feat_ops) == 3
    assert len(two_phase_feat_ops) == 2 # max_min_norm and edge_dst_hard_negative
    assert len(after_merge_feat_ops) == 1 # rank_gauss
    assert len(hard_edge_neg_ops) == 1 # edge_dst_hard_negative

    conf['format']['name'] = "csv"
    (feat_ops, two_phase_feat_ops, after_merge_feat_ops, hard_edge_neg_ops) = \
        parse_feat_ops(conf['features'], conf['format']['name'])
    assert len(feat_ops) == 3
    assert len(two_phase_feat_ops) == 2 # max_min_norm and edge_dst_hard_negative
    assert len(after_merge_feat_ops) == 1 # rank_gauss
    assert len(hard_edge_neg_ops) == 1 # edge_dst_hard_negative



def test_homogeneous():
    # single node type and edge type input
    conf = {
        "version": "gconstruct-v0.1", "nodes": [
            {"node_id_col": "id", "node_type": "movie", "format": {"name": "parquet"},
             "files": "/data/ml-100k/movie.parquet", "features": [
                {"feature_col": "title", "transform": {
                    "name": "bert_hf", "bert_model": "bert-base-uncased", "max_seq_length": 16}}],
             "labels": [{"label_col": "label", "task_type": "classification", "split_pct": [0.8, 0.1, 0.1]}]}],
        "edges": [
            {"source_id_col": "src_id", "dest_id_col": "dst_id", "relation": ["movie", "rating", "movie"],
             "format": {"name": "parquet"}, "files": "/data/ml-100k/edges_homo.parquet", "labels": [
                {"label_col": "rate", "task_type": "classification", "split_pct": [0.1, 0.1, 0.1]}]}]
    }
    assert is_homogeneous(conf)
    verify_confs(conf)
    assert conf['nodes'][0]["node_type"] == "_N"
    assert conf['edges'][0]['relation'] == ["_N", "_E", "_N"]
    conf["edges"][0]["relation"] = ["movie_fake", "rating", "movie"]
    conf["nodes"].append(copy.deepcopy(conf["nodes"][0]))
    conf["nodes"][0]["node_type"] = "movie"
    conf["nodes"][1]["node_type"] = "movie_fake"
    assert not is_homogeneous(conf)


    # multiple node types and edge types input
    conf = {
        "version": "gconstruct-v0.1", "nodes": [
            {"node_id_col": "id", "node_type": "movie", "format": {"name": "parquet"},
             "files": "/data/ml-100k/movie.parquet", "features": [
                {"feature_col": "title", "transform": {
                    "name": "bert_hf", "bert_model": "bert-base-uncased", "max_seq_length": 16}}],
             "labels": [{"label_col": "label", "task_type": "classification", "split_pct": [0.8, 0.1, 0.1]}]},
            {"node_type": "movie", "format": {"name": "parquet"}, "files": "/data/ml-100k/movie.parquet",
             "features": [{"feature_col": "id"}]}],
        "edges": [
            {"source_id_col": "src_id", "dest_id_col": "dst_id", "relation": ["movie", "rating", "movie"],
             "format": {"name": "parquet"}, "files": "/data/ml-100k/edges_homo.parquet", "labels": [
                {"label_col": "rate", "task_type": "classification", "split_pct": [0.1, 0.1, 0.1]}]},
            {"relation": ["movie", "rating", "movie"], "format": {"name": "parquet"},
             "files": "/data/ml-100k/edges_homo.parquet"}]
    }
    assert is_homogeneous(conf)
    verify_confs(conf)
    assert conf['nodes'][0]["node_type"] == "_N"
    assert conf['edges'][0]['relation'] == ["_N", "_E", "_N"]
    conf["edges"][0]["relation"] = ["movie_fake", "rating", "movie"]
    conf["nodes"].append(copy.deepcopy(conf["nodes"][0]))
    conf["nodes"][0]["node_type"] = "movie"
    conf["nodes"][1]["node_type"] = "movie_fake"
    assert not is_homogeneous(conf)

if __name__ == '__main__':
    test_multitask_label()
    test_partition_graph(2)
    test_parse_feat_ops_data_format()
    test_parse_edge_data()
    test_multiprocessing_checks()
    test_csv(None)
    test_hdf5()
    test_json()
    test_merge_arrays()
    test_map_node_ids()
    test_id_map()
    test_id_reverse_map()
    test_parquet()
    test_feat_ops()
    test_process_features()
    test_process_features_fp16()
    test_label()
    test_multicolumn(None)
    test_multicolumn("/tmp/")
    test_feature_wrapper()
