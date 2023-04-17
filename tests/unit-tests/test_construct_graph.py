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
import numpy as np
import graphstorm as gs
import dgl
import torch as th

from graphstorm.gconstruct.file_io import write_data_parquet, read_data_parquet
from graphstorm.gconstruct.file_io import write_data_json, read_data_json
from graphstorm.gconstruct.transform import parse_feat_ops, process_features
from graphstorm.gconstruct.transform import parse_label_ops, process_labels
from graphstorm.gconstruct.transform import Noop
from graphstorm.gconstruct.id_map import IdMap, map_node_ids
from graphstorm.gconstruct.utils import ExtMemArrayConverter, partition_graph

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

def test_json():
    handle, tmpfile = tempfile.mkstemp()
    os.close(handle)

    data = {}
    data["data1"] = np.random.rand(10, 3)
    data["data2"] = np.random.rand(10)
    write_data_json(data, tmpfile)
    data1 = read_data_json(tmpfile, ["data1", "data2"])
    assert len(data1) == 2
    assert "data1" in data1
    assert "data2" in data1
    assert np.all(data1['data1'] == data['data1'])
    assert np.all(data1['data2'] == data['data2'])

    # Test the case that some field doesn't exist.
    try:
        data1 = read_data_json(tmpfile, ["data1", "data3"])
        assert False, "This shouldn't happen"
    except:
        pass

    os.remove(tmpfile)

def test_feat_ops():
    # Just get the features without transformation.
    feat_op1 = [{
        "feature_col": "test1",
        "feature_name": "test2",
    }]
    res1 = parse_feat_ops(feat_op1)
    assert len(res1) == 1
    assert res1[0].col_name == feat_op1[0]["feature_col"]
    assert res1[0].feat_name == feat_op1[0]["feature_name"]
    assert isinstance(res1[0], Noop)

    # When the feature name is not specified.
    feat_op1 = [{
        "feature_col": "test1",
    }]
    res1 = parse_feat_ops(feat_op1)
    assert len(res1) == 1
    assert res1[0].col_name == feat_op1[0]["feature_col"]
    assert res1[0].feat_name == feat_op1[0]["feature_col"]
    assert isinstance(res1[0], Noop)

    # Test more complex cases.
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
    res2 = parse_feat_ops(feat_op2)
    assert len(res2) == 2
    assert res2[1].col_name == feat_op2[1]["feature_col"]
    assert res2[1].feat_name == feat_op2[1]["feature_name"]
    op = res2[1]
    tokens = op(["hello world", "hello world"])
    assert len(tokens) == 3
    assert tokens['test4_token_ids'].shape == (2, 16)
    assert tokens['test4_attention_mask'].shape == (2, 16)
    assert tokens['test4_token_type_ids'].shape == (2, 16)
    np.testing.assert_array_equal(tokens['test4_token_ids'][0],
                                  tokens['test4_token_ids'][1])
    np.testing.assert_array_equal(tokens['test4_attention_mask'][0],
                                  tokens['test4_attention_mask'][1])
    np.testing.assert_array_equal(tokens['test4_token_type_ids'][0],
                                  tokens['test4_token_type_ids'][1])

    data = {
        "test1": np.random.rand(2, 4),
        "test3": ["hello world", "hello world"],
    }
    proc_res = process_features(data, res2)
    np.testing.assert_array_equal(data['test1'], proc_res['test2'])
    assert "test4_token_ids" in proc_res
    assert "test4_attention_mask" in proc_res
    assert "test4_token_type_ids" in proc_res

def verify_split(res, label):
    assert len(res) == 4
    assert 'label' in res
    assert 'label_train_mask' in res
    assert 'label_val_mask' in res
    assert 'label_test_mask' in res
    assert res['label_train_mask'].shape == (len(label),)
    assert res['label_val_mask'].shape == (len(label),)
    assert res['label_test_mask'].shape == (len(label),)
    assert np.sum(res['label_train_mask']) == 8
    assert np.sum(res['label_val_mask']) == 1
    assert np.sum(res['label_test_mask']) == 1
    assert np.sum(res['label_train_mask'] + res['label_val_mask'] \
            + res['label_test_mask']) == 10

def verify_integer(label, res):
    train_mask = res['label_train_mask'] == 1
    val_mask = res['label_val_mask'] == 1
    test_mask = res['label_test_mask'] == 1
    assert np.all(np.logical_and(label[train_mask] >= 0, label[train_mask] <= 10))
    assert np.all(np.logical_and(label[val_mask] >= 0, label[val_mask] <= 10))
    assert np.all(np.logical_and(label[test_mask] >= 0, label[test_mask] <= 10))

def check_classification():
    def verify_classification(res):
        verify_split(res, data['label'])
        assert np.issubdtype(res['label'].dtype, np.integer)
        verify_integer(res['label'], res)

    # Check classification
    conf = {'task_type': 'classification',
            'label_col': 'label',
            'split_pct': [0.8, 0.1, 0.1]}
    ops = parse_label_ops([conf], True)
    data = {'label' : np.random.uniform(size=10) * 10}
    res = process_labels(data, ops)
    verify_classification(res)
    ops = parse_label_ops([conf], False)
    res = process_labels(data, ops)
    verify_classification(res)

    # Check classification with invalid labels.
    data = {'label' : np.random.uniform(size=13) * 10}
    data['label'][[0, 3, 4]] = np.NAN
    ops = parse_label_ops([conf], True)
    res = process_labels(data, ops)
    verify_classification(res)

    # Check classification with integer labels.
    data = {'label' : np.random.randint(10, size=10)}
    ops = parse_label_ops([conf], True)
    res = process_labels(data, ops)
    verify_classification(res)

    # Check classification with integer labels.
    # Data split doesn't use all labeled samples.
    conf = {'task_type': 'classification',
            'label_col': 'label',
            'split_pct': [0.4, 0.05, 0.05]}
    ops = parse_label_ops([conf], True)
    data = {'label' : np.random.randint(3, size=20)}
    res = process_labels(data, ops)
    verify_classification(res)

    # split_pct is not specified.
    conf = {'task_type': 'classification',
            'label_col': 'label'}
    ops = parse_label_ops([conf], True)
    data = {'label' : np.random.randint(3, size=20)}
    res = process_labels(data, ops)
    assert np.sum(res['label_train_mask']) == 20
    assert np.sum(res['label_val_mask']) == 0
    assert np.sum(res['label_test_mask']) == 0

def check_multilabel_classification():
    def verify_classification(res):
        verify_split(res, data['label'])
        assert np.issubdtype(res['label'].dtype, np.integer)

    conf = {'task_type': 'classification',
            'label_col': 'label',
            'split_pct': [0.8, 0.1, 0.1]}
    ops = parse_label_ops([conf], True)
    data = {'label' : np.random.uniform(size=(10, 5)) * 10}
    res = process_labels(data, ops)
    verify_classification(res)
    ops = parse_label_ops([conf], False)
    res = process_labels(data, ops)
    verify_classification(res)

    # Check classification with invalid labels.
    data = {'label' : np.random.uniform(size=(13, 5)) * 10}
    data['label'][[0, 3, 4],:] = np.NAN
    data['label'][1, 1] = np.NAN
    data['label'][2, 2] = np.NAN
    ops = parse_label_ops([conf], True)
    res = process_labels(data, ops)
    verify_classification(res)

def check_regression():
    conf = {'task_type': 'regression',
            'label_col': 'label',
            'split_pct': [0.8, 0.1, 0.1]}
    ops = parse_label_ops([conf], True)
    data = {'label' : np.random.uniform(size=10) * 10}
    res = process_labels(data, ops)
    def verify_regression(res):
        verify_split(res, data['label'])
    verify_regression(res)
    ops = parse_label_ops([conf], False)
    res = process_labels(data, ops)
    verify_regression(res)

    # Check regression with invalid labels.
    data = {'label' : np.random.uniform(size=13) * 10}
    data['label'][[0, 3, 4]] = np.NAN
    ops = parse_label_ops([conf], True)
    res = process_labels(data, ops)
    verify_regression(res)

def check_link_prediction():
    # Check link prediction
    conf = {'task_type': 'link_prediction',
            'split_pct': [0.8, 0.1, 0.1]}
    ops = parse_label_ops([conf], False)
    data = {'label' : np.random.uniform(size=10) * 10}
    res = process_labels(data, ops)
    assert len(res) == 3
    assert 'train_mask' in res
    assert 'val_mask' in res
    assert 'test_mask' in res
    assert np.sum(res['train_mask']) == 8
    assert np.sum(res['val_mask']) == 1
    assert np.sum(res['test_mask']) == 1

def test_label():
    check_classification()
    check_multilabel_classification()
    check_regression()
    check_link_prediction()

def check_id_map_exist(id_map, str_ids):
    # Test the case that all Ids exist in the map.
    rand_ids = np.array([str(random.randint(0, len(str_ids)) % len(str_ids)) for _ in range(5)])
    remap_ids, idx = id_map.map_id(rand_ids)
    assert len(idx) == len(rand_ids)
    assert np.issubdtype(remap_ids.dtype, np.integer)
    assert len(remap_ids) == len(rand_ids)
    for id1, id2 in zip(remap_ids, rand_ids):
        assert id1 == int(id2)

def check_id_map_not_exist(id_map, str_ids):
    # Test the case that some of the Ids don't exist.
    rand_ids = np.array([str(random.randint(0, len(str_ids)) % len(str_ids)) for _ in range(5)])
    rand_ids1 = np.concatenate([rand_ids, np.array(["11", "15", "20"])])
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
    check_id_map_not_exist(id_map, str_ids)
    check_id_map_dtype_not_match(id_map, str_ids)

def check_map_node_ids_exist(str_src_ids, str_dst_ids, id_map):
    # Test the case that both source node IDs and destination node IDs exist.
    src_ids = np.array([str(random.randint(0, len(str_src_ids) - 1)) for _ in range(15)])
    dst_ids = np.array([str(random.randint(0, len(str_dst_ids) - 1)) for _ in range(15)])
    new_src_ids, new_dst_ids = map_node_ids(src_ids, dst_ids, ("src", None, "dst"),
                                            id_map, False)
    assert len(new_src_ids) == len(src_ids)
    assert len(new_dst_ids) == len(dst_ids)
    for src_id1, src_id2 in zip(new_src_ids, src_ids):
        assert src_id1 == int(src_id2)
    for dst_id1, dst_id2 in zip(new_dst_ids, dst_ids):
        assert dst_id1 == int(dst_id2)

def check_map_node_ids_src_not_exist(str_src_ids, str_dst_ids, id_map):
    # Test the case that source node IDs don't exist.
    src_ids = np.array([str(random.randint(0, 20)) for _ in range(15)])
    dst_ids = np.array([str(random.randint(0, len(str_dst_ids) - 1)) for _ in range(15)])
    try:
        new_src_ids, new_dst_ids = map_node_ids(src_ids, dst_ids, ("src", None, "dst"),
                                                id_map, False)
        raise ValueError("fail")
    except:
        pass
    # Test the case that source node IDs don't exist and we skip non exist edges.
    new_src_ids, new_dst_ids = map_node_ids(src_ids, dst_ids, ("src", None, "dst"),
                                            id_map, True)
    num_valid = sum([int(id_) < len(str_src_ids) for id_ in src_ids])
    assert len(new_src_ids) == num_valid
    assert len(new_dst_ids) == num_valid

def check_map_node_ids_dst_not_exist(str_src_ids, str_dst_ids, id_map):
    # Test the case that destination node IDs don't exist.
    src_ids = np.array([str(random.randint(0, len(str_src_ids) - 1)) for _ in range(15)])
    dst_ids = np.array([str(random.randint(0, 20)) for _ in range(15)])
    try:
        new_src_ids, new_dst_ids = map_node_ids(src_ids, dst_ids, ("src", None, "dst"),
                                                id_map, False)
        raise ValueError("fail")
    except:
        pass
    # Test the case that destination node IDs don't exist and we skip non exist edges.
    new_src_ids, new_dst_ids = map_node_ids(src_ids, dst_ids, ("src", None, "dst"),
                                            id_map, True)
    num_valid = sum([int(id_) < len(str_dst_ids) for id_ in dst_ids])
    assert len(new_src_ids) == num_valid
    assert len(new_dst_ids) == num_valid

def test_map_node_ids():
    # This tests all cases in map_node_ids.
    str_src_ids = np.array([str(i) for i in range(10)])
    str_dst_ids = np.array([str(i) for i in range(15)])
    id_map = {"src": IdMap(str_src_ids),
              "dst": IdMap(str_dst_ids)}
    check_map_node_ids_exist(str_src_ids, str_dst_ids, id_map)
    check_map_node_ids_src_not_exist(str_src_ids, str_dst_ids, id_map)
    check_map_node_ids_dst_not_exist(str_src_ids, str_dst_ids, id_map)

def test_convert2ext_mem():
    # This is to verify the correctness of ExtMemArrayConverter
    converters = [ExtMemArrayConverter(None, 0),
                  ExtMemArrayConverter("/tmp", 2)]
    for converter in converters:
        arr = np.array([str(i) for i in range(10)])
        em_arr = converter(arr, "test1")
        np.testing.assert_array_equal(arr, em_arr)

        arr = np.random.uniform(size=(1000, 10))
        em_arr = converter(arr, "test2")
        np.testing.assert_array_equal(arr, em_arr)

def test_partition_graph():
    # This is to verify the correctness of partition_graph.
    # This function does some manual node/edge feature constructions for each partition.
    num_nodes = {'node1': 100,
                 'node2': 200,
                 'node3': 300}
    edges = {('node1', 'rel1', 'node2'): (np.random.randint(num_nodes['node1'], size=100),
                                          np.random.randint(num_nodes['node2'], size=100)),
             ('node1', 'rel2', 'node3'): (np.random.randint(num_nodes['node1'], size=200),
                                          np.random.randint(num_nodes['node3'], size=200))}
    node_data = {'node1': {'feat': np.random.uniform(size=(num_nodes['node1'], 10))},
                 'node2': {'feat': np.random.uniform(size=(num_nodes['node2'],))}}
    edge_data = {('node1', 'rel1', 'node2'): {'feat': np.random.uniform(size=(100, 10))}}

    # Partition the graph with our own partition_graph.
    g = dgl.heterograph(edges, num_nodes_dict=num_nodes)
    dgl.random.seed(0)
    num_parts = 2
    node_data1 = []
    edge_data1 = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        partition_graph(g, node_data, edge_data, 'test', num_parts, tmpdirname,
                        part_method="random")
        for i in range(num_parts):
            part_dir = os.path.join(tmpdirname, "part" + str(i))
            node_data1.append(dgl.data.utils.load_tensors(os.path.join(part_dir,
                                                                       'node_feat.dgl')))
            edge_data1.append(dgl.data.utils.load_tensors(os.path.join(part_dir,
                                                                       'edge_feat.dgl')))

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

if __name__ == '__main__':
    test_json()
    test_partition_graph()
    test_convert2ext_mem()
    test_map_node_ids()
    test_id_map()
    test_parquet()
    test_feat_ops()
    test_label()
