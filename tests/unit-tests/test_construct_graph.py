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

def test_parquet():
    from graphstorm.gconstruct import write_data_parquet, read_data_parquet
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
    assert np.all(data1['data1'] == data['data1'])
    assert np.all(data1['data2'] == data['data2'])

    data1 = read_data_parquet(tmpfile, data_fields=['data1'])
    assert len(data1) == 1
    assert "data1" in data1
    assert "data2" not in data1
    assert np.all(data1['data1'] == data['data1'])

def test_feat_ops():
    from graphstorm.gconstruct import parse_feat_ops, process_features

    feat_op1 = [{
        "feature_col": "test1",
        "feature_name": "test2",
    }]
    res1 = parse_feat_ops(feat_op1)
    assert len(res1) == 1
    assert len(res1[0]) == 4
    assert res1[0][0] == feat_op1[0]["feature_col"]
    assert res1[0][1] == feat_op1[0]["feature_name"]
    assert res1[0][2] is None   # dtype is always None for now.
    assert res1[0][3] is None   # There is not transformation.

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
    assert len(res2[0]) == 4
    assert res2[1][0] == feat_op2[1]["feature_col"]
    assert res2[1][1] == feat_op2[1]["feature_name"]
    assert res2[1][2] is None   # dtype is always None for now.
    op = res2[1][3]
    tokens = op(["hello world", "hello world"])
    assert len(tokens) == 3
    assert tokens['token_ids'].shape == (2, 16)
    assert tokens['attention_mask'].shape == (2, 16)
    assert tokens['token_type_ids'].shape == (2, 16)
    assert np.all(tokens['token_ids'][0] == tokens['token_ids'][1])
    assert np.all(tokens['attention_mask'][0] == tokens['attention_mask'][1])
    assert np.all(tokens['token_type_ids'][0] == tokens['token_type_ids'][1])

    data = {
        "test1": np.random.rand(2, 4),
        "test3": ["hello world", "hello world"],
    }
    proc_res = process_features(data, res2)
    assert np.all(data['test1'] == proc_res['test2'])
    assert "token_ids" in proc_res
    assert "attention_mask" in proc_res
    assert "token_type_ids" in proc_res

def test_label():
    from graphstorm.gconstruct import process_labels
    data = {
        "label": np.random.randint(5, size=10),
    }
    label_conf = [
        {
            "label_col": "label",
            "task_type": "classification",
            "split_type": [0.8, 0.1, 0.1],
        },
    ]
    res = process_labels(data, label_conf)
    assert np.all(res['label'] == data['label'])
    assert res['train_mask'].shape == (len(data['label']),)
    assert res['val_mask'].shape == (len(data['label']),)
    assert res['test_mask'].shape == (len(data['label']),)
    assert np.sum(res['train_mask']) == 8
    assert np.sum(res['val_mask']) == 1
    assert np.sum(res['test_mask']) == 1

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
    from graphstorm.gconstruct.construct_graph import IdMap
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
    from graphstorm.gconstruct.construct_graph import IdMap
    str_ids = np.array([str(i) for i in range(10)])
    id_map = IdMap(str_ids)

    check_id_map_exist(id_map, str_ids)
    check_id_map_not_exist(id_map, str_ids)
    check_id_map_dtype_not_match(id_map, str_ids)

def check_map_node_ids_exist(str_src_ids, str_dst_ids, id_map):
    # Test the case that both source node IDs and destination node IDs exist.
    from graphstorm.gconstruct.construct_graph import map_node_ids
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
    from graphstorm.gconstruct.construct_graph import map_node_ids
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
    from graphstorm.gconstruct.construct_graph import map_node_ids
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
    from graphstorm.gconstruct.construct_graph import IdMap
    str_src_ids = np.array([str(i) for i in range(10)])
    str_dst_ids = np.array([str(i) for i in range(15)])
    id_map = {"src": IdMap(str_src_ids),
              "dst": IdMap(str_dst_ids)}
    check_map_node_ids_exist(str_src_ids, str_dst_ids, id_map)
    check_map_node_ids_src_not_exist(str_src_ids, str_dst_ids, id_map)
    check_map_node_ids_dst_not_exist(str_src_ids, str_dst_ids, id_map)

if __name__ == '__main__':
    test_map_node_ids()
    test_id_map()
    test_parquet()
    test_feat_ops()
    test_label()
