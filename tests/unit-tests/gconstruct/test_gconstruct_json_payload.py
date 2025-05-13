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
import copy
import os
import pytest
import json
import torch as th

from graphstorm.gconstruct.construct_payload_graph import (process_json_payload_graph,
                                            get_conf, merge_payload_input,
                                            process_json_payload_nodes,
                                            verify_payload_conf,
                                            STATUS, MSG, GRAPH, NODE_MAPPING)

_ROOT = os.path.abspath(os.path.dirname(__file__))
gconstruct_file_path = os.path.join(_ROOT, "../../end2end-tests/"
                                           "data_gen/movielens.json")
with open(gconstruct_file_path, 'r', encoding="utf8") as json_file:
    gconstruct_confs = json.load(json_file)

json_payload_file_path = os.path.join(_ROOT, "../../end2end-tests/"
                                             "data_gen/movielens_realtime_payload.json")
with open(json_payload_file_path, 'r', encoding="utf8") as json_file:
    json_payload = json.load(json_file)


def check_heterogeneous_graph(dgl_hg):
    assert dgl_hg.ntypes == ["movie", "user"]
    assert dgl_hg.canonical_etypes == [("user", "rating", "movie")]
    expected_node_count = {"movie": 2, "user": 1}
    for ntype in dgl_hg.ntypes:
        assert dgl_hg.num_nodes(ntype) == expected_node_count[ntype]
        if ntype == "movie":
            assert "title" in dgl_hg.nodes[ntype].data
            assert len(dgl_hg.nodes[ntype].data["title"]) == expected_node_count[ntype]
        elif ntype == "user":
            assert "feat" in dgl_hg.nodes[ntype].data
            assert len(dgl_hg.nodes[ntype].data["feat"]) == expected_node_count[ntype]

    for etype in dgl_hg.canonical_etypes:
        assert dgl_hg.num_edges(etype) == 2
        src_actual, dest_actual = dgl_hg.edges(etype=etype, order='eid')
        assert th.equal(src_actual, th.tensor([0, 0]))
        assert th.equal(dest_actual, th.tensor([0, 1]))


def test_process_json_payload_graph():
    response = process_json_payload_graph(json_payload_file_path,
                               gconstruct_file_path)
    assert response[STATUS] == 200
    assert MSG in response
    expected_raw_node_id_maps = {'user': {'a1': 0}, 'movie': {'m1': 0, 'm2': 1}}
    assert response[NODE_MAPPING] == expected_raw_node_id_maps

    dgl_hg = response[GRAPH]
    check_heterogeneous_graph(dgl_hg)


def test_with_two_phase_transformation():
    gconstruct_confs["nodes"][0]["features"] = {
            "feature_col": "feat",
            "transform": {"name": "max_min_norm",
                          "max_val": 2,
                          "min_val": -2}
    }
    response = process_json_payload_graph(json_payload_file_path,
                               gconstruct_file_path)
    assert response[STATUS] == 200
    assert MSG in response
    expected_raw_node_id_maps = {'user': {'a1': 0}, 'movie': {'m1': 0, 'm2': 1}}
    assert response[NODE_MAPPING] == expected_raw_node_id_maps

    dgl_hg = response[GRAPH]
    check_heterogeneous_graph(dgl_hg)


def test_get_gconstruct_conf():
    # Test merge feature transformation
    node_movie_config = get_conf(gconstruct_confs["nodes"], "movie", "Node")
    node_movie_config["features"] = [
        {'feature_col': 'title', 'transform': {
            'name': 'bert_hf', 'bert_model': 'bert-base-uncased',
            'max_seq_length': 16}}
    ]


def test_merge_payloads():
    # Test Node Payload
    payload_node_conf_list = json_payload["graph"]["nodes"]
    merged_payload_node_conf_list = merge_payload_input(payload_node_conf_list)
    assert len(merged_payload_node_conf_list) == 2
    assert merged_payload_node_conf_list[0]["node_type"] == "user"
    assert merged_payload_node_conf_list[0]["node_id"] == ["a1"]
    assert merged_payload_node_conf_list[0]["features"] == {"feat": [[-0.0032965524587780237]]}

    assert merged_payload_node_conf_list[1]["node_type"] == "movie"
    assert merged_payload_node_conf_list[1]["node_id"] == ["m1", "m2"]
    assert merged_payload_node_conf_list[1]["features"] == {'feat': [[0.011269339360296726], [0.0235343543]],
                                                            'title': ['sample text 1', 'sample text 2']}
    # Test Edge Payload
    payload_edge_conf_list = json_payload["graph"]["edges"]
    merged_payload_edge_conf_list = merge_payload_input(payload_edge_conf_list)

    assert len(merged_payload_edge_conf_list) == 1
    assert merged_payload_edge_conf_list[0]["edge_type"] == ['user', 'rating', 'movie']
    assert merged_payload_edge_conf_list[0]["src_node_id"] == ['a1', 'a1']
    assert merged_payload_edge_conf_list[0]["dest_node_id"] == ['m1', 'm2']


def test_verify_payload_conf():
    # Empty Input Conf
    input_conf = {}
    with pytest.raises(AssertionError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # Empty Node Conf
    input_conf = {
        "graph": {
            "edges": []
        }
    }
    with pytest.raises(AssertionError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # Empty Edge Conf
    input_conf = {
        "graph": {
            "nodes": []
        }
    }
    with pytest.raises(AssertionError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # All nodes should have node_type
    input_conf = copy.deepcopy(json_payload)
    del input_conf["graph"]["nodes"][0]["node_type"]
    with pytest.raises(AssertionError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # All nodes should have node_id
    input_conf = copy.deepcopy(json_payload)
    del input_conf["graph"]["nodes"][0]["node_id"]
    with pytest.raises(AssertionError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # All edges should have edge_type
    input_conf = copy.deepcopy(json_payload)
    del input_conf["graph"]["edges"][0]["edge_type"]
    with pytest.raises(AssertionError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # All edges should have src_node_id
    input_conf = copy.deepcopy(json_payload)
    del input_conf["graph"]["edges"][0]["src_node_id"]
    with pytest.raises(AssertionError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # All edges should have dest_node_id
    input_conf = copy.deepcopy(json_payload)
    del input_conf["graph"]["edges"][0]["dest_node_id"]
    with pytest.raises(AssertionError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # All nodes should have consistency on features
    input_conf = copy.deepcopy(json_payload)
    del input_conf["graph"]["nodes"][1]["features"]
    with pytest.raises(AssertionError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # All edges should have consistency on features
    input_conf = copy.deepcopy(json_payload)
    del input_conf["graph"]["edges"][1]["features"]
    with pytest.raises(AssertionError):
        verify_payload_conf(input_conf, gconstruct_confs)
