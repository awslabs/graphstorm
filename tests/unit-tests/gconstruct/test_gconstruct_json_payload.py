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
import numpy as np

from graphstorm.gconstruct.construct_payload_graph import (process_json_payload_graph,
                                            get_gconstruct_conf, merge_payload_input,
                                            verify_payload_conf,
                                            PAYLOAD_PROCESSING_STATUS,
                                            PAYLOAD_PROCESSING_RETURN_MSG,
                                            PAYLOAD_GRAPH,
                                            PAYLOAD_GRAPH_NODE_MAPPING)
from graphstorm.gconstruct.payload_utils import BaseApplicationError
from graphstorm.config.config import GS_LE_FEATURE_KEY

_ROOT = os.path.abspath(os.path.dirname(__file__))
gconstruct_file_path = os.path.join(_ROOT, "../../end2end-tests/"
                                           "data_gen/movielens.json")
with open(gconstruct_file_path, 'r', encoding="utf8") as json_file:
    gconstruct_confs = json.load(json_file)
    # add this configuration to fit the data for real-time entry point test for 2 target ntypes
    gconstruct_confs['edges'].append(
        {
            "source_id_col":    "src_id",
            "dest_id_col":      "dst_id",
            "relation":         ["movie", "rating-rev", "user"],
            "format":           {"name": "parquet"},
            "files":        "/data/ml-100k/edges_rev.parquet",
        }
    )

json_payload_file_path = os.path.join(_ROOT, "../../end2end-tests/"
                                             "data_gen/movielens_realtime_payload.json")
with open(json_payload_file_path, 'r', encoding="utf8") as json_file:
    json_payload = json.load(json_file)


def check_heterogeneous_graph(dgl_hg):
    assert dgl_hg.ntypes == ["movie", "user"]
    assert dgl_hg.canonical_etypes == [("movie", "rating-rev", "user"),
                                       ("user", "rating", "movie")]
    expected_node_count = {"movie": 2, "user": 1}
    for ntype in dgl_hg.ntypes:
        assert dgl_hg.num_nodes(ntype) == expected_node_count[ntype]
        if ntype == "movie":
            assert "title" in dgl_hg.nodes[ntype].data or "feat" in dgl_hg.nodes[ntype].data
            # If bert feature transformation
            if "title" in dgl_hg.nodes[ntype].data:
                assert len(dgl_hg.nodes[ntype].data["title"]) == expected_node_count[ntype]
            # If rank_guass feature transformation
            if "feat" in dgl_hg.nodes[ntype].data:
                assert len(dgl_hg.nodes[ntype].data["feat"]) == expected_node_count[ntype]
        elif ntype == "user":
            # If Noop feature transformation
            assert "feat" in dgl_hg.nodes[ntype].data
            assert len(dgl_hg.nodes[ntype].data["feat"]) == expected_node_count[ntype]

    for etype in dgl_hg.canonical_etypes:
        if etype[1].endswith('-rev'):
            assert dgl_hg.num_edges(etype) == 1
            src_actual, dest_actual = dgl_hg.edges(etype=etype, order='eid')
            assert th.equal(src_actual, th.tensor([0]))
            assert th.equal(dest_actual, th.tensor([0]))
        else:
            assert dgl_hg.num_edges(etype) == 2
            src_actual, dest_actual = dgl_hg.edges(etype=etype, order='eid')
            assert th.equal(src_actual, th.tensor([0, 0]))
            assert th.equal(dest_actual, th.tensor([0, 1]))


def test_process_json_payload_graph():
    
    response = process_json_payload_graph(json_payload, gconstruct_confs)
    assert response[PAYLOAD_PROCESSING_STATUS] == 200
    assert PAYLOAD_PROCESSING_RETURN_MSG in response
    expected_raw_node_id_maps = {'user': {'a1': 0}, 'movie': {'m1': 0, 'm2': 1}}
    assert response[PAYLOAD_GRAPH_NODE_MAPPING] == expected_raw_node_id_maps

    dgl_hg = response[PAYLOAD_GRAPH]
    check_heterogeneous_graph(dgl_hg)

    # Test with Edge features
    edge_feat_gconstruct_confs = copy.deepcopy(gconstruct_confs)
    edge_feat_gconstruct_confs["edges"][0]["features"] = [{
            "feature_col": "rate"
    }]
    response = process_json_payload_graph(json_payload, edge_feat_gconstruct_confs)
    dgl_hg = response[PAYLOAD_GRAPH]
    check_heterogeneous_graph(dgl_hg)
    for etype in dgl_hg.canonical_etypes:
        if not etype[1].endswith('-rev'):
            assert "rate" in dgl_hg.edges[etype].data

    # ============================ v0.5.1 ============================ #

    # 1. test for some node features are not presented in the construction json
    # add the 'feat' to "movie" nodes into the construction json
    new_gconstruct_confs = copy.deepcopy(gconstruct_confs)
    new_gconstruct_confs['nodes'][2]['features'].append({'feature_col': 'feat'})
    # remove the 'feat' from movie nodes
    new_json_payload = copy.deepcopy(json_payload)
    new_json_payload['graph']['nodes'][1]['features'].pop('feat')
    new_json_payload['graph']['nodes'][2]['features'].pop('feat')

    # should be able to process the new json payload
    response = process_json_payload_graph(new_json_payload, new_gconstruct_confs)
    assert response[PAYLOAD_PROCESSING_STATUS] == 200

    dgl_hg = response[PAYLOAD_GRAPH]
    assert 'feat' in dgl_hg.nodes['user'].data
    assert not 'feat' in dgl_hg.nodes['movie'].data

    # 2. test for learnable embedding that construction json does not know
    new_json_payload['graph']['nodes'][1]['features'][GS_LE_FEATURE_KEY] = np.random.rand(16)
    new_json_payload['graph']['nodes'][2]['features'][GS_LE_FEATURE_KEY] = np.random.rand(16)

    response = process_json_payload_graph(new_json_payload, new_gconstruct_confs)
    assert response[PAYLOAD_PROCESSING_STATUS] == 200

    dgl_hg = response[PAYLOAD_GRAPH]
    assert GS_LE_FEATURE_KEY in dgl_hg.nodes['movie'].data
    assert dgl_hg.nodes['movie'].data[GS_LE_FEATURE_KEY].shape == (2, 16)

def test_with_two_phase_transformation():
    # Node Feature Transformation
    two_phase_gconstruct_confs = copy.deepcopy(gconstruct_confs)
    two_phase_gconstruct_confs["nodes"][0]["features"] = [{
            "feature_col": "feat",
            "transform": {"name": "max_min_norm",
                          "max_val": 2,
                          "min_val": -2}
    }]
    response = process_json_payload_graph(json_payload, two_phase_gconstruct_confs)
    assert response[PAYLOAD_PROCESSING_STATUS] == 200
    assert PAYLOAD_PROCESSING_RETURN_MSG in response
    expected_raw_node_id_maps = {'user': {'a1': 0}, 'movie': {'m1': 0, 'm2': 1}}
    assert response[PAYLOAD_GRAPH_NODE_MAPPING] == expected_raw_node_id_maps

    dgl_hg = response[PAYLOAD_GRAPH]
    check_heterogeneous_graph(dgl_hg)

    # Edge Feature Transformation
    edge_feat_gconstruct_confs = copy.deepcopy(gconstruct_confs)
    edge_feat_gconstruct_confs["edges"][0]["features"] = [{
            "feature_col": "rate",
            "transform": {"name": "max_min_norm",
                          "max_val": 2,
                          "min_val": -2}
    }]
    response = process_json_payload_graph(json_payload, edge_feat_gconstruct_confs)
    dgl_hg = response[PAYLOAD_GRAPH]
    check_heterogeneous_graph(dgl_hg)
    for etype in dgl_hg.canonical_etypes:
        if not etype[1].endswith('-rev'):
            assert "rate" in dgl_hg.edges[etype].data


def test_with_after_merge_transformation():
    # Node Feature Transformation
    after_merge_gconstruct_conf = copy.deepcopy(gconstruct_confs)
    after_merge_gconstruct_conf["nodes"][2]["features"] = [{
            "feature_col": "feat",
            "transform": {"name": "rank_gauss"}
    }]
    response = process_json_payload_graph(json_payload, after_merge_gconstruct_conf)
    assert response[PAYLOAD_PROCESSING_STATUS] == 200
    assert PAYLOAD_PROCESSING_RETURN_MSG in response
    expected_raw_node_id_maps = {'user': {'a1': 0}, 'movie': {'m1': 0, 'm2': 1}}
    assert response[PAYLOAD_GRAPH_NODE_MAPPING] == expected_raw_node_id_maps

    dgl_hg = response[PAYLOAD_GRAPH]
    check_heterogeneous_graph(dgl_hg)

    # Edge Feature Transformation
    edge_feat_gconstruct_confs = copy.deepcopy(gconstruct_confs)
    edge_feat_gconstruct_confs["edges"][0]["features"] = [{
            "feature_col": "rate",
            "transform": {"name": "rank_gauss"}
    }]
    response = process_json_payload_graph(json_payload, edge_feat_gconstruct_confs)
    dgl_hg = response[PAYLOAD_GRAPH]
    check_heterogeneous_graph(dgl_hg)
    for etype in dgl_hg.canonical_etypes:
        if not etype[1].endswith('-rev'):
            assert "rate" in dgl_hg.edges[etype].data


def test_get_gconstruct_conf():
    # Test merge feature transformation
    node_movie_config = get_gconstruct_conf(gconstruct_confs["nodes"], "movie", "Node")
    assert node_movie_config["features"] == [
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
    assert merged_payload_node_conf_list[0]["features"] == {"feat": [[-0.0032965524587780237, -0.1]]}

    assert merged_payload_node_conf_list[1]["node_type"] == "movie"
    assert merged_payload_node_conf_list[1]["node_id"] == ["m1", "m2"]
    assert merged_payload_node_conf_list[1]["features"] == {'feat': [[0.011269339360296726, 0.1], [0.0235343543, 0.1]],
                                                            'title': ['sample text 1', 'sample text 2']}
    # Test Edge Payload
    payload_edge_conf_list = json_payload["graph"]["edges"]
    merged_payload_edge_conf_list = merge_payload_input(payload_edge_conf_list)

    assert len(merged_payload_edge_conf_list) == 2
    assert merged_payload_edge_conf_list[0]["edge_type"] == ['movie', 'rating-rev', 'user']
    assert merged_payload_edge_conf_list[0]["src_node_id"] == ['m1']
    assert merged_payload_edge_conf_list[0]["dest_node_id"] == ['a1']

    assert merged_payload_edge_conf_list[1]["edge_type"] == ['user', 'rating', 'movie']
    assert merged_payload_edge_conf_list[1]["src_node_id"] == ['a1', 'a1']
    assert merged_payload_edge_conf_list[1]["dest_node_id"] == ['m1', 'm2']


def test_verify_payload_conf():
    # Case 1: Empty Input Conf
    input_conf = {}
    with pytest.raises(BaseApplicationError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # Case 2: Empty Node Conf
    input_conf = {
        "graph": {
            "edges": []
        }
    }
    with pytest.raises(BaseApplicationError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # Case 3: Empty Edge Conf
    input_conf = {
        "graph": {
            "nodes": []
        }
    }
    with pytest.raises(BaseApplicationError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # Case 4: All nodes should have node_type
    input_conf = copy.deepcopy(json_payload)
    del input_conf["graph"]["nodes"][0]["node_type"]
    with pytest.raises(BaseApplicationError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # Case 5: All nodes should have node_id
    input_conf = copy.deepcopy(json_payload)
    del input_conf["graph"]["nodes"][0]["node_id"]
    with pytest.raises(BaseApplicationError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # Case 6: All edges should have edge_type
    input_conf = copy.deepcopy(json_payload)
    del input_conf["graph"]["edges"][0]["edge_type"]
    with pytest.raises(BaseApplicationError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # Case 7: All edges should have src_node_id
    input_conf = copy.deepcopy(json_payload)
    del input_conf["graph"]["edges"][0]["src_node_id"]
    with pytest.raises(BaseApplicationError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # Case 8: All edges should have dest_node_id
    input_conf = copy.deepcopy(json_payload)
    del input_conf["graph"]["edges"][0]["dest_node_id"]
    with pytest.raises(BaseApplicationError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # Case 9: All nodes should have consistency on features
    # Raise error for case
    # {
    #     "node_type": "user",
    #     "features": {
    #         "feat": [
    #             -0.0032965524587780237, -0.1
    #         ]
    #     },
    #     "node_id": "u1"
    # },
    # {
    #     "node_type": "user",
    #     "node_id": "u2"
    # },
    input_conf = copy.deepcopy(json_payload)
    del input_conf["graph"]["nodes"][1]["features"]
    with pytest.raises(BaseApplicationError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # Case 10: All edges should have consistency on features
    input_conf = copy.deepcopy(json_payload)
    del input_conf["graph"]["edges"][1]["features"]
    with pytest.raises(BaseApplicationError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # Case 11: All nodes should have same feature name
    # Raise error for different feature name
    # Raise error for case
    # {
    #     "node_type": "user",
    #     "features": {
    #         "feat": [
    #             -0.0032965524587780237, -0.1
    #         ]
    #     },
    #     "node_id": "u1"
    # },
    # {
    #     "node_type": "user",
    #     "features": {
    #         "feat_err": [
    #             -0.0032965524587780237, -0.1
    #         ]
    #     },
    #     "node_id": "u2"
    # },
    input_conf = copy.deepcopy(json_payload)
    input_conf["graph"]["nodes"][1]["features"]["feat_err"] = [0.1]
    with pytest.raises(BaseApplicationError):
        verify_payload_conf(input_conf, gconstruct_confs)

    # Case 12: All edges should have same feature name
    input_conf = copy.deepcopy(json_payload)
    input_conf["graph"]["edges"][1]["features"]["feat_err"] = [0.1]
    with pytest.raises(BaseApplicationError):
        verify_payload_conf(input_conf, gconstruct_confs)
