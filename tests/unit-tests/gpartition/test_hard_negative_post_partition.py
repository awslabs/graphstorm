"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

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
import os
import json
import torch as th
import numpy as np
from typing import Dict

import pytest

from numpy.testing import assert_almost_equal
from graphstorm.model.utils import load_dist_nid_map
from dgl.data.utils import load_tensors, save_tensors
from graphstorm.gpartition.post_hard_negative import (shuffle_hard_negative_nids,
                                                      load_hard_negative_config)

@pytest.fixture(scope="module", name="gsprocessing_hard_negative_config")
def gsprocessing_config_hard_negative_dict_fixture() -> Dict:
    return{
        "graph": {
            "nodes": [
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "./nodes/author.parquet"
                        ]
                    },
                    "type": "author",
                    "column": "node_id",
                },
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "./nodes/paper.parquet"
                        ]
                    },
                    "type": "paper",
                    "column": "node_id",
                    "features": [
                        {
                            "column": "feat",
                            "name": "feat",
                            "transformation": {
                                "name": "no-op"
                            }
                        }
                    ],
                    "labels": [
                        {
                            "column": "label",
                            "type": "classification",
                            "split_rate": {
                                "train": 0.8,
                                "val": 0.1,
                                "test": 0.1
                            }
                        }
                    ]
                }
            ],
            "edges": [
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "./edges/author_writing_paper_hard_negative.parquet"
                        ]
                    },
                    "source": {
                        "column": "source_id",
                        "type": "author"
                    },
                    "dest": {
                        "column": "dest_id",
                        "type": "paper"
                    },
                    "relation": {
                        "type": "writing"
                    },
                    "features": [
                        {
                            "column": "hard_neg",
                            "name": "hard_neg_feat",
                            "transformation": {
                                "name": "edge_dst_hard_negative",
                                "kwargs": {
                                    "separator": ";"
                                }
                            }
                        }
                    ]
                },
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "./edges/paper_citing_paper.parquet"
                        ]
                    },
                    "source": {
                        "column": "source_id",
                        "type": "paper"
                    },
                    "dest": {
                        "column": "dest_id",
                        "type": "paper"
                    },
                    "relation": {
                        "type": "citing"
                    },
                    "labels": [
                        {
                            "column": "",
                            "type": "link_prediction",
                            "split_rate": {
                                "train": 0.8,
                                "val": 0.1,
                                "test": 0.1
                            }
                        }
                    ]
                }
            ]
        },
        "version": "gsprocessing-v1.0"
    }


@pytest.fixture(scope="module", name="gsprocessing_non_hard_negative_config")
def gsprocessing_config_non_hard_negative_dict_fixture() -> Dict:
    return{
        "graph": {
            "nodes": [
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "./nodes/author.parquet"
                        ]
                    },
                    "type": "author",
                    "column": "node_id",
                },
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "./nodes/paper.parquet"
                        ]
                    },
                    "type": "paper",
                    "column": "node_id",
                    "features": [
                        {
                            "column": "feat",
                            "name": "feat",
                            "transformation": {
                                "name": "no-op"
                            }
                        }
                    ],
                    "labels": [
                        {
                            "column": "label",
                            "type": "classification",
                            "split_rate": {
                                "train": 0.8,
                                "val": 0.1,
                                "test": 0.1
                            }
                        }
                    ]
                }
            ],
            "edges": [
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "./edges/author_writing_paper_hard_negative.parquet"
                        ]
                    },
                    "source": {
                        "column": "source_id",
                        "type": "author"
                    },
                    "dest": {
                        "column": "dest_id",
                        "type": "paper"
                    },
                    "relation": {
                        "type": "writing"
                    }
                },
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "./edges/paper_citing_paper.parquet"
                        ]
                    },
                    "source": {
                        "column": "source_id",
                        "type": "paper"
                    },
                    "dest": {
                        "column": "dest_id",
                        "type": "paper"
                    },
                    "relation": {
                        "type": "citing"
                    },
                    "labels": [
                        {
                            "column": "",
                            "type": "link_prediction",
                            "split_rate": {
                                "train": 0.8,
                                "val": 0.1,
                                "test": 0.1
                            }
                        }
                    ]
                }
            ]
        },
        "version": "gsprocessing-v1.0"
    }


def test_load_hard_negative_config(tmp_path, gsprocessing_hard_negative_config: Dict,
                                   gsprocessing_non_hard_negative_config: Dict):
    # For config with gsprocessing_config.json
    json_file_path = f"{tmp_path}/gsprocessing_config.json"

    # Write the dictionary to the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(gsprocessing_hard_negative_config, json_file, indent=4)

    res = load_hard_negative_config(json_file_path)

    assert res[0] == {'dst_node_type': 'paper', 'edge_type':
        'author:writing:paper', 'hard_neg_feat_name': 'hard_neg_feat'}

    # For config without hard negative feature definition
    json_file_path = f"{tmp_path}/gsprocessing_config.json"

    # Write the dictionary to the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(gsprocessing_non_hard_negative_config,
                  json_file, indent=4)

    res = load_hard_negative_config(json_file_path)

    assert res == []


def test_shuffle_hard_negative_nids(tmp_path, gsprocessing_hard_negative_config: Dict):
    # For config with gsprocessing_config.json
    json_file_path = f"{tmp_path}/gsprocessing_config.json"

    # Write the dictionary to the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(gsprocessing_hard_negative_config, json_file, indent=4)

    # Generate dgl graph
    partitioned_graph = f"{tmp_path}/partitioned_graph"

    # Generate ID mapping for each partition
    nid_map_dict_path0 = os.path.join(partitioned_graph, "dist_graph", "part0", "orig_nids.dgl")
    nid_map_dict_path1 = os.path.join(partitioned_graph, "dist_graph", "part1", "orig_nids.dgl")
    os.makedirs(os.path.dirname(nid_map_dict_path0), exist_ok=True)
    os.makedirs(os.path.dirname(nid_map_dict_path1), exist_ok=True)

    # Use randperm in the test otherwise there maybe no mapping necessary
    nid_map0 = {
        "paper": th.randperm(100),
        "author": th.arange(200, 300)
    }
    save_tensors(nid_map_dict_path0, nid_map0)

    nid_map1 = {
        "paper": th.randperm(100) + 100,
        "author": th.arange(300, 400)
    }
    save_tensors(nid_map_dict_path1, nid_map1)

    # Create reversed map
    node_mapping = load_dist_nid_map(f"{partitioned_graph}/dist_graph", ["author", "paper"])
    reverse_map_dst = {gid: i for i, gid in enumerate(node_mapping["paper"].tolist())}
    reverse_map_dst[-1] = -1

    # generate edge features
    etype = ("author", "writing", "paper")
    edge_feat_path0 = os.path.join(partitioned_graph, "dist_graph", "part0", "edge_feat.dgl")
    edge_feat_path1 = os.path.join(partitioned_graph, "dist_graph", "part1", "edge_feat.dgl")
    os.makedirs(os.path.dirname(edge_feat_path0), exist_ok=True)
    os.makedirs(os.path.dirname(edge_feat_path1), exist_ok=True)

    paper_writing_hard_neg0 = th.cat((th.randint(0, 100, (100, 100)),
                                        th.full((100, 10), -1, dtype=th.int32)), dim=1)
    paper_writing_hard_neg0_shuffled = [
        [reverse_map_dst[nid] for nid in negs] \
        for negs in paper_writing_hard_neg0.tolist()]
    paper_writing_hard_neg0_shuffled = np.array(paper_writing_hard_neg0_shuffled)
    paper_writing_hard_neg1 = th.cat((th.randint(100, 200, (100, 100)),
                                        th.full((100, 10), -1, dtype=th.int32)), dim=1)
    paper_writing_hard_neg1_shuffled = [
        [reverse_map_dst[nid] for nid in negs] \
        for negs in paper_writing_hard_neg1.tolist()]
    paper_writing_hard_neg1_shuffled = np.array(paper_writing_hard_neg1_shuffled)

    save_tensors(edge_feat_path0, {":".join(etype)+"/hard_neg_feat": paper_writing_hard_neg0})
    save_tensors(edge_feat_path1, {":".join(etype)+"/hard_neg_feat": paper_writing_hard_neg1})

    # Do the shuffling
    shuffle_hard_negative_nids(json_file_path, 2, partitioned_graph)

    # Assert
    paper_writing_hard_neg0 = load_tensors(edge_feat_path0)
    assert_almost_equal(paper_writing_hard_neg0[":".join(etype) + "/hard_neg_feat"].numpy(),
                        paper_writing_hard_neg0_shuffled)
    paper_writing_hard_neg1 = load_tensors(edge_feat_path1)
    assert_almost_equal(paper_writing_hard_neg1[":".join(etype) + "/hard_neg_feat"].numpy(),
                        paper_writing_hard_neg1_shuffled)