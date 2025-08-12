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

    Test functions and classes in the metadata.py
"""
import os
import tempfile
import json

import pytest
from dgl.distributed.constants import DEFAULT_ETYPE, DEFAULT_NTYPE

from graphstorm.dataloading import (GSGraphMetadata,
                                    GSDglDistGraphFromMetadata,
                                    config_json_sanity_check,
                                    load_metadata_from_json)

def build_gcons_json_example(gtype='heterogeneous'):
    """ An new JSON file created by gconstruct command.

    This JSON is a real example of the ACM example dataset.
    """
    if gtype == 'heterogeneous':
        conf = {
        "version": "gconstruct-v0.1",
        "nodes": [
            {
                "node_type": "author",
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/nodes/author.parquet"
                ],
                "node_id_col": "node_id",
                "features": [
                    {
                        "feature_col": "feat",
                        "feature_name": "feat",
                        "feature_dim": [
                            256
                        ]
                    }
                ]
            },
            {
                "node_type": "paper",
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/nodes/paper.parquet"
                ],
                "node_id_col": "node_id",
                "features": [
                    {
                        "feature_col": "feat",
                        "feature_name": "feat",
                        "feature_dim": [
                            256
                        ]
                    }
                ],
                "labels": [
                    {
                        "label_col": "label",
                        "task_type": "classification",
                        "split_pct": [
                            0.8,
                            0.1,
                            0.1
                        ],
                        "label_stats_type": "frequency_cnt"
                    }
                ]
            },
            {
                "node_type": "subject",
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/nodes/subject.parquet"
                ],
                "node_id_col": "node_id",
                "features": [
                    {
                        "feature_col": "feat",
                        "feature_name": "feat",
                        "feature_dim": [
                            256
                        ]
                    }
                ]
            }
        ],
        "edges": [
            {
                "relation": [
                    "author",
                    "writing",
                    "paper"
                ],
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/edges/author_writing_paper.parquet"
                ],
                "source_id_col": "source_id",
                "dest_id_col": "dest_id"
            },
            {
                "relation": [
                    "paper",
                    "cited",
                    "paper"
                ],
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/edges/paper_cited_paper.parquet"
                ],
                "source_id_col": "source_id",
                "dest_id_col": "dest_id"
            },
            {
                "relation": [
                    "paper",
                    "citing",
                    "paper"
                ],
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/edges/paper_citing_paper.parquet"
                ],
                "source_id_col": "source_id",
                "dest_id_col": "dest_id",
                "features": [
                    {
                        "feature_col": "cate_feat",
                        "feature_name": "cate_feat",
                        "transform": {
                            "name": "to_categorical",
                            "mapping": {
                                "C_1": 0,
                                "C_10": 1,
                                "C_13": 2,
                                "C_16": 3,
                                "C_4": 4,
                                "C_7": 5
                            }
                        },
                        "feature_dim": [
                            6
                        ]
                    }
                ],
                "labels": [
                    {
                        "task_type": "link_prediction",
                        "split_pct": [
                            0.8,
                            0.1,
                            0.1
                        ]
                    }
                ]
            },
            {
                "relation": [
                    "paper",
                    "is-about",
                    "subject"
                ],
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/edges/paper_is-about_subject.parquet"
                ],
                "source_id_col": "source_id",
                "dest_id_col": "dest_id"
            },
            {
                "relation": [
                    "paper",
                    "written-by",
                    "author"
                ],
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/edges/paper_written-by_author.parquet"
                ],
                "source_id_col": "source_id",
                "dest_id_col": "dest_id"
            },
            {
                "relation": [
                    "subject",
                    "has",
                    "paper"
                ],
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/edges/subject_has_paper.parquet"
                ],
                "source_id_col": "source_id",
                "dest_id_col": "dest_id"
            }
        ],
        "add_reverse_edge": "false",
        "is_homogeneous": "false"
    }
    elif gtype == 'homogeneous':
        conf = {
            "version": "gconstruct-v0.1",
            "nodes": [
                {
                    "node_type": "_N",
                    "format": {
                        "name": "parquet"
                    },
                    "files": [
                        "/tmp/acm_raw/nodes/paper.parquet"
                    ],
                    "node_id_col": "node_id",
                    "features": [
                        {
                            "feature_col": "feat",
                            "feature_name": "feat",
                            "feature_dim": [
                                256
                            ]
                        }
                    ],
                    "labels": [
                        {
                            "label_col": "label",
                            "task_type": "classification",
                            "split_pct": [
                                0.8,
                                0.1,
                                0.1
                            ],
                            "label_stats_type": "frequency_cnt"
                        }
                    ]
                },
            ],
            "edges": [
                {
                    "relation": [
                        "_N",
                        "_E",
                        "_N"
                    ],
                    "format": {
                        "name": "parquet"
                    },
                    "files": [
                        "/tmp/acm_raw/edges/paper_citing_paper.parquet"
                    ],
                    "source_id_col": "source_id",
                    "dest_id_col": "dest_id",
                    "features": [
                        {
                            "feature_col": "cate_feat",
                            "feature_name": "cate_feat",
                            "transform": {
                                "name": "to_categorical",
                                "mapping": {
                                    "C_1": 0,
                                    "C_10": 1,
                                    "C_13": 2,
                                    "C_16": 3,
                                    "C_4": 4,
                                    "C_7": 5
                                }
                            },
                            "feature_dim": [
                                6
                            ]
                        }
                    ],
                    "labels": [
                        {
                            "task_type": "link_prediction",
                            "split_pct": [
                                0.8,
                                0.1,
                                0.1
                            ]
                        }
                    ]
                },
            ],
            "add_reverse_edge": "false",
            "is_homogeneous": "true"
        }
    else:
        raise NotImplementedError('Only support \"heterogeneous\" and \"homogeneous\" options.' \
            f'but got {gtype}.')

    return conf

def build_gsproc_json_example(gtype='heterogeneous'):
    """ An new JSON file created by GSProcessing command.

    This JSON is a real example of the ACM example dataset.
    """
    if gtype == 'heterogeneous':
        conf = {
        "graph": {
            "add_reverse_edge": "false",
            "nodes": [
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "nodes/author.parquet"
                        ]
                    },
                    "type": "author",
                    "column": "node_id",
                    "features": [
                        {
                            "column": "feat",
                            "name": "feat",
                            "transformation": {
                                "name": "no-op"
                            },
                            "dim": [
                                256
                            ]
                        }
                    ]
                },
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "nodes/paper.parquet"
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
                            },
                            "dim": [
                                256
                            ]
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
                },
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "nodes/subject.parquet"
                        ]
                    },
                    "type": "subject",
                    "column": "node_id",
                    "features": [
                        {
                            "column": "feat",
                            "name": "feat",
                            "transformation": {
                                "name": "no-op"
                            },
                            "dim": [
                                256
                            ]
                        }
                    ]
                }
            ],
            "edges": [
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "edges/author_writing_paper.parquet"
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
                            "edges/paper_cited_paper.parquet"
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
                        "type": "cited"
                    }
                },
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "edges/paper_citing_paper.parquet"
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
                    "features": [
                        {
                            "column": "cate_feat",
                            "name": "cate_feat",
                            "transformation": {
                                "name": "categorical",
                                "kwargs": {}
                            },
                            "dim": [
                                6
                            ],
                            "precomputed_transformation": {
                                "string_indexer_labels_arrays": [
                                    [
                                        "C_16",
                                        "C_4",
                                        "C_7",
                                        "C_1",
                                        "C_10",
                                        "C_13"
                                    ]
                                ],
                                "cols": [
                                    "cate_feat"
                                ],
                                "per_col_label_to_one_hot_idx": {
                                    "cate_feat": {
                                        "C_1": 3,
                                        "C_10": 4,
                                        "C_13": 5,
                                        "C_4": 1,
                                        "C_16": 0,
                                        "C_7": 2
                                    }
                                },
                                "transformation_name": "DistCategoryTransformation"
                            }
                        }
                    ],
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
                },
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "edges/paper_is-about_subject.parquet"
                        ]
                    },
                    "source": {
                        "column": "source_id",
                        "type": "paper"
                    },
                    "dest": {
                        "column": "dest_id",
                        "type": "subject"
                    },
                    "relation": {
                        "type": "is-about"
                    }
                },
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "edges/paper_written-by_author.parquet"
                        ]
                    },
                    "source": {
                        "column": "source_id",
                        "type": "paper"
                    },
                    "dest": {
                        "column": "dest_id",
                        "type": "author"
                    },
                    "relation": {
                        "type": "written-by"
                    }
                },
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "edges/subject_has_paper.parquet"
                        ]
                    },
                    "source": {
                        "column": "source_id",
                        "type": "subject"
                    },
                    "dest": {
                        "column": "dest_id",
                        "type": "paper"
                    },
                    "relation": {
                        "type": "has"
                    }
                }
            ],
            "is_homogeneous": "false"
        },
        "version": "gsprocessing-v1.0"
    }
    elif gtype == 'homogeneous':
        conf = {
        "graph": {
            "add_reverse_edge": "false",
            "nodes": [
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "nodes/paper.parquet"
                        ]
                    },
                    "type": "_N",
                    "column": "node_id",
                    "features": [
                        {
                            "column": "feat",
                            "name": "feat",
                            "transformation": {
                                "name": "no-op"
                            },
                            "dim": [
                                256
                            ]
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
                },
            ],
            "edges": [
                {
                    "data": {
                        "format": "parquet",
                        "files": [
                            "edges/paper_citing_paper.parquet"
                        ]
                    },
                    "source": {
                        "column": "source_id",
                        "type": "_N"
                    },
                    "dest": {
                        "column": "dest_id",
                        "type": "_N"
                    },
                    "relation": {
                        "type": "_E"
                    },
                    "features": [
                        {
                            "column": "cate_feat",
                            "name": "cate_feat",
                            "transformation": {
                                "name": "categorical",
                                "kwargs": {}
                            },
                            "dim": [
                                6
                            ],
                            "precomputed_transformation": {
                                "string_indexer_labels_arrays": [
                                    [
                                        "C_16",
                                        "C_4",
                                        "C_7",
                                        "C_1",
                                        "C_10",
                                        "C_13"
                                    ]
                                ],
                                "cols": [
                                    "cate_feat"
                                ],
                                "per_col_label_to_one_hot_idx": {
                                    "cate_feat": {
                                        "C_1": 3,
                                        "C_10": 4,
                                        "C_13": 5,
                                        "C_4": 1,
                                        "C_16": 0,
                                        "C_7": 2
                                    }
                                },
                                "transformation_name": "DistCategoryTransformation"
                            }
                        }
                    ],
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
                },
            ],
            "is_homogeneous": "true"
        },
        "version": "gsprocessing-v1.0"
    }
    else:
        raise NotImplementedError('Only support \"heterogeneous\" and \"homogeneous\" options.' \
            f'but got {gtype}.')

    return conf

def test_GSGraphMetadata():
    """ Test the GSGraphMetadata class.

    GSGraphMetadata contains graph structure information and feature information, without real
    graph data. Here will test:
    1. correct intialization
    2. correct retrieval
    3. incorrect intialization, raise Assertion errors.
    """
    # Test case 1: normal cases
    #       1.1 initialize with only structure information
    gtype_hetero = 'heterogeneous'
    ntypes_hetero = ['ntype1', 'ntype2', 'ntype3']
    etypes_hetero = [('ntype1', 'etype1', 'ntype2'), ('ntype2','etype2', 'ntype3')]

    gmd = GSGraphMetadata(gtype=gtype_hetero,
                          ntypes=ntypes_hetero,
                          etypes=etypes_hetero)
    assert not gmd.is_homogeneous()
    assert gmd.get_ntypes() == ntypes_hetero
    assert gmd.get_etypes() == etypes_hetero
    # predefined ntype shoud be in the metadata
    assert all([gmd.has_ntype(ntype) for ntype in ntypes_hetero])
    assert all([gmd.has_etype(etype) for etype in etypes_hetero])
    # not predefined types should return False
    assert not gmd.has_ntype('an_ntype')
    assert not gmd.has_etype('an_etype')
    # predefined netype should not have any feature name
    assert all([gmd.get_nfeat_all_dims(ntype) is None for ntype in ntypes_hetero])
    assert all([gmd.get_nfeat_all_dims(etype) is None for etype in etypes_hetero])

    #       1.2 initialize with node and edge feature information
    nfeat_dims = {ntype: {'nfeat1': [4, 7]} for ntype in ntypes_hetero}
    efeat_dims = {etype: {'efeat1': [8]} for etype in etypes_hetero}
    gmd = GSGraphMetadata(gtype=gtype_hetero,
                          ntypes=ntypes_hetero,
                          etypes=etypes_hetero,
                          nfeat_dims=nfeat_dims,
                          efeat_dims=efeat_dims)
    # only test feature info as others have been tested in the case 1.1
    assert all([gmd.get_nfeat_all_dims(ntype)=={'nfeat1': [4, 7]} for ntype in ntypes_hetero])
    assert all([gmd.get_efeat_all_dims(etype)=={'efeat1': [8]} for etype in etypes_hetero])

    #       1.3 test to_dict and to_string
    gmd_dict = gmd.to_dict()

    assert gmd_dict['graph_type'] == gtype_hetero
    nodes = gmd_dict['nodes']
    assert all([nodes[i]['node_type'] == ntype for i, ntype in enumerate(ntypes_hetero)])
    assert all([node['features'] == [{'feat_name': 'nfeat1', 'feat_dim': [4, 7]}] for node in nodes])
    edges = gmd_dict['edges']
    assert all([edges[i]['source_node_type'] == can_etype[0] \
                and edges[i]['etype'] == can_etype[1]
                and edges[i]['destination_node_type'] == can_etype[2] \
                    for i, can_etype in enumerate(etypes_hetero)])
    assert all([edge['features'] == [{'feat_name': 'efeat1', 'feat_dim': [8]}] for edge in edges])

    #       1.4 test homogeneous graph with string for node type and tuple for edge type
    gtype_homo = 'homogeneous'
    ntypes_homo = 'ntype1'
    etypes_homo = ('ntype1', 'etype1', 'ntype1')
    nfeat_dims = {ntype: {'nfeat1': [4, 7]} for ntype in [ntypes_homo]}
    efeat_dims = {etype: {'efeat1': [8]} for etype in [etypes_homo]}

    gmd = GSGraphMetadata(gtype=gtype_homo,
                          ntypes=ntypes_homo,
                          etypes=etypes_homo,
                          nfeat_dims=nfeat_dims,
                          efeat_dims=efeat_dims)
    assert gmd.is_homogeneous()
    assert gmd.get_ntypes() == [ntypes_homo]
    assert gmd.get_etypes() == [etypes_homo]
    assert all([gmd.has_ntype(ntype) for ntype in [ntypes_homo]])
    assert all([gmd.has_etype(etype) for etype in [etypes_homo]])
    assert all([gmd.get_nfeat_all_dims(ntype)=={'nfeat1': [4, 7]} for ntype in [ntypes_homo]])
    assert all([gmd.get_efeat_all_dims(etype)=={'efeat1': [8]} for etype in [etypes_homo]])


    #       Test case 2: abnormal cases
    #       2.1 Not supported graph types
    gtype_error = 'hypergraph'
    with pytest.raises(AssertionError, match='Graph types can only be in .* but got'):
        gmd = GSGraphMetadata(gtype=gtype_error,
                              ntypes=None,
                              etypes=None)

    #       2.2 node types are not list or a single string
    gtype_hetero = 'heterogeneous'
    ntypes_hetero = {'ntype1', 'ntype2'}
    with pytest.raises(AssertionError, match='Node types should be in a list of strings or a single .* but got'):
        gmd = GSGraphMetadata(gtype=gtype_hetero,
                              ntypes=ntypes_hetero,
                              etypes=None)

    #       2.3 edge type are not list or a single tuple with 3 elements
    gtype_hetero = 'heterogeneous'
    ntypes_hetero = ['ntype1', 'ntype2', 'ntype3']
    etypes_hetero_error1 = {('ntype1', 'etype1', 'ntype2'), ('ntype2','etype2', 'ntype3')}
    with pytest.raises(AssertionError, match='Edge types should be in a list .* but got'):
        gmd = GSGraphMetadata(gtype=gtype_hetero,
                              ntypes=ntypes_hetero,
                              etypes=etypes_hetero_error1)
    etypes_hetero_error2 = ('etype1')
    with pytest.raises(AssertionError, match='Edge types should be in a list .* but got'):
        gmd = GSGraphMetadata(gtype=gtype_hetero,
                              ntypes=ntypes_hetero,
                              etypes=etypes_hetero_error2)
    etypes_hetero_error3 = [('ntype1', 'etype1', 'ntype2'), ('etype2')]
    with pytest.raises(AssertionError, match='Edge types should be in a list .* but got'):
        gmd = GSGraphMetadata(gtype=gtype_hetero,
                              ntypes=ntypes_hetero,
                              etypes=etypes_hetero_error3)
    etypes_hetero_error4 = [('ntype1', 'etype1', 'ntype2'), ['ntype2','etype2', 'ntype3']]
    with pytest.raises(AssertionError, match='Edge types should be in a list .* but got'):
        gmd = GSGraphMetadata(gtype=gtype_hetero,
                              ntypes=ntypes_hetero,
                              etypes=etypes_hetero_error4)

    #       2.4 node feature dimension must be a dictionary, whose keys are string and values
    #           are another dictionaries with feature name strings as keys and dimension lists
    #           as values.
    gtype_hetero = 'heterogeneous'
    ntypes_hetero = ['ntype1', 'ntype2', 'ntype3']
    etypes_hetero = [('ntype1', 'etype1', 'ntype2'), ('ntype2','etype2', 'ntype3')]
    nfeat_dims_error1 = [[4, 7] for ntype in ntypes_hetero]
    with pytest.raises(AssertionError, match='The node feature dimensions should be .* but got'):
        gmd = GSGraphMetadata(gtype=gtype_hetero,
                              ntypes=ntypes_hetero,
                              etypes=etypes_hetero,
                              nfeat_dims=nfeat_dims_error1)
    nfeat_dims_error2 = {i: {'nfeat1': [4, 7]} for i, ntype in enumerate(ntypes_hetero)}
    with pytest.raises(AssertionError, match='The key of node feature dimensions should be .* but got'):
        gmd = GSGraphMetadata(gtype=gtype_hetero,
                              ntypes=ntypes_hetero,
                              etypes=etypes_hetero,
                              nfeat_dims=nfeat_dims_error2)
    nfeat_dims_error3 = {ntype: {i: [4, 7]} for i, ntype in enumerate(ntypes_hetero)}
    with pytest.raises(AssertionError, match='The feature dimension object should be .* but '):
        gmd = GSGraphMetadata(gtype=gtype_hetero,
                              ntypes=ntypes_hetero,
                              etypes=etypes_hetero,
                              nfeat_dims=nfeat_dims_error3)
    nfeat_dims_error4 = {ntype: {'nfeat1': 4} for i, ntype in enumerate(ntypes_hetero)}
    with pytest.raises(AssertionError, match='The feature dimension object should be .* but'):
        gmd = GSGraphMetadata(gtype=gtype_hetero,
                              ntypes=ntypes_hetero,
                              etypes=etypes_hetero,
                              nfeat_dims=nfeat_dims_error4)
    
    #       2.5 edge feature dimension must be a dictionary, whose keys are 3-element tuples and
    #           values are another dictionarieswith feature name strings as keys and dimension lists
    #           as values.
    efeat_dims_error1 = [etype for etype in etypes_hetero]
    with pytest.raises(AssertionError, match='The edge feature dimensions should be .* but got'):
        gmd = GSGraphMetadata(gtype=gtype_hetero,
                              ntypes=ntypes_hetero,
                              etypes=etypes_hetero,
                              efeat_dims=efeat_dims_error1)
    efeat_dims_error2 = {i: {'efeat1': [8]} for i, etype in enumerate(etypes_hetero)}
    with pytest.raises(AssertionError, match='The key of edge feature dimension dictionary .* but got'):
        gmd = GSGraphMetadata(gtype=gtype_hetero,
                              ntypes=ntypes_hetero,
                              etypes=etypes_hetero,
                              efeat_dims=efeat_dims_error2)
    efeat_dims_error3 = {etype: {i: [8]} for i, etype in enumerate(etypes_hetero)}
    with pytest.raises(AssertionError, match='The feature dimension object should be .* but'):
        gmd = GSGraphMetadata(gtype=gtype_hetero,
                              ntypes=ntypes_hetero,
                              etypes=etypes_hetero,
                              efeat_dims=efeat_dims_error3)
    efeat_dims_error4 = {etype: {'efeat1': 8} for i, etype in enumerate(etypes_hetero)}
    with pytest.raises(AssertionError, match='The feature dimension object should be .* but'):
        gmd = GSGraphMetadata(gtype=gtype_hetero,
                              ntypes=ntypes_hetero,
                              etypes=etypes_hetero,
                              efeat_dims=efeat_dims_error4)

    #       2.6 sanity checks
    with pytest.raises(AssertionError, match='For a homogeneous graph, .* but got'):
        gmd = GSGraphMetadata(gtype=gtype_homo,
                              ntypes=ntypes_hetero,
                              etypes=etypes_hetero)

    ntypes_hetero = ['ntype1', 'ntype2', 'ntype3']
    etypes_hetero_error = [('ntype1', 'etype1', 'ntype2'), ('ntype2','etype2', 'ntype4')]
    with pytest.raises(AssertionError, match='Some node types .* do not exist in the'):
        gmd = GSGraphMetadata(gtype=gtype_hetero,
                              ntypes=ntypes_hetero,
                              etypes=etypes_hetero_error)

def test_config_json_santiy_check():
    """ Test the sanity check of config json contents
    """
    # gsconstruct json
    gcont_config_json = build_gcons_json_example()
    gcont_config_json.pop('version')
    with pytest.raises(AssertionError, match='A \"version\" field must be defined in the'):
        config_json_sanity_check(gcont_config_json)

    gcont_config_json = build_gcons_json_example()
    gcont_config_json.pop('is_homogeneous')
    with pytest.raises(AssertionError, match='A \"is_homogeneous\" field must be defined'):
        config_json_sanity_check(gcont_config_json)

    gcont_config_json = build_gcons_json_example()
    gcont_config_json['is_homogeneous'] = 'Yes'
    with pytest.raises(AssertionError, match='The value of \"is_homogeneous\" can only be'):
        config_json_sanity_check(gcont_config_json)

    gcont_config_json = build_gcons_json_example()
    gcont_config_json.pop('nodes')
    with pytest.raises(AssertionError, match='A \"nodes\" field must be defined in'):
        config_json_sanity_check(gcont_config_json)

    gcont_config_json = build_gcons_json_example()
    gcont_config_json['nodes'] = []
    with pytest.raises(AssertionError, match='Need at least one node in the \"nodes\" object'):
        config_json_sanity_check(gcont_config_json)

    gcont_config_json = build_gcons_json_example()
    gcont_config_json['nodes'][0].pop('node_type')
    with pytest.raises(AssertionError, match='A \"node_type" field must be defined in a node'):
        config_json_sanity_check(gcont_config_json)

    gcont_config_json = build_gcons_json_example()
    gcont_config_json['nodes'][0]['features'][0].pop('feature_name')
    with pytest.raises(AssertionError, match='A \"feature_name\" field must be'):
        config_json_sanity_check(gcont_config_json)

    gcont_config_json = build_gcons_json_example()
    gcont_config_json['nodes'][0]['features'][0].pop('feature_dim')
    with pytest.raises(AssertionError, match='A \"feature_dim\" field must be'):
        config_json_sanity_check(gcont_config_json)

    gcont_config_json = build_gcons_json_example()
    gcont_config_json['nodes'][0]['features'][0]['feature_dim'] = 16
    with pytest.raises(AssertionError, match='Values of \"feature_dim\" field must be a list'):
        config_json_sanity_check(gcont_config_json)

    gcont_config_json = build_gcons_json_example()
    # set two node types to be the same to test duplicated node types.
    gcont_config_json['nodes'][-1]['node_type'] = gcont_config_json['nodes'][0]['node_type']
    with pytest.raises(AssertionError, match='There are duplicated node types in the'):
        config_json_sanity_check(gcont_config_json)

    gcont_config_json = build_gcons_json_example()
    gcont_config_json.pop('edges')
    with pytest.raises(AssertionError, match='An \"edges\" field must be defined in'):
        config_json_sanity_check(gcont_config_json)

    gcont_config_json = build_gcons_json_example()
    gcont_config_json['edges'][0].pop('relation')
    with pytest.raises(AssertionError, match='A \"relation\" field must be defined in an'):
        config_json_sanity_check(gcont_config_json)

    gcont_config_json = build_gcons_json_example()
    gcont_config_json['edges'][2]['features'][0].pop('feature_name')
    with pytest.raises(AssertionError, match='A \"feature_name\" field must be'):
        config_json_sanity_check(gcont_config_json)

    gcont_config_json = build_gcons_json_example()
    gcont_config_json['edges'][2]['features'][0].pop('feature_dim')
    with pytest.raises(AssertionError, match='A \"feature_dim\" field must be'):
        config_json_sanity_check(gcont_config_json)

    gcont_config_json = build_gcons_json_example()
    gcont_config_json['edges'][2]['features'][0]['feature_dim'] = 16
    with pytest.raises(AssertionError, match='Values of \"feature_dim\" field must be a list'):
        config_json_sanity_check(gcont_config_json)

    gcont_config_json = build_gcons_json_example()
    # set two edge types to be the same to test duplicated edge types.
    gcont_config_json['edges'][-1]['relation'] = gcont_config_json['edges'][0]['relation']
    with pytest.raises(AssertionError, match='There are duplicated edge types in the'):
        config_json_sanity_check(gcont_config_json)

    # gsprocessing json
    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json.pop('version')
    with pytest.raises(AssertionError, match='A \"version\" field must be defined in the'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json.pop('graph')
    with pytest.raises(AssertionError, match='A \"graph\" field must be defined in the'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph'].pop('is_homogeneous')
    with pytest.raises(AssertionError, match='An \"is_homogeneous\" field must be defined'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['is_homogeneous'] = 'No'
    with pytest.raises(AssertionError, match='The value of \"is_homogeneous\" can only be'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph'].pop('nodes')
    with pytest.raises(AssertionError, match='A \"nodes\" field must be defined'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['nodes'] = []
    with pytest.raises(AssertionError, match='Need at least one node in the \"nodes\" object.'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['nodes'][0].pop('type')
    with pytest.raises(AssertionError, match='A \"type\" field must be defined in the node'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['nodes'][0]['features'][0].pop('name')
    with pytest.raises(AssertionError, match='A \"name\" field must be defined in a feature'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['nodes'][0]['features'][0].pop('dim')
    with pytest.raises(AssertionError, match='A \"dim\" field must be defined in a feature'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['nodes'][0]['features'][0]['dim'] = 16
    with pytest.raises(AssertionError, match='Values of \"dim\" field must be a list'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['nodes'][0]['type'] = gsproc_config_json['graph']['nodes'][-1]['type']
    with pytest.raises(AssertionError, match='There are duplicated node types in the'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph'].pop('edges')
    with pytest.raises(AssertionError, match='An \"edges\" field must be defined'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['edges'] = []
    with pytest.raises(AssertionError, match='Need at least one edge in the \"edges\" object.'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['edges'][0].pop('source')
    with pytest.raises(AssertionError, match='A \"source\" field must be defined in'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['edges'][0]['source'].pop('type')
    with pytest.raises(AssertionError, match='A \"type\" field must be defined in the source'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['edges'][0].pop('dest')
    with pytest.raises(AssertionError, match='A \"dest\" field must be defined in'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['edges'][0]['dest'].pop('type')
    with pytest.raises(AssertionError, match='A \"type\" field must be defined in the dest'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['edges'][0].pop('relation')
    with pytest.raises(AssertionError, match='A \"relation\" field must be defined in'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['edges'][0]['relation'].pop('type')
    with pytest.raises(AssertionError, match='A \"type\" field must be defined in the relation'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['edges'][2]['features'][0].pop('name')
    with pytest.raises(AssertionError, match='A \"name\" field must be defined in a feature'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['edges'][2]['features'][0].pop('dim')
    with pytest.raises(AssertionError, match='A \"dim\" field must be defined in a feature'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['edges'][2]['features'][0]['dim'] = 16
    with pytest.raises(AssertionError, match='Values of \"dim\" field must be a list'):
        config_json_sanity_check(gsproc_config_json)

    gsproc_config_json = build_gsproc_json_example()
    gsproc_config_json['graph']['edges'][0]['source']['type'] = \
        gsproc_config_json['graph']['edges'][-1]['source']['type']
    gsproc_config_json['graph']['edges'][0]['dest']['type'] = \
        gsproc_config_json['graph']['edges'][-1]['dest']['type']
    gsproc_config_json['graph']['edges'][0]['relation']['type'] = \
        gsproc_config_json['graph']['edges'][-1]['relation']['type']
    with pytest.raises(AssertionError, match='There are duplicated edge types in the'):
        config_json_sanity_check(gsproc_config_json)

    gcont_config_json = build_gcons_json_example()
    gcont_config_json['version'] = 'another_version'
    with pytest.raises(NotImplementedError, match='GSGraphMetadata can only be loaded'):
        config_json_sanity_check(gcont_config_json)

def test_load_metadata_from_json():
    """ Test the load function of json to mateadata.
    
    All field and value checks are done via the config json sanity check function. So will
    test normal cases only.
    """
    # Test case 1: load JSON to be a GSGraphMetadata instance
    #       1.1 load gconst JSON
    with tempfile.TemporaryDirectory() as tmpdirname:
        # heterograph
        config_json_hetero = build_gcons_json_example(gtype='heterogeneous')
        with open(os.path.join(tmpdirname, 'gconstruct_acm_output_config_hetero.json'), "w") as f:
            json.dump(config_json_hetero, f, indent=4)

        with open(os.path.join(tmpdirname, 'gconstruct_acm_output_config_hetero.json'), "r") as f:
            gcon_config_hetero = json.load(f)

        gmd = load_metadata_from_json(gcon_config_hetero)
        expected_ntypes = ['author', 'paper', 'subject']
        expected_etypes = [('author', 'writing', 'paper'),
                           ('paper', 'cited', 'paper'),
                           ('paper', 'citing', 'paper'),
                           ('paper', 'is-about', 'subject'),
                           ('paper', 'written-by', 'author'),
                           ('subject', 'has', 'paper')]
        assert not gmd.is_homogeneous()
        assert sorted(gmd.get_ntypes()) == sorted(expected_ntypes)
        assert set(gmd.get_etypes()) == set(expected_etypes)
        # predefined ntype shoud be in the metadata
        assert all([gmd.has_ntype(ntype) for ntype in expected_ntypes])
        assert all([gmd.has_etype(etype) for etype in expected_etypes])
        # not predefined types should return False
        assert not gmd.has_ntype('an_ntype')
        assert not gmd.has_etype('an_etype')

        expected_nfeat_dims = {
            'author': {'feat':[256]},
            'paper': {'feat':[256]},
            'subject': {'feat':[256]}
        }
        expected_efeat_dims = {
            ('author', 'writing', 'paper'): None,
            ('paper', 'cited', 'paper'): None,
            ('paper', 'citing', 'paper'): {'cate_feat':[6]},
            ('paper', 'is-about', 'subject'): None,
            ('paper', 'written-by', 'author'): None,
            ('subject', 'has', 'paper'): None
        }
        # test feature info
        assert all([gmd.get_nfeat_all_dims(ntype)==nfeat_dims \
                    for ntype, nfeat_dims in expected_nfeat_dims.items()])
        assert all([gmd.get_efeat_all_dims(etype)==efeat_dims \
                    for etype, efeat_dims in expected_efeat_dims.items()])

        # homograph
        config_json_homo = build_gcons_json_example(gtype='homogeneous')
        with open(os.path.join(tmpdirname, 'gconstruct_acm_output_config_homo.json'), "w") as f:
            json.dump(config_json_homo, f, indent=4)

        with open(os.path.join(tmpdirname, 'gconstruct_acm_output_config_homo.json'), "r") as f:
            gcon_config_homo = json.load(f)

        gmd = load_metadata_from_json(gcon_config_homo)
        expected_ntypes = ['_N']
        expected_etypes = [('_N', '_E', '_N')]
        assert gmd.is_homogeneous()
        assert sorted(gmd.get_ntypes()) == sorted(expected_ntypes)
        assert set(gmd.get_etypes()) == set(expected_etypes)
        # predefined ntype shoud be in the metadata
        assert all([gmd.has_ntype(ntype) for ntype in expected_ntypes])
        assert all([gmd.has_etype(etype) for etype in expected_etypes])
        # not predefined types should return False
        assert not gmd.has_ntype('paper')
        assert not gmd.has_etype(('paper', 'citing', 'paper'))

    #       1.2 load gsproc json
    with tempfile.TemporaryDirectory() as tmpdirname:
        # heterograph
        config_json_hetero = build_gsproc_json_example(gtype='heterogeneous')
        with open(os.path.join(tmpdirname, 'gsprocess_acm_output_config_hetero.json'), "w") as f:
            json.dump(config_json_hetero, f, indent=4)

        with open(os.path.join(tmpdirname, 'gsprocess_acm_output_config_hetero.json'), "r") as f:
            gsproc_config_hetero = json.load(f)

        gmd = load_metadata_from_json(gsproc_config_hetero)
        expected_ntypes = ['author', 'paper', 'subject']
        expected_etypes = [('author', 'writing', 'paper'),
                           ('paper', 'cited', 'paper'),
                           ('paper', 'citing', 'paper'),
                           ('paper', 'is-about', 'subject'),
                           ('paper', 'written-by', 'author'),
                           ('subject', 'has', 'paper')]
        assert not gmd.is_homogeneous()
        assert sorted(gmd.get_ntypes()) == sorted(expected_ntypes)
        assert set(gmd.get_etypes()) == set(expected_etypes)
        # predefined ntype shoud be in the metadata
        assert all([gmd.has_ntype(ntype) for ntype in expected_ntypes])
        assert all([gmd.has_etype(etype) for etype in expected_etypes])
        # not predefined types should return False
        assert not gmd.has_ntype('an_ntype')
        assert not gmd.has_etype('an_etype')

        expected_nfeat_dims = {
            'author': {'feat':[256]},
            'paper': {'feat':[256]},
            'subject': {'feat':[256]}
        }
        expected_efeat_dims = {
            ('author', 'writing', 'paper'): None,
            ('paper', 'cited', 'paper'): None,
            ('paper', 'citing', 'paper'): {'cate_feat':[6]},
            ('paper', 'is-about', 'subject'): None,
            ('paper', 'written-by', 'author'): None,
            ('subject', 'has', 'paper'): None
        }
        # test feature info
        assert all([gmd.get_nfeat_all_dims(ntype)==nfeat_dims \
                    for ntype, nfeat_dims in expected_nfeat_dims.items()])
        assert all([gmd.get_efeat_all_dims(etype)==efeat_dims \
                    for etype, efeat_dims in expected_efeat_dims.items()])

        # homograph
        config_json_homo = build_gsproc_json_example(gtype='homogeneous')
        with open(os.path.join(tmpdirname, 'gsprocess_acm_output_config_homo.json'), "w") as f:
            json.dump(config_json_homo, f, indent=4)

        with open(os.path.join(tmpdirname, 'gsprocess_acm_output_config_homo.json'), "r") as f:
            gsproc_config_homo = json.load(f)

        gmd = load_metadata_from_json(gsproc_config_homo)
        expected_ntypes = ['_N']
        expected_etypes = [('_N', '_E', '_N')]
        assert gmd.is_homogeneous()
        assert sorted(gmd.get_ntypes()) == sorted(expected_ntypes)
        assert set(gmd.get_etypes()) == set(expected_etypes)
        # predefined ntype shoud be in the metadata
        assert all([gmd.has_ntype(ntype) for ntype in expected_ntypes])
        assert all([gmd.has_etype(etype) for etype in expected_etypes])
        # not predefined types should return False
        assert not gmd.has_ntype('paper')
        assert not gmd.has_etype(('paper', 'citing', 'paper'))

def test_GSMetadataDglDistGraph():
    """
    The GSMetadataDglDistGraph is a superset of GSMetadataGraph, and DGMetadataDglGraph. So will
    test GSMetadataDglDistGraph class.
    """
    # Test case 1: normal case
    #       1.1 create a heterogeneous metadata dgl distributed graph w/t features
    gtype_hetero = 'heterogeneous'
    ntypes_hetero = ['ntype1', 'ntype2', 'ntype3']
    etypes_hetero = [('ntype1', 'etype1', 'ntype2'), ('ntype2','etype2', 'ntype3')]

    gmd = GSGraphMetadata(gtype=gtype_hetero,
                          ntypes=ntypes_hetero,
                          etypes=etypes_hetero)
    md_dist_g = GSDglDistGraphFromMetadata(gmd, device='cpu')
    # properties check
    assert md_dist_g.ntypes == ntypes_hetero
    assert md_dist_g.etypes == [can_etype[1] for can_etype in etypes_hetero]
    assert md_dist_g.canonical_etypes == etypes_hetero
    assert md_dist_g.device() == 'cpu'
    assert not md_dist_g.is_homogeneous()
    assert all([md_dist_g.nodes[ntype].data=={} for ntype in ntypes_hetero])
    assert all([md_dist_g.edges[etype].data=={} for etype in etypes_hetero])

    # TODO(Jian)       1.2 test metadata graphs w/ features
    nfeat_dims = {ntype: {'nfeat1': [4, 7]} for ntype in ntypes_hetero}
    efeat_dims = {etype: {'efeat1': [8]} for etype in etypes_hetero}

    gmd = GSGraphMetadata(gtype=gtype_hetero,
                          ntypes=ntypes_hetero,
                          etypes=etypes_hetero,
                          nfeat_dims=nfeat_dims,
                          efeat_dims=efeat_dims)
    md_dist_g = GSDglDistGraphFromMetadata(gmd, device='cpu')
    # node and edge feature name check
    assert all([list(md_dist_g.nodes[ntype].data.keys())==['nfeat1'] for ntype in ntypes_hetero])
    assert all([list(md_dist_g.edges[etype].data.keys())==['efeat1'] for etype in etypes_hetero])
    # node and edge feature dimension check
    assert all([list(md_dist_g.nodes[ntype].data['nfeat1'].shape)==[0,4,7] for ntype in ntypes_hetero])
    assert all([list(md_dist_g.edges[etype].data['efeat1'].shape)==[0,8] for etype in etypes_hetero])

    #       1.3 test homogenous metadata graphs
    gtype_homo = 'homogeneous'
    ntypes_homo = 'ntype1'
    etypes_homo = ('ntype1', 'etype1', 'ntype1')
    gmd = GSGraphMetadata(gtype=gtype_homo,
                          ntypes=ntypes_homo,
                          etypes=etypes_homo)
    md_dist_g = GSDglDistGraphFromMetadata(gmd, device='cpu')
    # properties check
    assert md_dist_g.ntypes == [DEFAULT_NTYPE]
    assert md_dist_g.etypes == [DEFAULT_ETYPE[1]]
    assert md_dist_g.canonical_etypes == [DEFAULT_ETYPE]
    assert md_dist_g.device() == 'cpu'
    assert md_dist_g.is_homogeneous()
    assert md_dist_g.ndata == {}
    assert md_dist_g.edata == {}

    # special ndata and edata properties
    nfeat_dims = {ntypes_homo: {'nfeat1': [4, 7]}}
    efeat_dims = {etypes_homo: {'efeat1': [8]}}

    gmd = GSGraphMetadata(gtype=gtype_homo,
                          ntypes=ntypes_homo,
                          etypes=etypes_homo,
                          nfeat_dims=nfeat_dims,
                          efeat_dims=efeat_dims)
    md_dist_g = GSDglDistGraphFromMetadata(gmd, device='cpu')
    assert list(md_dist_g.ndata['nfeat1'].shape) == [0, 4, 7]
    assert list(md_dist_g.edata['efeat1'].shape) == [0, 8]

    #       1.4 Unsupported APIs
    with pytest.raises(NotImplementedError, match='The .* the \"get_node_partition_policy\"'):
        md_dist_g.get_node_partition_policy(DEFAULT_NTYPE)

    with pytest.raises(NotImplementedError, match='The .* the \"get_partition_book\"'):
        md_dist_g.get_partition_book()

    #       1.5 end to end test of building a GSDglDistGraphFromMetadata from a JSON file
    with tempfile.TemporaryDirectory() as tmpdirname:
        # heterograph by gconstruct
        config_json_hetero = build_gcons_json_example(gtype='heterogeneous')
        with open(os.path.join(tmpdirname, 'gconstruct_acm_output_config_hetero.json'), "w") as f:
            json.dump(config_json_hetero, f, indent=4)

        with open(os.path.join(tmpdirname, 'gconstruct_acm_output_config_hetero.json'), "r") as f:
            gcon_config_hetero = json.load(f)

        gmd = load_metadata_from_json(gcon_config_hetero)
        md_dist_g = GSDglDistGraphFromMetadata(gmd, device='cpu')
        expected_ntypes = ['author', 'paper', 'subject']
        expected_etypes = [('author', 'writing', 'paper'),
                           ('paper', 'cited', 'paper'),
                           ('paper', 'citing', 'paper'),
                           ('paper', 'is-about', 'subject'),
                           ('paper', 'written-by', 'author'),
                           ('subject', 'has', 'paper')]
        expected_nfeat_dims = {ntype: {'feat': [256]} for ntype in expected_ntypes}
        expected_efeat_dims = {etype: {'cate_feat': [6]} for etype in [expected_etypes[2]]}

        assert md_dist_g.ntypes == expected_ntypes
        assert md_dist_g.etypes == [can_etype[1] for can_etype in expected_etypes]
        assert md_dist_g.canonical_etypes == expected_etypes
        assert md_dist_g.device() == 'cpu'
        assert not md_dist_g.is_homogeneous()
        assert all(list(md_dist_g.nodes[ntype].data[feat_name].shape) == [0, dim[0]] for \
            ntype, feat_dims in expected_nfeat_dims.items() for feat_name, dim in feat_dims.items())
        assert all(list(md_dist_g.edges[etype].data[feat_name].shape) == [0, dim[0]] for \
            etype, feat_dims in expected_efeat_dims.items() for feat_name, dim in feat_dims.items())
        assert all([md_dist_g.edges[etype].data=={} for etype in \
            expected_etypes[:2] + expected_etypes[3:]])

        # heterograph by gsproc, the assertion should have the same results as gconstruct
        config_json_hetero = build_gsproc_json_example(gtype='heterogeneous')
        with open(os.path.join(tmpdirname, 'gconstruct_acm_output_config_hetero.json'), "w") as f:
            json.dump(config_json_hetero, f, indent=4)

        with open(os.path.join(tmpdirname, 'gconstruct_acm_output_config_hetero.json'), "r") as f:
            gcon_config_hetero = json.load(f)

        gmd = load_metadata_from_json(gcon_config_hetero)
        md_dist_g = GSDglDistGraphFromMetadata(gmd, device='cpu')

        assert md_dist_g.ntypes == expected_ntypes
        assert md_dist_g.etypes == [can_etype[1] for can_etype in expected_etypes]
        assert md_dist_g.canonical_etypes == expected_etypes
        assert md_dist_g.device() == 'cpu'
        assert not md_dist_g.is_homogeneous()
        assert all(list(md_dist_g.nodes[ntype].data[feat_name].shape) == [0, dim[0]] for \
            ntype, feat_dims in expected_nfeat_dims.items() for feat_name, dim in feat_dims.items())
        assert all(list(md_dist_g.edges[etype].data[feat_name].shape) == [0, dim[0]] for \
            etype, feat_dims in expected_efeat_dims.items() for feat_name, dim in feat_dims.items())
        assert all([md_dist_g.edges[etype].data=={} for etype in \
            expected_etypes[:2] + expected_etypes[3:]])

        # homograph from gconstruct
        config_json_homo = build_gcons_json_example(gtype='homogeneous')
        with open(os.path.join(tmpdirname, 'gconstruct_acm_output_config_homo.json'), "w") as f:
            json.dump(config_json_homo, f, indent=4)

        with open(os.path.join(tmpdirname, 'gconstruct_acm_output_config_homo.json'), "r") as f:
            gcon_config_homo = json.load(f)

        gmd = load_metadata_from_json(gcon_config_homo)
        md_dist_g = GSDglDistGraphFromMetadata(gmd, device='cpu')
        expected_ntypes = [DEFAULT_NTYPE]
        expected_etypes = [DEFAULT_ETYPE]
        expected_nfeat_dims = {ntype: {'feat': [256]} for ntype in expected_ntypes}
        expected_efeat_dims = {etype: {'cate_feat': [6]} for etype in expected_etypes}

        assert md_dist_g.ntypes == expected_ntypes
        assert md_dist_g.etypes == [can_etype[1] for can_etype in expected_etypes]
        assert md_dist_g.canonical_etypes == expected_etypes
        assert md_dist_g.device() == 'cpu'
        assert md_dist_g.is_homogeneous()
        assert all(list(md_dist_g.nodes[ntype].data[feat_name].shape) == [0, dim[0]] for \
            ntype, feat_dims in expected_nfeat_dims.items() for feat_name, dim in feat_dims.items())
        assert all(list(md_dist_g.edges[etype].data[feat_name].shape) == [0, dim[0]] for \
            etype, feat_dims in expected_efeat_dims.items() for feat_name, dim in feat_dims.items())

        # homograph from gsproc, the assertion should have the same results as gconstruct
        config_json_homo = build_gsproc_json_example(gtype='homogeneous')
        with open(os.path.join(tmpdirname, 'gconstruct_acm_output_config_homo.json'), "w") as f:
            json.dump(config_json_homo, f, indent=4)

        with open(os.path.join(tmpdirname, 'gconstruct_acm_output_config_homo.json'), "r") as f:
            gcon_config_homo = json.load(f)

        gmd = load_metadata_from_json(gcon_config_homo)
        md_dist_g = GSDglDistGraphFromMetadata(gmd, device='cpu')

        assert md_dist_g.ntypes == expected_ntypes
        assert md_dist_g.etypes == [can_etype[1] for can_etype in expected_etypes]
        assert md_dist_g.canonical_etypes == expected_etypes
        assert md_dist_g.device() == 'cpu'
        assert md_dist_g.is_homogeneous()
        assert all(list(md_dist_g.nodes[ntype].data[feat_name].shape) == [0, dim[0]] for \
            ntype, feat_dims in expected_nfeat_dims.items() for feat_name, dim in feat_dims.items())
        assert all(list(md_dist_g.edges[etype].data[feat_name].shape) == [0, dim[0]] for \
            etype, feat_dims in expected_efeat_dims.items() for feat_name, dim in feat_dims.items())
