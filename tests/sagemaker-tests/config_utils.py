"""
    Copyright Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Create dummy configurations for sagemaker unit tests
"""

import os
import json


def create_graph_config_json_object(tmpdirname, has_tokenize=True, json_fname='ml.json'):
    """ Create a real graph construction json file
    """
    json_object = \
    {
        "version": "gconstruct-runtime-v0.1",
        "nodes": [
                {
                        "node_id_col":  "id",
                        "node_type":    "user",
                        "format":       {"name": "hdf5"},
                        "files":        "/data/ml-100k/user.hdf5",
                        "features":     [
                            {
                                    "feature_col":  "feat",
                                    "feature_name": "feat",
                                    "feature_dim": [
                                        2
                                    ]
                            }
                        ]
                },
                {
                        "node_id_col":  "id",
                        "node_type":    "movie",
                        "format":       {"name": "parquet"},
                        "files":        "/data/ml-100k/movie.parquet",
                        "features":     [
                            {
                                    "feature_col":  "feat",
                                    "feature_name": "feat",
                                    "feature_dim": [
                                        2
                                    ]
                            },
                            {
                                "feature_col":  "title",
                                "transform":    {
                                        "name": "bert_hf",
                                        "bert_model": "bert-base-uncased",
                                        "max_seq_length": 16
                                },
                                "feature_name": "title",
                                "feature_dim": [
                                    16
                                ]
                        }
                    ],
                        "labels":	[
                            {
                                "label_col":	"label",
                                "task_type":	"classification",
                                "split_pct":	[0.8, 0.1, 0.1]
                            }
                        ]
                }
        ],
        "edges": [
                {
                        "source_id_col":    "src_id",
                        "dest_id_col":      "dst_id",
                        "relation":         ["user", "rating", "movie"],
                        "format":           {"name": "parquet"},
                        "files":        "/data/ml-100k/edges.parquet",
                        "labels":	[
                            {
                                "label_col":	"rate",
                                "task_type":	"classification",
                                "split_pct":	[0.1, 0.1, 0.1]
                            }
                        ]
                },
                {
                        "source_id_col":    "src_id",
                        "dest_id_col":      "dst_id",
                        "relation":         ["movie", "rating-rev", "user"],
                        "format":           {"name": "parquet"},
                        "files":        "/data/ml-100k/edges_rev.parquet",
                }
        ],
        "is_homogeneous": "false",
        "add_reverse_edges": "true"
    }

    if has_tokenize:
        feat2 = {
                    "feature_col":  "text",
                    "feature_name": "text",
                    "transform": {"name": "tokenize_hf",
                                    "bert_model": "bert-base-uncased",
                                    "max_seq_length": 16},
                    "feature_dim": [
                        784
                    ]
                }
        json_object["nodes"][0]["features"]= feat2

    with open(os.path.join(tmpdirname, json_fname), "w") as f:
        json.dump(json_object, f, indent=4)

    with open(os.path.join(tmpdirname, json_fname), "r") as f:
        gs_json = json.load(f)

    return gs_json
