"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Generate graph data for testing multi-task graph construction.
"""

import json
import os
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

from graphstorm.gconstruct.file_io import write_data_parquet, write_data_json, write_data_csv
from graphstorm.gconstruct.file_io import write_data_hdf5, write_index_json

# Here we construct a graph with multiple node types and edge types
# with multiple node labels and edge labels associated with different
# node types and edge types to test the graph construction pipeline
# for multi-task learning. To test the pipeline in a comprehensive way,
# we store node data in multiple cases: 1) different formats,
# 2) with or without node features; 3) one node type has one or multiple
# labels. We create multiple edges in a similar way.
def gen_rand_nid(max_nid, num_nodes):
    node_ids = np.unique(np.random.randint(0, max_nid, num_nodes))
    if len(node_ids) != num_nodes:
        print(f"The length of generated node ids is {len(node_ids)}."
              "But {num_nodes} is needed. Regenerate the node ids.")
        return gen_rand_nid(max_nid, num_nodes)
    return node_ids

np.random.seed(1)
# Generate node, node features and node labels
node_id1 = gen_rand_nid(1000000000, 10000)
node_data1 = {
    'id': node_id1,
    'float': np.random.rand(node_id1.shape[0], 2),
    'label_class': node_id1 % 100,
    'label_reg': np.random.rand(node_id1.shape[0], ),
}

node_id2 = np.arange(20000)
node_data2 = {
    'id': node_id2,
    'label_class': node_id2 % 100,
    'label_class2': node_id2 % 10,
}

node_id3 = gen_rand_nid(1000000000, 5000)
node_id3_str = np.array([str(nid) for nid in node_id3])
node_data3 = {
    'id': node_id3_str,
    'data': np.repeat(node_id3, 5).reshape(len(node_id3), 5)
}

# Generate edge, edge features and edge labels
src1 = node_data1['id'][np.random.randint(0, 9999, 100000)]
dst1 = node_data2['id'][np.random.randint(0, 19999, 100000)]
edge_data1 = {
    'src': src1,
    'dst': dst1,
    'label_class': (src1 + dst1) % 100,
    'label_reg': np.random.rand(src1.shape[0], ),
}
src2 = node_data1['id'][np.random.randint(0, 9999, 50000)]
dst2 = node_data1['id'][np.random.randint(0, 9999, 50000)]
edge_data2 = {
    'src': src2,
    'dst': dst2,
    "hard_neg" : np.concatenate((src2.reshape(-1,1), dst2.reshape(-1,1)), axis=1).astype(str)
}

src3 = node_data2['id'][np.random.randint(0, 20000, 100000)]
dst_idx = np.random.randint(0, 5000, 100000)
dst3 = node_data3['id'][dst_idx]
edge_data3 = {
    'src': src3,
    'dst': dst3,
    'label_class1': src3 % 100,
    'label_class2': dst3.astype(np.int32) % 20
}

in_dir = '/tmp/multitask_test_data/'
out_dir = '/tmp/multitask_test_out/'

os.makedirs(in_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

def split_data(data, num):
    new_data_list = [{} for _ in range(num)]
    for key, val in data.items():
        for i, val in enumerate(np.array_split(val, num)):
            new_data_list[i][key] = val
    return new_data_list

write_index_json(node_data1['id'][:100], os.path.join(in_dir, 'node1_train.json'))
write_index_json(node_data1['id'][100:200], os.path.join(in_dir, 'node1_valid.json'))

for i, node_data in enumerate(split_data(node_data1, 5)):
    write_data_parquet(node_data, os.path.join(in_dir, f'node_data1_{i}.parquet'))
for i, node_data in enumerate(split_data(node_data2, 5)):
    write_data_json(node_data, os.path.join(in_dir, f'node_data2_{i}.json'))
for i, node_data in enumerate(split_data(node_data3, 10)):
    write_data_parquet(node_data, os.path.join(in_dir, f'node_data3_{i}.parquet'))
for i, edge_data in enumerate(split_data(edge_data1, 10)):
    write_data_csv(edge_data, os.path.join(in_dir, f'edge_data1_{i}.csv'))
for i, edge_data in enumerate(split_data(edge_data2, 10)):
    write_data_parquet(edge_data, os.path.join(in_dir, f'edge_data2_{i}.parquet'))
for i, edge_data in enumerate(split_data(edge_data3, 10)):
    write_data_parquet(edge_data, os.path.join(in_dir, f'edge_data3_{i}.parquet'))

node_conf = [
    {
        "node_id_col": "id",
        "node_type": "node1",
        "format": {"name": "parquet"},
        "files": os.path.join(in_dir, "node_data1_*.parquet"),
        "features": [
            {
                "feature_col": "float",
                "feature_name": "feat",
            }
        ],
        "labels":[
            {
                "label_col":    "label_class",
                "task_type":    "classification",
                "label_stats_type": "frequency_cnt",
                "mask_field_names": ["train_mask_class",
                                     "val_mask_class",
                                     "test_mask_class"],
                "custom_split_filenames": {"train": os.path.join(in_dir, 'node1_train.json'),
                                           "valid": os.path.join(in_dir, 'node1_valid.json')},
            },
            {
                "label_col":    "label_reg",
                "task_type":    "regression",
                "mask_field_names": ["train_mask_reg",
                                     "val_mask_reg",
                                     "test_mask_reg"],
            },
        ],
    },
    {
        "node_id_col": "id",
        "node_type": "node2",
        "format": {"name": "json"},
        "files": os.path.join(in_dir, "node_data2_*.json"),
        "labels":[
            {
                "label_col":    "label_class",
                "task_type":    "classification",
                "label_stats_type": "frequency_cnt",
                "mask_field_names": ["train_mask_class",
                                     "val_mask_class",
                                     "test_mask_class"],
            },
            {
                "label_col":    "label_class2",
                "task_type":    "classification",
                "label_stats_type": "frequency_cnt",
                "mask_field_names": ["train_mask_class2",
                                     "val_mask_class2",
                                     "test_mask_class2"],
            },
        ]
    },
    {
        "node_id_col": "id",
        "node_type": "node3",
        "format": {"name": "parquet"},
        "files": os.path.join(in_dir, "node_data3_*.parquet"),
        "features": [
            {
                "feature_col": "data",
                "feature_name": "feat",
            },
        ]
    }
]

edge_conf = [
    {
        "source_id_col":    "src",
        "dest_id_col":      "dst",
        "relation":         ("node1", "relation1", "node2"),
        "format":           {"name": "csv"},
        "files":            os.path.join(in_dir, "edge_data1_*.csv"),
        "labels":       [
            {
                "label_col":    "label_class",
                "task_type":    "classification",
                "split_pct":   [0.8, 0.2, 0.0],
                "label_stats_type": "frequency_cnt",
                "mask_field_names": ["train_mask_class",
                                     "val_mask_class",
                                     "test_mask_class"],
            },
            {
                "label_col":    "label_reg",
                "task_type":    "regression",
                "split_pct":   [0.8, 0.1, 0.1],
                "mask_field_names": ["train_mask_reg",
                                     "val_mask_reg",
                                     "test_mask_reg"],
            },
        ],
    },
    {
        "source_id_col":    "src",
        "dest_id_col":      "dst",
        "relation":         ("node1", "relation2", "node1"),
        "format":           {"name": "parquet"},
        "files":            os.path.join(in_dir, "edge_data2_*.parquet"),
        "labels":       [
            {
                "task_type":    "link_prediction",
                "split_pct":   [0.8, 0.2, 0.0]
            },
        ],
        "features": [
            {
                "feature_col": "hard_neg",
                "feature_name": "hard_neg",
                "transform": {"name": "edge_dst_hard_negative"}
            }
        ],
    },
    {
        "source_id_col":    "src",
        "dest_id_col":      "dst",
        "relation":         ("node2", "relation3", "node3"),
        "format":           {"name": "parquet"},
        "files":            os.path.join(in_dir, "edge_data3_*.parquet"),
        "labels":       [
            {
                "task_type":    "link_prediction",
                "mask_field_names": ["train_mask_lp",
                                     "val_mask_lp",
                                     "test_mask_lp"],
            },
            {
                "label_col":    "label_class1",
                "task_type":    "classification",
                "split_pct":   [0.8, 0.2, 0.0],
                "label_stats_type": "frequency_cnt",
                "mask_field_names": ["train_mask_class1",
                                     "val_mask_class1",
                                     "test_mask_class1"],
            },
            {
                "label_col":    "label_class2",
                "task_type":    "classification",
                "split_pct":   [0.8, 0.1, 0.1],
                "mask_field_names": ["train_mask_class2",
                                     "val_mask_class2",
                                     "test_mask_class2"],
            },
        ],
    },
]

transform_conf = {
    "nodes": node_conf,
    "edges": edge_conf,
}
json.dump(transform_conf, open(os.path.join(in_dir, 'test_multitask_data_transform.conf'), 'w'), indent=4)
