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

    Generate example graph data using built-in datasets for node classifcation,
    node regression, edge classification and edge regression.
"""
import copy
import json
import os
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

from graphstorm.gconstruct.file_io import write_data_parquet, write_data_json
from graphstorm.gconstruct.file_io import write_data_hdf5

# Here we construct a graph with multiple node types and edge types
# to test the graph construction pipeline. To test the pipeline
# in a comprehensive way, we store node data in multiple cases:
# 1) different formats, 2) with or without node features, 3) with or
# without labels, 4) features of different types, 5) node features
# with different dimensions.
# We create multiple edges in a similar way.

node_id1 = np.unique(np.random.randint(0, 1000000000, 10000))
node_text = np.array([str(nid) for nid in node_id1])
node_data1 = {
    'id': node_id1,
    'data': node_id1,
    'text': node_text,
    'label': node_id1 % 100,
    'float_max_min': np.random.rand(node_id1.shape[0], 2),
    'float2': node_id1,
}

node_data1_2 = {
    'data': node_id1,
    'float_2': np.random.rand(node_id1.shape[0], 2),
    'float_feat_rank_gauss': np.random.rand(node_id1.shape[0], 2),
    'float_feat_rank_gauss_fp16': np.random.rand(node_id1.shape[0], 2),
    'float_max_min_2': np.random.rand(node_id1.shape[0], 2),
    'float3': node_id1,
}

node_id2 = np.arange(20000)
node_data2 = {
    'id': node_id2,
    'data': [str(i) for i in np.random.randint(0, 10, len(node_id2) - 10)] \
            + [str(i) for i in range(10)],
}

node_id3 = np.unique(np.random.randint(0, 1000000000, 5000))
node_id3_str = np.array([str(nid) for nid in node_id3])
node_data3 = {
    'id': node_id3_str,
    'data': np.repeat(node_id3, 5).reshape(len(node_id3), 5),
    'data1': node_id3,
}

node_id4 = np.unique(np.random.randint(0, 1000000000, 5000))
node_id4_str = np.array([str(nid) for nid in node_id3])
node_data4 = {
    'id': node_id4_str,
}

src1 = node_data1['id'][np.random.randint(0, 9999, 100000)]
dst1 = node_data2['id'][np.random.randint(0, 19999, 100000)]
edge_data1 = {
    'src': src1,
    'dst': dst1,
    'label': (src1 + dst1) % 100,
}
edge_data2 = {
    'src': node_data1['id'][np.random.randint(0, 9999, 50000)],
    'dst': node_data1['id'][np.random.randint(0, 9999, 50000)],
}

edge_data1_2_float = np.random.rand(src1.shape[0], 10) * 2
edge_data1_2 = {
    'float1': edge_data1_2_float,
    'float1_fp16': edge_data1_2_float,
    'float_feat_rank_gauss': np.random.rand(src1.shape[0], 2),
    'float_feat_rank_gauss_fp16': np.random.rand(src1.shape[0], 2),
    'float1_max_min': edge_data1_2_float,
}

src3 = node_data2['id'][np.random.randint(0, 20000, 100000)]
dst_idx = np.random.randint(0, 5000, 100000)
edge_data3 = {
    'src': src3,
    'dst': node_data3['id'][dst_idx],
    'data': src3 + node_id3[dst_idx],
    'data1': np.repeat(src3 + node_id3[dst_idx], 5).reshape(len(src3), 5),
}
edge_data3_2 = {
    'data': src3 + node_id3[dst_idx],
}

in_dir = '/tmp/test_data/'
out_dir = '/tmp/test_out/'

os.makedirs(in_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

def split_data(data, num):
    new_data_list = [{} for _ in range(num)]
    for key, val in data.items():
        for i, val in enumerate(np.array_split(val, num)):
            new_data_list[i][key] = val
    return new_data_list

for i, node_data in enumerate(split_data(node_data1, 5)):
    write_data_parquet(node_data, os.path.join(in_dir, f'node_data1_{i}.parquet'))
write_data_hdf5(node_data1_2, os.path.join(in_dir, f'node_data1_2.hdf5'))
for i, node_data in enumerate(split_data(node_data2, 5)):
    write_data_parquet(node_data, os.path.join(in_dir, f'node_data2_{i}.parquet'))
for i, node_data in enumerate(split_data(node_data3, 10)):
    write_data_json(node_data, os.path.join(in_dir, f'node_data3_{i}.json'))
for i, node_data in enumerate(split_data(node_data4, 10)):
    write_data_json(node_data, os.path.join(in_dir, f'node_data4_{i}.json'))
for i, edge_data in enumerate(split_data(edge_data1, 10)):
    write_data_parquet(edge_data, os.path.join(in_dir, f'edge_data1_{i}.parquet'))
for i, edge_data in enumerate(split_data(edge_data2, 10)):
    write_data_parquet(edge_data, os.path.join(in_dir, f'edge_data2_{i}.parquet'))
write_data_hdf5(edge_data1_2, os.path.join(in_dir, f'edge_data1_2.hdf5'))
for i, edge_data in enumerate(split_data(edge_data3, 10)):
    write_data_parquet(edge_data, os.path.join(in_dir, f'edge_data3_{i}.parquet'))
write_data_hdf5(edge_data3_2, os.path.join(in_dir, f'edge_data3_2.hdf5'))

node_conf = [
    {
        "node_type": "node1",
        "format": {"name": "hdf5"},
        "files": os.path.join(in_dir, "node_data1_2.hdf5"),
        "features": [
            {
                "feature_col": "data",
                "feature_name": "feat1",
            },
            {
                "feature_col": "float_max_min_2",
                "feature_name": "feat3",
                "transform": {"name": 'max_min_norm'}
            },
            {
                "feature_col": "float_feat_rank_gauss",
                "feature_name": "feat_rank_gauss",
                "transform": {"name": 'rank_gauss'}
            },
            {
                "feature_col": "float_feat_rank_gauss_fp16",
                "feature_name": "feat_rank_gauss_fp16",
                "out_dtype": 'float16',
                "transform": {"name": 'rank_gauss'}
            },
            {
                "feature_col": "float3",
                "feature_name": "feat_fp16_hdf5",
                "out_dtype": 'float16',
            },
        ],
    },
    {
        "node_id_col": "id",
        "node_type": "node1",
        "format": {"name": "parquet"},
        "files": os.path.join(in_dir, "node_data1_*.parquet"),
        "features": [
            {
                "feature_col": "data",
                "feature_name": "feat",
            },
            {
                "feature_col": "text",
                # tokenize_hf generates multiple features.
                # It defines feature names itself.
                "transform": {"name": "tokenize_hf",
                              "bert_model": "/root/.cache/huggingface/hub/models--bert-base-uncased/snapshots/a265f773a47193eed794233aa2a0f0bb6d3eaa63/",
                              "max_seq_length": 16},
            },
            {
                "feature_col": "text",
                "feature_name": "bert",
                "out_dtype": 'float32',
                "transform": {"name": "bert_hf",
                              "bert_model": "/root/.cache/huggingface/hub/models--bert-base-uncased/snapshots/a265f773a47193eed794233aa2a0f0bb6d3eaa63/",
                              "max_seq_length": 16},
            },
            {
                "feature_col": "float_max_min",
                "feature_name": "feat2",
                "out_dtype": 'float16',
                "transform": {"name": "max_min_norm",
                              "max_bound": 2.,
                              "min_bound": -2.}
            },
            {
                "feature_col": "float2",
                "feature_name": "feat_fp16",
                "out_dtype": 'float16',
            },
        ],
        "labels":       [
            {
                "label_col":    "label",
                "task_type":    "classification",
                "split_pct":   [0.8, 0.2, 0.0],
            },
        ],
    },
    {
        "node_id_col": "id",
        "node_type": "node2",
        "format": {"name": "parquet"},
        "files": os.path.join(in_dir, "node_data2_*.parquet"),
        "features": [
            {
                "feature_col": "data",
                "feature_name": "category",
                "transform": {"name": "to_categorical"},
            },
        ],
    },
    {
        "node_id_col": "id",
        "node_type": "node3",
        "format": {"name": "json"},
        "files": os.path.join(in_dir, "node_data3_*.json"),
        "features": [
            {
                "feature_col": "data",
                "feature_name": "feat",
            },
        ],
    },
    {
        "node_id_col": "id",
        "node_type": "node4",
        "format": {"name": "json"},
        "files": os.path.join(in_dir, "node_data4_*.json"),
        # No feature
    },
]
edge_conf = [
    {
        "source_id_col":    "src",
        "dest_id_col":      "dst",
        "relation":         ("node1", "relation1", "node2"),
        "format":           {"name": "parquet"},
        "files":            os.path.join(in_dir, "edge_data1_*.parquet"),
        "labels":       [
            {
                "label_col":    "label",
                "task_type":    "classification",
                "split_pct":   [0.8, 0.2, 0.0],
            },
        ],
    },
    {
        "relation": ("node1", "relation1", "node2"),
        "format": {"name": "hdf5"},
        "files": os.path.join(in_dir, "edge_data1_2.hdf5"),
        "features": [
            {
                "feature_col": "float1",
                "feature_name": "feat1",
            },
            {
                "feature_col": "float1_max_min",
                "feature_name": "max_min_norm",
                "transform": {"name": 'max_min_norm'}
            },
            {
                "feature_col": "float_feat_rank_gauss",
                "feature_name": "feat_rank_gauss",
                "transform": {"name": 'rank_gauss'}
            },
            {
                "feature_col": "float_feat_rank_gauss_fp16",
                "feature_name": "feat_rank_gauss_fp16",
                "out_dtype": 'float16',
                "transform": {"name": 'rank_gauss'}
            },
            {
                "feature_col": "float1_fp16",
                "feature_name": "feat_fp16_hdf5",
                "out_dtype": 'float16',
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
                "split_pct":   [0.8, 0.2, 0.0],
            },
        ],
    },
    {
        "source_id_col":    "src",
        "dest_id_col":      "dst",
        "relation":         ("node2", "relation3", "node3"),
        "format":           {"name": "parquet"},
        "files":            os.path.join(in_dir, "edge_data3_*.parquet"),
        "features": [
            {
                "feature_col": "data",
                "feature_name": "feat",
            },
        ],
    },
    {
        "relation":         ("node2", "relation3", "node3"),
        "format":           {"name": "hdf5"},
        "files":            os.path.join(in_dir, "edge_data3_2.hdf5"),
        "features": [
            {
                "feature_col": "data",
                "feature_name": "feat2",
            },
        ],
    },
]
transform_conf = {
    "nodes": node_conf,
    "edges": edge_conf,
}
json.dump(transform_conf, open(os.path.join(in_dir, 'test_data_transform.conf'), 'w'), indent=4)
