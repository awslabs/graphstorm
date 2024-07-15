import pandas as pd
import numpy as np
import json
import torch
import math
from graphstorm.gconstruct.file_io import write_data_parquet
from graphstorm.gconstruct.file_io import write_index_json

"""
Save graph structure (i.e., nodes and edges information) as parquet files.
"""
# load dataset
data = 'MAG'
df = pd.read_csv('./DATA/{}/edges.csv'.format(data))
num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
print('num_nodes: ', num_nodes)

# Extract graph structure
# We only consider edge types from 100 to 109
timestamp = df['time'].values
mask = np.logical_and(timestamp >= 100, timestamp < 110)

timestamp = timestamp[mask]
src_nodes = df['src'].values[mask]
dst_nodes = df['dst'].values[mask]

# get the new node mappings
unique_nodes = np.unique(
    np.concatenate([src_nodes, dst_nodes])
)

remain_nodes = len(unique_nodes)
old_to_new_nodes = {}
for i, j in enumerate(unique_nodes):
    old_to_new_nodes[j] = i

new_src_nodes = np.array([
    old_to_new_nodes[i] for i in src_nodes
])

new_dst_nodes = np.array([
    old_to_new_nodes[i] for i in dst_nodes
])

# construct graph
etypes = [
    ('paper', 'cite_00', 'paper'),
    ('paper', 'cite_01', 'paper'),
    ('paper', 'cite_02', 'paper'),
    ('paper', 'cite_03', 'paper'),
    ('paper', 'cite_04', 'paper'),
    ('paper', 'cite_05', 'paper'),
    ('paper', 'cite_06', 'paper'),
    ('paper', 'cite_07', 'paper'),
    ('paper', 'cite_08', 'paper'),
    ('paper', 'cite_09', 'paper'),
]


def to_bidirect_remove_duplicate(src_nodes, dst_nodes):
    edges = np.concatenate([
        np.stack([src_nodes, dst_nodes]),
        np.stack([dst_nodes, src_nodes])
    ], axis=1)
    edges = np.unique(edges, axis=1)
    edges = {
        'src_id': edges[0],
        'dst_id': edges[1]
    }
    return edges

for etype in etypes:
    t = int(etype[1].split('_')[1])
    edge_mask = timestamp == (100 + t)
    to_bidirect_remove_duplicate
    edge_data = to_bidirect_remove_duplicate(
        src_nodes=new_src_nodes[edge_mask],
        dst_nodes=new_dst_nodes[edge_mask],
    )
    write_data_parquet(edge_data, f'./DATA/edge-{etype[0]}-{etype[1]}-{etype[2]}.parquet')

# add node features to dgl graph
labels_df = pd.read_csv('./DATA/{}/labels.csv'.format(data))
label_timestamp = labels_df['time'].values

mask = np.logical_and(label_timestamp >= 100, label_timestamp < 110)
label_timestamp = label_timestamp[mask]
node = labels_df['node'].values
label = labels_df['label'].values - 1 # label [0, 1, ..., 151]

# nodes not in train val test set has label -1
new_label = - np.ones(remain_nodes, np.int32)
train_val_test_nodes = []

for i, j in zip(node, label):
    if i in old_to_new_nodes:
        i = old_to_new_nodes[i]
        new_label[i] = j
        train_val_test_nodes.append(i)

# generate data split 80% train 10% valid 10 test
perm = np.random.permutation(len(train_val_test_nodes))
train_val_test_nodes = np.array(train_val_test_nodes)[perm]
train_ids = train_val_test_nodes[: int(0.8 * len(train_val_test_nodes))]
valid_ids = train_val_test_nodes[int(0.8 * len(train_val_test_nodes)): int(0.9 * len(train_val_test_nodes))]
test_ids  = train_val_test_nodes[int(0.9 * len(train_val_test_nodes)): ]
write_index_json(train_ids, './DATA/author_train_idx.json')
write_index_json(valid_ids, './DATA/author_val_idx.json')
write_index_json(test_ids, './DATA/author_test_idx.json')

# node features
node_feats = torch.load('./DATA/{}/node_features.pt'.format(data)).numpy()

data = {
    'id': np.arange(len(unique_nodes)),
    'label': new_label.astype(np.int32),
    'feat': node_feats[unique_nodes].astype(np.float32)
}
chunk_size = math.ceil(len(data['id']) / 20)
chunks = list(range(chunk_size, len(data['id']), chunk_size))
for i, idx in enumerate(np.split(data['id'], chunks)):
    chunk_data = {
        'id': idx,
        'feat': data['feat'][idx],
        'label': data['label'][idx],
    }
    write_data_parquet(chunk_data, f'./DATA/node-paper-{i:03d}.parquet')

"""
Generate the configuration JSON file, which describes the graph structure,
the tasks to perform, the node features, and data file paths.
"""
# prepare partition config
partition_config = {
	"nodes": [
		{
			"node_id_col": "id",
			"node_type": "paper",
			"format": {
				"name": "parquet"
			},
			"files": "./DATA/node-paper-*.parquet",
			"features": [
				{
					"feature_col": "feat",
                    "out_dtype": "float32"
				},
			],
			"labels": [
				{
					"label_col": "label",
					"task_type": "classification",
                    "custom_split_filenames":
                        {
                            "train": "./DATA/author_train_idx.json",
                            "valid": "./DATA/author_val_idx.json",
                            "test": "./DATA/author_test_idx.json"
                        }
				}
			]
		},
	],
	"edges": [
		{
			"source_id_col": "src_id",
			"dest_id_col": "dst_id",
			"relation": [
				etype[0],
				etype[1],
				etype[2]
			],
			"format": {
				"name": "parquet"
			},
			"files": f'./DATA/edge-{etype[0]}-{etype[1]}-{etype[2]}.parquet'
		} for etype in etypes
	]
}

json.dump(
    partition_config,
    open('./DATA/partition_config.json', 'w'),
    indent=4
)