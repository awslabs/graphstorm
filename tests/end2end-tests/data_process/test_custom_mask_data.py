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

    Test constructed graph data with customized mask
"""
import os
import json
import dgl
import pyarrow.parquet as pq
import numpy as np
import torch as th
import argparse

def read_data_parquet(data_file):
    table = pq.read_table(data_file)
    pd = table.to_pandas()
    return {key: np.array(pd[key]) for key in pd}

argparser = argparse.ArgumentParser("Preprocess graphs")
argparser.add_argument("--graph-format", type=str, required=True,
                       help="The constructed graph format.")
argparser.add_argument("--graph_dir", type=str, required=True,
                       help="The path of the constructed graph.")
argparser.add_argument("--conf_file", type=str, required=True,
                       help="The configuration file.")
args = argparser.parse_args()
out_dir = args.graph_dir
with open(args.conf_file, 'r') as f:
    conf = json.load(f)

# only test DistDGL graph now
if args.graph_format == "DistDGL":
    from dgl.distributed.graph_partition_book import _etype_str_to_tuple
    g, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list = \
            dgl.distributed.load_partition(os.path.join(out_dir, 'test.json'), 0)
    g = dgl.to_heterogeneous(g, ntypes_list, [etype[1] for etype in etypes_list])
    for key, val in node_feats.items():
        ntype, name = key.split('/')
        g.nodes[ntype].data[name] = val
    for key, val in edge_feats.items():
        etype, name = key.split('/')
        etype = _etype_str_to_tuple(etype)
        g.edges[etype].data[name] = val
else:
    raise ValueError('Invalid graph format: {}'.format(args.graph_format))

node1_map = read_data_parquet(os.path.join(out_dir, "raw_id_mappings", "node1"))
reverse_node1_map = {val: key for key, val in zip(node1_map['orig'], node1_map['new'])}
# Test the first node data
assert g.nodes['node1'].data['feat'].dtype is th.float32
assert g.nodes['node1'].data['feat1'].dtype is th.float32
label = g.nodes['node1'].data['label'].numpy()
assert label.dtype == np.int32
data = g.nodes['node1'].data['feat'].numpy()
data1 = g.nodes['node1'].data['feat1'].numpy()
orig_ids = np.array([reverse_node1_map[new_id] for new_id in range(g.number_of_nodes('node1'))])
np.testing.assert_allclose(data, orig_ids.reshape(-1, 1))
np.testing.assert_allclose(data1, orig_ids.reshape(-1, 1))
assert np.all(label == orig_ids % 100)
assert 'train_mask' not in g.nodes['node1'].data
assert 'val_mask' not in g.nodes['node1'].data
assert 'test_mask' not in g.nodes['node1'].data
assert np.all(np.nonzero(g.nodes['node1'].data['train_m'].numpy()) == np.arange(100))
assert np.all(np.nonzero(g.nodes['node1'].data['val_m'].numpy()) == np.arange(100, 200))
assert th.sum(g.nodes['node1'].data['test_m']) == 0

# Test the edge data of edge type 1
src_ids, dst_ids = g.edges(etype=('node1', 'relation1', 'node2'))
assert 'label' in g.edges[('node1', 'relation1', 'node2')].data
label = g.edges[('node1', 'relation1', 'node2')].data['label'].numpy()
assert label.dtype == np.int32
src_ids = np.array([reverse_node1_map[src_id] for src_id in src_ids.numpy()])
dst_ids = dst_ids.numpy()
assert np.all((src_ids + dst_ids) % 100 == label)
assert 'train_mask' not in g.edges[('node1', 'relation1', 'node2')].data
assert 'val_mask' not in g.edges[('node1', 'relation1', 'node2')].data
assert 'test_mask' not in g.edges[('node1', 'relation1', 'node2')].data
assert th.sum(g.edges[('node1', 'relation1', 'node2')].data['train_m']) \
        == int(g.number_of_edges(('node1', 'relation1', 'node2')) * 0.8)
assert th.sum(g.edges[('node1', 'relation1', 'node2')].data['val_m']) \
        == int(g.number_of_edges(('node1', 'relation1', 'node2')) * 0.2)
assert th.sum(g.edges[('node1', 'relation1', 'node2')].data['test_m']) == 0
#test data type
data = g.edges[('node1', 'relation1', 'node2')].data['feat1']
assert data.dtype is th.float32
data = g.edges[('node1', 'relation1', 'node2')].data['max_min_norm']
assert data.dtype is th.float32
assert th.max(data) <= 1.0
assert th.min(data) >= 0

# Test customized link prediction data split
src_ids, dst_ids = g.edges(etype=('node1', 'relation_custom', 'node2'))
src_ids = np.array([reverse_node1_map[src_id] for src_id in src_ids.numpy()])
dst_ids = dst_ids.numpy()
assert np.all((src_ids + dst_ids) % 100 == label)
assert 'train_mask' not in g.edges[('node1', 'relation_custom', 'node2')].data
assert 'val_mask' not in g.edges[('node1', 'relation_custom', 'node2')].data
assert 'test_mask' not in g.edges[('node1', 'relation_custom', 'node2')].data
assert th.sum(g.edges[('node1', 'relation_custom', 'node2')].data['train_m']) \
        == 100
assert th.sum(g.edges[('node1', 'relation_custom', 'node2')].data['val_m']) \
        == 20
assert th.sum(g.edges[('node1', 'relation_custom', 'node2')].data['test_m']) == 20