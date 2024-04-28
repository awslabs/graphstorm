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
argparser.add_argument("--with-reverse-edge",
                       type=lambda x: (str(x).lower() in ['true', '1']),
                       default=False,
                       help="Whether check reverse edges")
args = argparser.parse_args()
out_dir = args.graph_dir
with open(args.conf_file, 'r') as f:
    conf = json.load(f)

if args.graph_format == "DGL":
    g = dgl.load_graphs(os.path.join(out_dir, "test.dgl"))[0][0]
elif args.graph_format == "DistDGL":
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
node3_map = read_data_parquet(os.path.join(out_dir, "raw_id_mappings", "node3"))
reverse_node3_map = {val: key for key, val in zip(node3_map['orig'], node3_map['new'])}

# Test the first node data
assert g.nodes['node1'].data['feat'].dtype is th.float32
data = g.nodes['node1'].data['feat'].numpy()
orig_ids = np.array([reverse_node1_map[new_id] for new_id in range(g.number_of_nodes('node1'))])
assert np.all(data <= 2)
assert np.all(data >= -2)

label_class = g.nodes['node1'].data['label_class'].numpy()
assert label_class.dtype == np.int32
assert np.all(label_class == orig_ids % 100)
assert np.all(np.nonzero(g.nodes['node1'].data['train_mask_class'].numpy()) == np.arange(100))
assert np.all(np.nonzero(g.nodes['node1'].data['val_mask_class'].numpy()) == np.arange(100, 200))
assert th.sum(g.nodes['node1'].data['test_mask_class']) == 0

label_reg = g.nodes['node1'].data['label_reg'].numpy()
assert np.issubdtype(label_reg.dtype, np.floating)
assert th.sum(g.nodes['node1'].data['train_mask_reg']) == int(g.number_of_nodes('node1') * 0.8)
assert th.sum(g.nodes['node1'].data['val_mask_reg']) == int(g.number_of_nodes('node1') * 0.1)
assert th.sum(g.nodes['node1'].data['test_mask_reg']) == int(g.number_of_nodes('node1') * 0.1)

# Test the second node data
label_class_1 = g.nodes['node2'].data['label_class'].numpy()
assert label_class_1.dtype == np.int32
assert th.sum(g.nodes['node2'].data['train_mask_class']) == int(g.number_of_nodes('node2') * 0.8)
assert th.sum(g.nodes['node2'].data['val_mask_class']) == int(g.number_of_nodes('node2') * 0.1)
assert th.sum(g.nodes['node2'].data['test_mask_class']) == int(g.number_of_nodes('node2') * 0.1)

label_class_2 = g.nodes['node2'].data['label_class2'].numpy()
assert label_class_1.dtype == np.int32
assert th.sum(g.nodes['node2'].data['train_mask_class2']) == int(g.number_of_nodes('node2') * 0.8)
assert th.sum(g.nodes['node2'].data['val_mask_class2']) == int(g.number_of_nodes('node2') * 0.1)
assert th.sum(g.nodes['node2'].data['test_mask_class2']) == int(g.number_of_nodes('node2') * 0.1)

# Test the third node data
assert g.nodes['node3'].data['feat'].dtype is th.float32
data = g.nodes['node3'].data['feat'].numpy()
orig_ids = np.array([reverse_node3_map[new_id] for new_id in range(g.number_of_nodes('node3'))]).astype(np.float32)
np.testing.assert_allclose(data, np.repeat(orig_ids, 5).reshape(len(orig_ids), 5))

# Test the edge data of edge type 1
src_ids, dst_ids = g.edges(etype=('node1', 'relation1', 'node2'))
assert 'label_class' in g.edges[('node1', 'relation1', 'node2')].data
label_class = g.edges[('node1', 'relation1', 'node2')].data['label_class'].numpy()
assert label_class.dtype == np.int32
src_ids = np.array([reverse_node1_map[src_id] for src_id in src_ids.numpy()])
dst_ids = dst_ids.numpy()
assert np.all((src_ids + dst_ids) % 100 == label_class)
assert th.sum(g.edges[('node1', 'relation1', 'node2')].data['train_mask_class']) \
        == int(g.number_of_edges(('node1', 'relation1', 'node2')) * 0.8)
assert th.sum(g.edges[('node1', 'relation1', 'node2')].data['val_mask_class']) \
        == int(g.number_of_edges(('node1', 'relation1', 'node2')) * 0.2)
assert th.sum(g.edges[('node1', 'relation1', 'node2')].data['test_mask_class']) == 0

assert 'label_reg' in g.edges[('node1', 'relation1', 'node2')].data
assert th.sum(g.edges[('node1', 'relation1', 'node2')].data['train_mask_reg']) \
        == int(g.number_of_edges(('node1', 'relation1', 'node2')) * 0.8)
assert th.sum(g.edges[('node1', 'relation1', 'node2')].data['val_mask_reg']) \
        == int(g.number_of_edges(('node1', 'relation1', 'node2')) * 0.1)
assert th.sum(g.edges[('node1', 'relation1', 'node2')].data['test_mask_reg']) \
        == int(g.number_of_edges(('node1', 'relation1', 'node2')) * 0.1)

# Test the edge data of edge type 2
assert th.sum(g.edges[('node1', 'relation2', 'node1')].data['train_mask']) \
        == int(g.number_of_edges(('node1', 'relation2', 'node1')) * 0.8)
assert th.sum(g.edges[('node1', 'relation2', 'node1')].data['val_mask']) \
        == int(g.number_of_edges(('node1', 'relation2', 'node1')) * 0.2)
assert th.sum(g.edges[('node1', 'relation2', 'node1')].data['test_mask']) == 0

# Test hard negatives
hard_neg = g.edges[('node1', 'relation2', 'node1')].data["hard_neg"]
src_ids, dst_ids = g.edges(etype=("node1", "relation2", "node1"))
ground_truth = th.cat((src_ids.reshape(-1,1), dst_ids.reshape(-1,1)), dim=1)
assert th.sum(hard_neg-ground_truth) == 0

# Test the edge data of edge type 3
src_ids, dst_ids = g.edges(etype=('node2', 'relation3', 'node3'))
src_ids = src_ids.numpy()
dst_ids = dst_ids.numpy()
assert th.sum(g.edges[('node2', 'relation3', 'node3')].data['train_mask_lp']) \
        == int(g.number_of_edges(('node2', 'relation3', 'node3')) * 0.8)
assert th.sum(g.edges[('node2', 'relation3', 'node3')].data['val_mask_lp']) \
        == int(g.number_of_edges(('node2', 'relation3', 'node3')) * 0.1)
assert th.sum(g.edges[('node2', 'relation3', 'node3')].data['test_mask_lp']) \
        == int(g.number_of_edges(('node2', 'relation3', 'node3')) * 0.1)

assert 'label_class1' in g.edges[('node2', 'relation3', 'node3')].data
assert th.sum(g.edges[('node2', 'relation3', 'node3')].data['train_mask_class1']) \
        == int(g.number_of_edges(('node2', 'relation3', 'node3')) * 0.8)
assert th.sum(g.edges[('node2', 'relation3', 'node3')].data['val_mask_class1']) \
        == int(g.number_of_edges(('node2', 'relation3', 'node3')) * 0.2)
assert th.sum(g.edges[('node2', 'relation3', 'node3')].data['test_mask_class1']) == 0
assert np.all((src_ids % 100) == g.edges[('node2', 'relation3', 'node3')].data['label_class1'].numpy())

assert 'label_class2' in g.edges[('node2', 'relation3', 'node3')].data
assert th.sum(g.edges[('node2', 'relation3', 'node3')].data['train_mask_class2']) \
        == int(g.number_of_edges(('node2', 'relation3', 'node3')) * 0.8)
assert th.sum(g.edges[('node2', 'relation3', 'node3')].data['val_mask_class2']) \
        == int(g.number_of_edges(('node2', 'relation3', 'node3')) * 0.1)
assert th.sum(g.edges[('node2', 'relation3', 'node3')].data['test_mask_class2']) \
        == int(g.number_of_edges(('node2', 'relation3', 'node3')) * 0.1)
dst_ids = np.array([reverse_node3_map[dst_id] for dst_id in dst_ids]).astype(np.int32)
assert np.all((dst_ids % 20) == g.edges[('node2', 'relation3', 'node3')].data['label_class2'].numpy())

if args.with_reverse_edge:
    assert g.number_of_edges(('node1', 'relation1', 'node2')) == g.number_of_edges(('node2', 'relation1-rev', 'node1'))
    assert 'label_class' not in g.edges[('node2', 'relation1-rev', 'node1')].data

    assert g.number_of_edges(('node1', 'relation2', 'node1')) == g.number_of_edges(('node1', 'relation2-rev', 'node1'))
    assert 'train_mask' not in g.edges[('node1', 'relation2-rev', 'node1')].data

    assert g.number_of_edges(('node2', 'relation3', 'node3')) == g.number_of_edges(('node3', 'relation3-rev', 'node2'))
    assert 'label_class1' not in g.edges[('node3', 'relation3-rev', 'node2')].data
