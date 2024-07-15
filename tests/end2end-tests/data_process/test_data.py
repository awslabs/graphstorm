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
import os
import json
import dgl
import pyarrow.parquet as pq
import numpy as np
import torch as th
import argparse

from transformers import BertModel, BertConfig

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
assert g.nodes['node1'].data['feat1'].dtype is th.float32
data = g.nodes['node1'].data['feat'].numpy()
data1 = g.nodes['node1'].data['feat1'].numpy()
assert 'input_ids' in g.nodes['node1'].data
assert 'attention_mask' in g.nodes['node1'].data
assert 'token_type_ids' in g.nodes['node1'].data
# Test BERT embeddings.
model_name = "bert-base-uncased"
config = BertConfig.from_pretrained(model_name)
lm_model = BertModel.from_pretrained(model_name, config=config)
with th.no_grad():
    bert_emb = lm_model(g.nodes['node1'].data['input_ids'],
                        g.nodes['node1'].data['attention_mask'].long(),
                        g.nodes['node1'].data['token_type_ids'].long())
assert 'bert' in g.nodes['node1'].data
np.testing.assert_allclose(bert_emb.pooler_output.numpy(),
                           g.nodes['node1'].data['bert'].numpy(),
                           atol=1e-4)
label = g.nodes['node1'].data['label'].numpy()
assert label.dtype == np.int32
orig_ids = np.array([reverse_node1_map[new_id] for new_id in range(g.number_of_nodes('node1'))])
# After graph construction, any 1D features will be converted to 2D features, so
# here need to convert orig_ids to 2D to pass test
np.testing.assert_allclose(data, orig_ids.reshape(-1, 1))
np.testing.assert_allclose(data1, orig_ids.reshape(-1, 1))
assert np.all(label == orig_ids % 100)
assert np.all(np.nonzero(g.nodes['node1'].data['train_mask'].numpy()) == np.arange(100))
assert np.all(np.nonzero(g.nodes['node1'].data['val_mask'].numpy()) == np.arange(100, 200))
assert th.sum(g.nodes['node1'].data['test_mask']) == 0

# test extra node1 feats
data = g.nodes['node1'].data['feat_rank_gauss']
assert data.dtype is th.float32
data = np.sort(data.numpy(), axis=0)
rev_data = np.flip(data, axis=0)
assert np.all(data + rev_data == 0)
data = g.nodes['node1'].data['feat_rank_gauss_fp16']
assert data.dtype is th.float16
data = np.sort(data.numpy(), axis=0)
rev_data = np.flip(data, axis=0)
assert np.all(data + rev_data == 0)
data = g.nodes["node1"].data['feat_multicol']
assert data.dtype is th.float16


#test data type
data = g.nodes['node1'].data['feat2']
assert data.dtype is th.float16
data = g.nodes['node1'].data['feat_bucket']
assert data.dtype is th.float16
data = g.nodes['node1'].data['feat_fp16']
assert data.dtype is th.float16
data = g.nodes['node1'].data['feat_fp16_hdf5']
assert data.dtype is th.float16

# Test the second node data
data = g.nodes['node2'].data['category'].numpy()
assert data.shape[1] == 10
assert data.dtype == np.int8
assert np.all(np.sum(data, axis=1) == 1)
for node_conf in conf["nodes"]:
    if node_conf["node_type"] == "node2":
        assert len(node_conf["features"]) == 1
        print(node_conf["features"][0]["transform"])
        assert node_conf["features"][0]["transform"]["name"] == "to_categorical"
        assert "mapping" in node_conf["features"][0]["transform"]
        assert len(node_conf["features"][0]["transform"]["mapping"]) == 10

# id remap for node4 exists
assert os.path.isdir(os.path.join(out_dir, "raw_id_mappings", "node4"))

# Test the edge data of edge type 1
src_ids, dst_ids = g.edges(etype=('node1', 'relation1', 'node2'))
assert 'label' in g.edges[('node1', 'relation1', 'node2')].data
label = g.edges[('node1', 'relation1', 'node2')].data['label'].numpy()
assert label.dtype == np.int32
src_ids = np.array([reverse_node1_map[src_id] for src_id in src_ids.numpy()])
dst_ids = dst_ids.numpy()
assert np.all((src_ids + dst_ids) % 100 == label)
assert th.sum(g.edges[('node1', 'relation1', 'node2')].data['train_mask']) \
        == int(g.number_of_edges(('node1', 'relation1', 'node2')) * 0.8)
assert th.sum(g.edges[('node1', 'relation1', 'node2')].data['val_mask']) \
        == int(g.number_of_edges(('node1', 'relation1', 'node2')) * 0.2)
assert th.sum(g.edges[('node1', 'relation1', 'node2')].data['test_mask']) == 0

# Test ('node1', 'relation1', 'node2') edge feat
data = g.edges[('node1', 'relation1', 'node2')].data['feat_rank_gauss']
assert data.dtype is th.float32
data = np.sort(data.numpy(), axis=0)
rev_data = np.flip(data, axis=0)
assert np.all(data + rev_data == 0)
data = g.edges[('node1', 'relation1', 'node2')].data['feat_rank_gauss_fp16']
assert data.dtype is th.float16
data = np.sort(data.numpy(), axis=0)
rev_data = np.flip(data, axis=0)
assert np.all(data + rev_data == 0)
data = g.edges[('node1', 'relation1', 'node2')].data['feat_multicol']
assert data.dtype is th.float16

#test data type
data = g.edges[('node1', 'relation1', 'node2')].data['feat1']
assert data.dtype is th.float32
data = g.edges[('node1', 'relation1', 'node2')].data['max_min_norm']
assert data.dtype is th.float32
assert th.max(data) <= 1.0
assert th.min(data) >= 0
data = g.edges[('node1', 'relation1', 'node2')].data['feat_fp16_hdf5']
assert data.dtype is th.float16

# Test the edge data of edge type 2
assert th.sum(g.edges[('node1', 'relation2', 'node1')].data['train_mask']) \
        == int(g.number_of_edges(('node1', 'relation2', 'node1')) * 0.8)
assert th.sum(g.edges[('node1', 'relation2', 'node1')].data['val_mask']) \
        == int(g.number_of_edges(('node1', 'relation2', 'node1')) * 0.2)
assert th.sum(g.edges[('node1', 'relation2', 'node1')].data['test_mask']) == 0

# Test the edge data of edge type 3
src_ids, dst_ids = g.edges(etype=('node2', 'relation3', 'node3'))
feat = g.edges[('node2', 'relation3', 'node3')].data['feat'].numpy()
feat2 = g.edges[('node2', 'relation3', 'node3')].data['feat2'].numpy()
src_ids = src_ids.numpy()
dst_ids = np.array([int(reverse_node3_map[dst_id]) for dst_id in dst_ids.numpy()])
# After graph construction, any 1D features will be converted to 2D features, so
# here need to convert feat back to 1D to pass test
np.testing.assert_allclose(src_ids + dst_ids, feat.reshape(-1,))
np.testing.assert_allclose(src_ids + dst_ids, feat2.reshape(-1,))

assert os.path.exists(os.path.join(out_dir, "node_label_stats.json"))
assert os.path.exists(os.path.join(out_dir, "edge_label_stats.json"))

with open(os.path.join(out_dir, "node_label_stats.json"), 'r') as f:
    node_label_stats = json.load(f)
    assert "node1" in node_label_stats
    assert "label" in node_label_stats["node1"]

with open(os.path.join(out_dir, "edge_label_stats.json"), 'r') as f:
  edge_label_stats = json.load(f)
  assert ("node1,relation1,node2") in edge_label_stats
  assert "label" in edge_label_stats[("node1,relation1,node2")]

# Test customized link prediction data split
src_ids, dst_ids = g.edges(etype=('node1', 'relation_custom', 'node2'))
src_ids = np.array([reverse_node1_map[src_id] for src_id in src_ids.numpy()])
dst_ids = dst_ids.numpy()
assert np.all((src_ids + dst_ids) % 100 == label)
assert th.sum(g.edges[('node1', 'relation_custom', 'node2')].data['train_mask']) \
        == 100
assert th.sum(g.edges[('node1', 'relation_custom', 'node2')].data['val_mask']) \
        == 20
assert th.sum(g.edges[('node1', 'relation_custom', 'node2')].data['test_mask']) == 20

src_ids, dst_ids = g.edges(etype=('node1', 'relation_custom_multi', 'node2'))
src_ids = np.array([reverse_node1_map[src_id] for src_id in src_ids.numpy()])
dst_ids = dst_ids.numpy()
assert np.all((src_ids + dst_ids) % 100 == label)
assert th.sum(g.edges[('node1', 'relation_custom_multi', 'node2')].data['train_mask']) \
        == 100
assert th.sum(g.edges[('node1', 'relation_custom_multi', 'node2')].data['val_mask']) \
        == 20
assert th.sum(g.edges[('node1', 'relation_custom_multi', 'node2')].data['test_mask']) == 20

# Test hard negatives
hard_neg = g.edges[('node1', 'relation2', 'node1')].data["hard_neg"]
src_ids, dst_ids = g.edges(etype=("node1", "relation2", "node1"))
ground_truth = th.cat((src_ids.reshape(-1,1), dst_ids.reshape(-1,1)), dim=1)
assert th.sum(hard_neg-ground_truth) == 0

hard_neg = g.edges[("node2", "relation3", "node3")].data["hard_neg"]
_, dst_ids = g.edges(etype=("node2", "relation3", "node3"))
ground_truth = th.cat((dst_ids.reshape(-1,1), dst_ids.reshape(-1,1)), dim=1)
assert th.sum(hard_neg-ground_truth) == 0

hard_neg = g.edges[("node2", "relation3", "node3")].data["hard_neg2"]
_, dst_ids = g.edges(etype=("node2", "relation3", "node3"))
ground_truth = th.cat([dst_ids.reshape(-1,1), dst_ids.reshape(-1,1), th.full((dst_ids.shape[0], 2), -1, dtype=dst_ids.dtype)], dim=1)
ground_truth[0][2] = dst_ids[0]
ground_truth[0][3] = dst_ids[0]
assert th.sum(hard_neg-ground_truth) == 0
