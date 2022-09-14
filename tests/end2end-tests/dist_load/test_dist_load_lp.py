import dgl
import os
import json
import numpy as np
import torch as th
from torch import nn
import argparse
import time
from m5_dataloaders.datasets.constants import REGRESSION_TASK, CLASSIFICATION_TASK

from graphstorm.data import StandardM5gnnDataset
from graphstorm.data.constants import TOKEN_IDX, VALID_LEN_IDX

import test_dist_load_nc

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("test")
    argparser.add_argument("--path", type=str, default="/data/ml-json")
    argparser.add_argument('--undirected', action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument('--num_parts', type=int, default=2)
    args = argparser.parse_args()
    json_graph_path = args.path
    processed_graph = "ml-output"

    # load meta info
    with open(os.path.join(processed_graph, "metadata.json"), "r") as f:
        meta_info = json.load(f)

    # process using single process
    dataset = StandardM5gnnDataset(json_graph_path,
                                   "ml",
                                   hf_bert_model="bert-base-uncased",
                                   nid_fields={},
                                   src_field="src_id",
                                   dst_field="dst_id",
                                   nlabel_fields={},
                                   ntask_types={},
                                   split_ntypes=[],
                                   elabel_fields={},
                                   etask_types={},
                                   split_etypes=[],
                                   ntext_fields={"movie":["title"]},
                                   max_node_seq_length={"movie": 128},
                                   num_worker=8)
    g = dataset[0]
    if args.undirected:
        edges = {}
        for src_ntype, etype, dst_ntype in g.canonical_etypes:
            src, dst = g.edges(etype=(src_ntype, etype, dst_ntype))
            edges[(src_ntype, etype, dst_ntype)] = (src, dst)
            edges[(dst_ntype, etype + '-rev', src_ntype)] = (dst, src)
        new_g = dgl.heterograph(edges, num_nodes_dict={name: len(nid_map) for name, nid_map in dataset.nid_maps.items()})
        # Copy the node data and edge data to the new graph. The reverse edges will
        # not have data.
        for ntype in g.ntypes:
            for name in g.nodes[ntype].data:
                new_g.nodes[ntype].data[name] = g.nodes[ntype].data[name]
        for etype in g.canonical_etypes:
            for name in g.edges[etype].data:
                new_g.edges[etype].data[name] = g.edges[etype].data[name]
        g = new_g
        new_g = None

    def load_feat(ntype, feat_name, num_part):
        feats = []
        offset = 0
        for i in range(num_part):
            feats.append(np.load(os.path.join(processed_graph, "{}/{}/{}.npy".format(ntype, feat_name, i))))
            assert meta_info["node_data"][ntype][feat_name]["data"][i][1] == offset
            assert meta_info["node_data"][ntype][feat_name]["data"][i][2] == (offset + feats[i].shape[0])
            offset += feats[i].shape[0]
        return np.concatenate(feats, axis=0)
    # load features
    movie_feat = {}
    input_ids = load_feat("movie", "input_ids", args.num_parts)
    valid_len = load_feat("movie", "valid_len", args.num_parts)

    # build dglgraph
    g_dp = test_dist_load_nc.load_processed_graph(args, meta_info)
    g_dp.nodes["movie"].data["input_ids"] = th.tensor(input_ids)
    g_dp.nodes["movie"].data["valid_len"] = th.tensor(valid_len)

    # compare if two graph is identical
    print(g)
    print(g_dp)
    assert g.num_nodes("user") == g_dp.num_nodes("user")
    assert g.num_nodes("movie") == g_dp.num_nodes("movie")
    assert g.num_nodes("occupation") == g_dp.num_nodes("occupation")
    assert g.num_edges(("user", "has-occupation", "occupation")) == g_dp.num_edges(("user", "has-occupation", "occupation"))
    assert g.num_edges(("user", "rating", "movie")) == g_dp.num_edges(("user", "rating", "movie"))
    assert np.array_equal(g.nodes["movie"].data["input_ids"], g_dp.nodes["movie"].data["input_ids"])
    assert np.array_equal(g.nodes["movie"].data["valid_len"], g_dp.nodes["movie"].data["valid_len"])

    edges_g = g.edges(etype="has-occupation")
    edges_g_dp = g_dp.edges(etype="has-occupation")

    src_g, dst_g = edges_g
    src_g_dp, dst_g_dp = edges_g_dp
    sort_idx_g = th.argsort(src_g)
    sort_idx_g_dp = th.argsort(src_g_dp)
    assert np.array_equal(src_g[sort_idx_g].numpy(), src_g_dp[sort_idx_g_dp].numpy())
    assert np.array_equal(dst_g[sort_idx_g].numpy(), dst_g_dp[sort_idx_g_dp].numpy())
