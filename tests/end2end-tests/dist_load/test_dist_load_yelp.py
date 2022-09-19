import dgl
import os
import json
import numpy as np
import torch as th
from torch import nn
import argparse
import time
from graphstorm.data.constants import CLASSIFICATION_TASK

from graphstorm.data import StandardGSgnnDataset
from graphstorm.data.constants import TOKEN_IDX, VALID_LEN_IDX

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("test")
    argparser.add_argument("--json_input", type=str, default="/data/ml-json")
    argparser.add_argument("--processed_input", type=str, default="/data/ml-json")
    argparser.add_argument('--undirected', action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument('--num_parts', type=int, default=4)
    args = argparser.parse_args()
    json_graph_path = args.json_input
    processed_graph = args.processed_input

    # process using single process
    dataset = StandardGSgnnDataset(json_graph_path,
                                   "yelp",
                                   hf_bert_model="bert-base-uncased",
                                   nid_fields={},
                                   src_field="src_id",
                                   dst_field="dst_id",
                                   nlabel_fields={"business":"stars"},
                                   ntask_types={"business":REGRESSION_TASK},
                                   split_ntypes=[],
                                   elabel_fields={},
                                   etask_types={},
                                   split_etypes=[],
                                   ntext_fields={"review":["text"]},
                                   max_node_seq_length={"review": 128},
                                   num_worker=8,
                                   feat_format=None)
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

    # load from processed_graph
    # load type map
    with open(os.path.join(processed_graph, "map_info.json"), 'r') as f:
        map_info = json.load(f)
        ntype_map = map_info["ntype_map"]
        etype_map = map_info["etype_map"]
    print(ntype_map)
    print(etype_map)
    rev_ntype_map = {int(id): ntype for ntype, id in ntype_map.items()}
    rev_etype_map = {int(id): etype for etype, id in etype_map.items()}

    # load nodes
    num_nodes = {ntype: 0 for ntype in ntype_map.keys()}
    gnid2lnid = {}
    with open(os.path.join(processed_graph, "yelp_nodes.txt"), 'r') as f:
        lines = f.readlines()
        for gnid, line in enumerate(lines):
            # skip the last .
            line = line.strip()[:-1].split(' ')
            ntype_id = int(line[0])
            gnid2lnid[int(gnid)] = num_nodes[rev_ntype_map[ntype_id]]
            num_nodes[rev_ntype_map[ntype_id]] += 1

    # load edges
    # we know there are two edge files
    edges = {etype:[] for etype in etype_map.keys()}
    for i in range(args.num_parts):
        with open(os.path.join(processed_graph, "yelp_edges_{}.txt".format(i)), 'r') as f:
            lines = f.readlines()
            for line in lines:
                # skip the last .
                line = line.strip()[:-1].split(' ')
                src_id = gnid2lnid[int(line[0])]
                dst_id = gnid2lnid[int(line[1])]
                edges[rev_etype_map[int(line[3])]].append((src_id, dst_id))

    dgl_edges = {}
    for etype, edge in edges.items():
        edge = np.array(edge)
        src = edge[:,0]
        dst = edge[:,1]
        etype = etype.split("_")
        etype = (etype[0], etype[1], etype[2])
        dgl_edges[etype] = (src, dst)

    # load features
    def load_feat(ntype, feat_name, num_part):
        feats = []
        for i in range(num_part):
            print("load {}".format(os.path.join(processed_graph, "{}/{}/{}.npy".format(ntype, feat_name, i))))
            feats.append(np.load(os.path.join(processed_graph, "{}/{}/{}.npy".format(ntype, feat_name, i))))
        for f in feats:
            print(f.shape)
        return np.concatenate(feats, axis=0)
    # load features
    stars = load_feat("business", "stars", args.num_parts)
    input_ids = load_feat("review", "input_ids", args.num_parts)
    valid_len = load_feat("review", "valid_len", args.num_parts)
    test_mask = load_feat("business", "test_mask", args.num_parts)
    train_mask = load_feat("business", "train_mask", args.num_parts)
    val_mask = load_feat("business", "val_mask", args.num_parts)

    # build dglgraph
    g_dp = dgl.heterograph(dgl_edges, num_nodes_dict=num_nodes)
    print(g_dp)
    g_dp.nodes["business"].data["stars"] = th.tensor(stars)
    g_dp.nodes["review"].data["input_ids"] = th.tensor(input_ids)
    g_dp.nodes["review"].data["valid_len"] = th.tensor(valid_len)
    g_dp.nodes["business"].data["test_mask"] = th.tensor(test_mask)
    g_dp.nodes["business"].data["train_mask"] = th.tensor(train_mask)
    g_dp.nodes["business"].data["val_mask"] = th.tensor(val_mask)

    # compare if two graph is identical
    print(g)
    assert g.num_nodes("review") == g_dp.num_nodes("review")
    assert g.num_nodes("business") == g_dp.num_nodes("business")
    assert g.num_nodes("category") == g_dp.num_nodes("category")
    assert g.num_nodes("city") == g_dp.num_nodes("city")
    assert g.num_nodes("user") == g_dp.num_nodes("user")
    assert g.num_edges(("business", "incategory", "category")) == g_dp.num_edges(("business", "incategory", "category"))
    assert g.num_edges(("business", "in", "city")) == g_dp.num_edges(("business", "in", "city"))
    assert g.num_edges(("review", "on", "business")) == g_dp.num_edges(("review", "on", "business"))
    assert g.num_edges(("user", "friendship", "user")) == g_dp.num_edges(("user", "friendship", "user"))
    assert g.num_edges(("user", "write", "review")) == g_dp.num_edges(("user", "write", "review"))
    assert np.array_equal(g.nodes["business"].data["stars"], g_dp.nodes["business"].data["stars"])
    assert np.array_equal(g.nodes["review"].data["input_ids"], g_dp.nodes["review"].data["input_ids"])
    assert np.array_equal(g.nodes["review"].data["valid_len"], g_dp.nodes["review"].data["valid_len"])

    edges_g = g.edges(etype="incategory")
    edges_g_dp = g_dp.edges(etype="incategory")

    src_g, dst_g = edges_g
    src_g_dp, dst_g_dp = edges_g_dp
    sort_idx_g = th.argsort(src_g)
    sort_idx_g_dp = th.argsort(src_g_dp)
    assert np.array_equal(src_g[sort_idx_g].numpy(), src_g_dp[sort_idx_g_dp].numpy())
    assert np.array_equal(dst_g[sort_idx_g].numpy(), dst_g_dp[sort_idx_g_dp].numpy())

    edges_g = g.edges(etype="write")
    edges_g_dp = g_dp.edges(etype="write")

    src_g, dst_g = edges_g
    src_g_dp, dst_g_dp = edges_g_dp
    sort_idx_g = th.argsort(src_g)
    sort_idx_g_dp = th.argsort(src_g_dp)
    assert np.array_equal(src_g[sort_idx_g].numpy(), src_g_dp[sort_idx_g_dp].numpy())
    assert np.array_equal(dst_g[sort_idx_g].numpy(), dst_g_dp[sort_idx_g_dp].numpy())
