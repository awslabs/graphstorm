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

def load_processed_graph(args, meta_info):
    # load from processed_graph
    # load type map
    processed_graph = "ml-output"

    with open(os.path.join(processed_graph, "metadata.json"), 'r') as f:
        map_info = json.load(f)
        node_types = map_info["node_type"]
        edge_types = map_info["edge_type"]
    rev_ntype_map = {int(id): ntype for id, ntype in enumerate(node_types)}
    rev_etype_map = {int(id): etype for id, etype in enumerate(edge_types)}

    num_nodes_per_chunk = map_info["num_nodes_per_chunk"]
    # load nodes
    num_nodes = {ntype: 0 for ntype in node_types}
    offset = 0
    ntype_offset = 0
    # iterate node type 0 (ml_nodes0)
    for i in range(args.num_parts):
        with open(os.path.join(processed_graph, f"ml_nodes0_{i}.txt"), 'r') as f:
            lines = f.readlines()
            start_offset = ntype_offset
            for line in lines:
                # skip the last .
                line = line.strip().split(' ')
                ntype_id = int(line[0])
                num_nodes[rev_ntype_map[ntype_id]] += 1
                offset += 1
                ntype_offset += 1
                assert ntype_id == 0
            assert meta_info["num_nodes_per_chunk"][0][i] == ntype_offset - start_offset

    ntype_offset = 0
    # iterate node type 1 (ml_nodes1)
    for i in range(args.num_parts):
        with open(os.path.join(processed_graph, f"ml_nodes1_{i}.txt"), 'r') as f:
            lines = f.readlines()
            start_offset = ntype_offset
            for line in lines:
                # skip the last .
                line = line.strip().split(' ')
                ntype_id = int(line[0])
                num_nodes[rev_ntype_map[ntype_id]] += 1
                offset += 1
                ntype_offset += 1
                assert ntype_id == 1
            assert meta_info["num_nodes_per_chunk"][1][i] == ntype_offset - start_offset

    ntype_offset = 0
    # iterate node type 2 (ml_nodes2)
    for i in range(args.num_parts):
        with open(os.path.join(processed_graph, f"ml_nodes2_{i}.txt"), 'r') as f:
            lines = f.readlines()
            start_offset = ntype_offset
            for line in lines:
                # skip the last .
                line = line.strip().split(' ')
                ntype_id = int(line[0])
                num_nodes[rev_ntype_map[ntype_id]] += 1
                offset += 1
                ntype_offset += 1
                assert ntype_id == 2
            assert meta_info["num_nodes_per_chunk"][2][i] == ntype_offset - start_offset

    # load edges
    # we know there are two edge files
    etype_offset = 0
    # iterate edge type 0 (ml_edges0)
    edges = {etype:[] for etype in edge_types}
    for i in range(args.num_parts):
        with open(os.path.join(processed_graph, f"ml_edges0_{i}.txt"), 'r') as f:
            lines = f.readlines()
            start_offset = etype_offset
            for line in lines:
                # skip the last .
                line = line.strip().split(' ')
                src_id = int(line[0])
                dst_id = int(line[1])
                edges[rev_etype_map[0]].append((src_id, dst_id))
                etype_offset += 1
            assert meta_info["num_edges_per_chunk"][0][i] == etype_offset - start_offset

    etype_offset = 0
    # iterate edge type 1 (ml_edges1)
    for i in range(args.num_parts):
        with open(os.path.join(processed_graph, f"ml_edges1_{i}.txt"), 'r') as f:
            lines = f.readlines()
            start_offset = etype_offset
            for line in lines:
                # skip the last .
                line = line.strip().split(' ')
                src_id = int(line[0])
                dst_id = int(line[1])
                edges[rev_etype_map[1]].append((src_id, dst_id))
                etype_offset += 1
            assert meta_info["num_edges_per_chunk"][1][i] == etype_offset - start_offset

    if args.undirected is True:
        etype_offset = 0
        # iterate edge type 2 (ml_edges2)
        for i in range(args.num_parts):
            with open(os.path.join(processed_graph, f"ml_edges2_{i}.txt"), 'r') as f:
                lines = f.readlines()
                start_offset = etype_offset
                for line in lines:
                    # skip the last .
                    line = line.strip().split(' ')
                    src_id = int(line[0])
                    dst_id = int(line[1])
                    edges[rev_etype_map[2]].append((src_id, dst_id))
                    etype_offset += 1
                assert meta_info["num_edges_per_chunk"][2][i] == etype_offset - start_offset

        etype_offset = 0
        # iterate edge type 3 (ml_edges3)
        for i in range(args.num_parts):
            with open(os.path.join(processed_graph, f"ml_edges3_{i}.txt"), 'r') as f:
                lines = f.readlines()
                start_offset = etype_offset
                for line in lines:
                    # skip the last .
                    line = line.strip().split(' ')
                    src_id = int(line[0])
                    dst_id = int(line[1])
                    edges[rev_etype_map[3]].append((src_id, dst_id))
                    etype_offset += 1
                assert meta_info["num_edges_per_chunk"][3][i] == etype_offset - start_offset

    dgl_edges = {}
    for etype, edge in edges.items():
        edge = np.array(edge)
        src = edge[:,0]
        dst = edge[:,1]
        etype = etype.split(":")
        etype = (etype[0], etype[1], etype[2])
        dgl_edges[etype] = (src, dst)
    for ntype, cnt in zip(node_types, num_nodes_per_chunk):
        assert num_nodes[ntype] == sum(cnt)

    g_dp = dgl.heterograph(dgl_edges, num_nodes_dict=num_nodes)
    return g_dp

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
                                   nlabel_fields={"movie":"genre"},
                                   ntask_types={"movie":CLASSIFICATION_TASK},
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
    genre = load_feat("movie", "genre", args.num_parts)
    input_ids = load_feat("movie", "input_ids", args.num_parts)
    valid_len = load_feat("movie", "valid_len", args.num_parts)
    test_mask = load_feat("movie", "test_mask", args.num_parts)
    train_mask = load_feat("movie", "train_mask", args.num_parts)
    val_mask = load_feat("movie", "val_mask", args.num_parts)

    # build dglgraph
    g_dp = load_processed_graph(args, meta_info)
    g_dp.nodes["movie"].data["genre"] = th.tensor(genre)
    g_dp.nodes["movie"].data["input_ids"] = th.tensor(input_ids)
    g_dp.nodes["movie"].data["valid_len"] = th.tensor(valid_len)
    g_dp.nodes["movie"].data["test_mask"] = th.tensor(test_mask)
    g_dp.nodes["movie"].data["train_mask"] = th.tensor(train_mask)
    g_dp.nodes["movie"].data["val_mask"] = th.tensor(val_mask)

    # compare if two graph is identical
    print(g)
    print(g_dp)
    assert g.num_nodes("user") == g_dp.num_nodes("user")
    assert g.num_nodes("movie") == g_dp.num_nodes("movie")
    assert g.num_nodes("occupation") == g_dp.num_nodes("occupation")
    assert g.num_edges(("user", "has-occupation", "occupation")) == g_dp.num_edges(("user", "has-occupation", "occupation"))
    assert g.num_edges(("user", "rating", "movie")) == g_dp.num_edges(("user", "rating", "movie"))
    assert np.array_equal(g.nodes["movie"].data["genre"], g_dp.nodes["movie"].data["genre"])
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
