""" utils for generate data
"""
import os
import json

import numpy as np

def write_edges(edges, path):
    for etype, edge in edges.items():
        edge_file_dir = os.path.join(path, f"edges-{'_'.join(etype)}")
        os.mkdir(edge_file_dir)
        edge_file_path = os.path.join(edge_file_dir, "0.json")
        with open(edge_file_path, 'w') as f:
            for src, dst in zip(edge[0], edge[1]):
                edge_info = {"src_id": f"{etype[0]}_{src}", "dst_id": f"{etype[2]}_{dst}", "type": etype[1]}
                line = json.dumps(edge_info)
                f.write(line + '\n')

def write_nodes(ntype, num_nodes, path):
    with open(os.path.join(path, f"nid-{ntype}.txt"), "w") as f:
        for i in range(num_nodes):
            line = json.dumps(f"{ntype}_{i}")
            f.write(line + "\n")

    node_file_dir = os.path.join(path, f"nodes-{ntype}")
    os.mkdir(node_file_dir)
    with open(os.path.join(node_file_dir, "0.json"), "w") as f:
        for i in range(num_nodes):
            info = {
                "id": f"{ntype}_{i}",
            }

            line = json.dumps(info)
            f.write(line + '\n')

def write_nfeat(feat, ntype, feat_name, path):
    nfeat_dir = os.path.join(path, f"nfeats-{ntype}")
    os.mkdir(nfeat_dir)
    nfeat_dir = os.path.join(nfeat_dir, feat_name)
    os.mkdir(nfeat_dir)
    nfeat_path = os.path.join(nfeat_dir, "part-001.npy")
    np.save(nfeat_path, feat)

def write_edges_with_labels(edges, labels, path):
    for etype, edge in edges.items():
        edge_file_dir = os.path.join(path, f"edges-{'_'.join(etype)}")
        os.mkdir(edge_file_dir)
        edge_file_path = os.path.join(edge_file_dir, "0.json")
        with open(edge_file_path, 'w') as f:
            for i, (src, dst) in enumerate(zip(edge[0], edge[1])):
                edge_info = {"src_id": f"{etype[0]}_{src}", "dst_id": f"{etype[2]}_{dst}", "type": etype[1]}

                if etype[1] == "r1":
                    edge_info["label"] = labels[i]
                line = json.dumps(edge_info)
                f.write(line + '\n')

def write_nodes_with_labels(ntype, num_nodes, labels, path):
    with open(os.path.join(path, f"nid-{ntype}.txt"), "w") as f:
        for i in range(num_nodes):
            line = json.dumps(f"{ntype}_{i}")
            f.write(line + "\n")

    node_file_dir = os.path.join(path, f"nodes-{ntype}")
    os.mkdir(node_file_dir)
    with open(os.path.join(node_file_dir, "0.json"), "w") as f:
        for i in range(num_nodes):
            info = {
                "id": f"{ntype}_{i}",
            }
            info["label"] = labels[i]
            line = json.dumps(info)
            f.write(line + '\n')
