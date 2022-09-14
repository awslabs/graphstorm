""" Generate a test graph for multi-label classification task
"""

import os
import json
import argparse
import shutil

import numpy as np

from utils import write_edges, write_nodes, write_nfeat
from utils import write_nodes_with_labels

def gen_multi_label(num_class, size):
    labels0 = np.random.randint(num_class, size=size)
    labels1 = np.random.randint(num_class, size=size)
    labels = np.zeros((size, num_class))
    labels[np.arange(size), labels0] =1
    labels[np.arange(size), labels1] =1

    return labels.tolist()

def gen_graph():
    # edges
    edges = {
        ("ntype0", "r0", "ntype1"): (np.random.randint(100, size=100).tolist(), np.random.randint(100, size=100).tolist()),
        ("ntype0", "r1", "ntype1"): (np.random.randint(100, size=200).tolist(), np.random.randint(100, size=200).tolist()),
    }

    node_feat_ntype0 = np.random.rand(100,20).astype('f')
    node_feat_ntype1 = np.random.rand(100,40).astype('f')

    return (edges, node_feat_ntype0, node_feat_ntype1)

def generate_graph_data(path):
    graph = gen_graph()
    edges, node_feat_ntype0, node_feat_ntype1 = graph
    labels = gen_multi_label(6, 100)

    write_edges(edges, path)
    write_nfeat(node_feat_ntype0, "ntype0", "feat", path)
    write_nfeat(node_feat_ntype1, "ntype1", "feat", path)
    write_nodes("ntype0", 100, path)
    write_nodes_with_labels("ntype1", 100, labels, path)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Generate graph")
    argparser.add_argument("--path", type=str, required=True,
                           help="Path to save the generated graph")
    args = argparser.parse_args()
    if os.path.exists(args.path):
        shutil.rmtree(args.path)
    os.mkdir(args.path)
    generate_graph_data(args.path)