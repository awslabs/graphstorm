""" Generate a test graph for multi-label edge classification task
"""

import os
import argparse
import shutil

from gen_multilabel_nc import gen_multi_label, gen_graph
from utils import write_nodes, write_nfeat
from utils import write_edges_with_labels

def generate_graph_data(path):
    graph = gen_graph()
    edges, node_feat_ntype0, node_feat_ntype1 = graph
    labels = gen_multi_label(6, 200)

    write_edges_with_labels(edges, labels, path)
    write_nfeat(node_feat_ntype0, "ntype0", "feat", path)
    write_nfeat(node_feat_ntype1, "ntype1", "feat", path)
    write_nodes("ntype0", 100, path)
    write_nodes("ntype1", 100, path)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Generate graph")
    argparser.add_argument("--path", type=str, required=True,
                           help="Path to save the generated graph")
    args = argparser.parse_args()
    if os.path.exists(args.path):
        shutil.rmtree(args.path)
    os.mkdir(args.path)
    generate_graph_data(args.path)