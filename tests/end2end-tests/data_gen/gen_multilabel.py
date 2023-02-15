""" This augment the dataset with fake multi-labels on each node.
"""

import os
import argparse
import dgl
import numpy as np
import torch as th

def gen_multi_label(num_class, size):
    labels0 = np.random.randint(num_class, size=size)
    labels1 = np.random.randint(num_class, size=size)
    labels = np.zeros((size, num_class))
    labels[np.arange(size), labels0] =1
    labels[np.arange(size), labels1] =1

    return th.tensor(labels)

def generate_graph_data(data, field):
    new_data = {}
    for name in data:
        if field in name:
            new_data[name] = gen_multi_label(6, data[name].shape[0])
        else:
            new_data[name] = data[name]
    return new_data

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Generate graph")
    argparser.add_argument("--path", type=str, required=True,
                           help="Path to save the generated graph")
    argparser.add_argument("--node_class",
                           type=lambda x: (str(x).lower() in ['true', '1']), default=False,
                           help="Indicate whether to generate multi-label on nodes.")
    argparser.add_argument("--field", type=str, required=True,
                           help="The label field")
    args = argparser.parse_args()

    for d in os.listdir(args.path):
        part_dir = os.path.join(args.path, d)
        if not os.path.isfile(part_dir):
            if args.node_class:
                data = dgl.data.load_tensors(os.path.join(part_dir, 'node_feat.dgl'))
                data = generate_graph_data(data, args.field)
                dgl.data.save_tensors(os.path.join(part_dir, 'node_feat.dgl'), data)
            else:
                data = dgl.data.load_tensors(os.path.join(part_dir, 'edge_feat.dgl'))
                data = generate_graph_data(data, args.field)
                dgl.data.save_tensors(os.path.join(part_dir, 'edge_feat.dgl'), data)
