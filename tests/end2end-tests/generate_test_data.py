import numpy as np
import argparse
import os
import h5py
import json

def generate_nodes(path):
    node_dir = os.path.join(path, "nodes-item")
    os.mkdir(node_dir)
    with open(os.path.join(node_dir, "0.json"), "w") as f:
        line = json.dumps({"id":"i0"})
        f.write(line + '\n')
        line = json.dumps({"id":"i1"})
        f.write(line + '\n')
        line = json.dumps({"id":"i2"})
        f.write(line + '\n')
        line = json.dumps({"id":"i3"})
        f.write(line + '\n')

    node_dir = os.path.join(path, "nodes-node")
    os.mkdir(node_dir)
    with open(os.path.join(node_dir, "0.json"), "w") as f:
        line = json.dumps({"id":"n0", "text": "zero item", "label": 0, "train": 0, "valid": 1, "test": 0})
        f.write(line + '\n')
        line = json.dumps({"id":"n1", "text": "one item", "label": 1, "train": 0, "valid": 1, "test": 0})
        f.write(line + '\n')
        line = json.dumps({"id":"n2", "text": "two item", "label": 2, "train": 0, "valid": 1, "test": 0})
        f.write(line + '\n')
        line = json.dumps({"id":"n3", "text": "three item", "label": 0, "train": 0, "valid": 1, "test": 0})
        f.write(line + '\n')

    with open(os.path.join(path, "nid-item.txt"), "w") as f:
        f.write("\"i0\"\n")
        f.write("\"i1\"\n")
        f.write("\"i2\"\n")
        f.write("\"i3\"\n")

    with open(os.path.join(path, "nid-node.txt"), "w") as f:
        f.write("\"n0\"\n")
        f.write("\"n1\"\n")
        f.write("\"n2\"\n")
        f.write("\"n3\"\n")

def generate_edges(path):
    edge_dir = os.path.join(path, "edges-node_r1_node")
    os.mkdir(edge_dir)
    with open(os.path.join(edge_dir, "0.json"), "w") as f:
        line = json.dumps({"src_id": "n0", "dst_id": "n1", "type": "r1"})
        f.write(line + '\n')
        line = json.dumps({"src_id": "n1", "dst_id": "n2", "type": "r1"})
        f.write(line + '\n')
        line = json.dumps({"src_id": "n1", "dst_id": "n3", "type": "r1"})
        f.write(line + '\n')
        line = json.dumps({"src_id": "n3", "dst_id": "n0", "type": "r1"})
        f.write(line + '\n')

    edge_dir = os.path.join(path, "edges-node_r0_item")
    os.mkdir(edge_dir)
    with open(os.path.join(edge_dir, "0.json"), "w") as f:
        line = json.dumps({"src_id": "n1", "dst_id": "i0", "type": "r0", "label": 0, "train": 0, "valid": 1, "test": 0})
        f.write(line + '\n')
        line = json.dumps({"src_id": "n1", "dst_id": "i3", "type": "r0", "label": 2, "train": 0, "valid": 1, "test": 0})
        f.write(line + '\n')
        line = json.dumps({"src_id": "n2", "dst_id": "i0", "type": "r0", "label": 0, "train": 0, "valid": 0, "test": 1})
        f.write(line + '\n')
        line = json.dumps({"src_id": "n3", "dst_id": "i1", "type": "r0", "label": 5, "train": 0, "valid": 1, "test": 0})
        f.write(line + '\n')
        line = json.dumps({"src_id": "n3", "dst_id": "i2", "type": "r0", "label": 0, "train": 0, "valid": 1, "test": 0})
        f.write(line + '\n')
        line = json.dumps({"src_id": "n3", "dst_id": "i3", "type": "r0", "label": 10, "train": 1, "valid": 0, "test": 0})
        f.write(line + '\n')

def generate_node_feats(path):
    vectors = np.random.rand(4,2).astype(np.float32)

    if not os.path.exists(path+'/nfeats-item/embedding'):
        os.makedirs(path+'/nfeats-item/embedding')
    if not os.path.exists(path+'/nfeats-node/embedding'):
        os.makedirs(path+'/nfeats-node/embedding')

    with h5py.File(path+'/nfeats-item/embedding/feat.hdf5', 'w') as f:
        f.create_dataset("embeddings", data=vectors)
    with h5py.File(path+'/nfeats-node/embedding/feat.hdf5', 'w') as f:
        f.create_dataset("embeddings", data=vectors)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument("--path", type=str, required=True,
                           help="dataset to use")
    args = argparser.parse_args()
    if os.path.exists(args.path):
        print(f'Generate test data in {args.path}')

    generate_nodes(args.path)
    generate_edges(args.path)
    generate_node_feats(args.path)
