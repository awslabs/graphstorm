""" This removes train/val/test masks on a dataset.
"""

import dgl
import os
import argparse

def remove_mask(data):
    new_data = {}
    for name in data:
        if 'mask' not in name:
            new_data[name] = data[name]
    return new_data

def print_feat_names(dataset):
    for d in os.listdir(dataset):
        part_dir = os.path.join(dataset, d)
        if not os.path.isfile(part_dir):
            data = dgl.data.load_tensors(os.path.join(part_dir, 'node_feat.dgl'))
            print('node feat of {}'.format(d), data.keys())
            data = dgl.data.load_tensors(os.path.join(part_dir, 'edge_feat.dgl'))
            print('edge feat of {}'.format(d), data.keys())

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Remove train/val/test masks")
    argparser.add_argument("--dataset", type=str, required=True,
                           help="The path to the partitioned graph.")
    argparser.add_argument("--remove_node_mask",
                           type=lambda x: (str(x).lower() in ['true', '1']), default=False,
                           help="Indicate to remove node masks or edge masks.")
    args = argparser.parse_args()

    print('before removing {} masks'.format('node' if args.remove_node_mask else 'edge'))
    print_feat_names(args.dataset)

    for d in os.listdir(args.dataset):
        part_dir = os.path.join(args.dataset, d)
        if not os.path.isfile(part_dir):
            if args.remove_node_mask:
                data = dgl.data.load_tensors(os.path.join(part_dir, 'node_feat.dgl'))
                data = remove_mask(data)
                dgl.data.save_tensors(os.path.join(part_dir, 'node_feat.dgl'), data)
            else:
                data = dgl.data.load_tensors(os.path.join(part_dir, 'edge_feat.dgl'))
                data = remove_mask(data)
                dgl.data.save_tensors(os.path.join(part_dir, 'edge_feat.dgl'), data)

    print('after removing {} masks'.format('node' if args.remove_node_mask else 'edge'))
    print_feat_names(args.dataset)
