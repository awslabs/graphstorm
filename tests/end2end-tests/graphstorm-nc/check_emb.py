import os, argparse
import torch as th
import json

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Check emb")
    argparser.add_argument("--emb-path", type=str, required=True,
        help="Path to node embedings to load")
    argparser.add_argument("--graph-config", type=str, required=True,
        help="Config file of the graph")
    argparser.add_argument("--ntypes", type=str, required=True,
        help="Node types to check. The format is 'ntype1 ntype2")
    argparser.add_argument("--emb-size", type=int, required=True,
        help="Output emb dim")
    args = argparser.parse_args()
    with open(args.graph_config, 'r') as f:
        graph_config = json.load(f)

        # Get all ntypes
        ntypes = args.ntypes.strip().split(' ')
        node_map = {}
        for ntype, range in graph_config['node_map'].items():
            node_map[ntype] = 0
            for r in range:
                node_map[ntype] += r[1] - r[0]

    # multiple node types
    files = os.listdir(args.emb_path)
    if len(node_map) > 1:
        for ntype, num_nodes in node_map.items():
            ntype_files = []
            for file in files:
                if file.startswith(ntype):
                    ntype_files.append(file)
            print(ntype_files)
            ntype_files.sort()

            feats = [th.load(os.path.join(args.emb_path, nfile)) for nfile in ntype_files]
            feats = th.cat(feats, dim=0)
            assert feats.shape[0] == num_nodes
            assert feats.shape[1] == args.emb_size
    else:
        files.sort()
        feats = [th.load(nfile) for nfile in files]
        feats = th.cat(feats, dim=0)
        assert feats.shape[0] == node_map.values()[0]
        assert feats.shape[1] == args.emb_size
