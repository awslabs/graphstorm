"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

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
    for ntype, num_nodes in node_map.items():
        ntype_files = os.listdir(os.path.join(args.emb_path, ntype))

        # Only work with torch 1.13+
        feats = [th.load(os.path.join(os.path.join(args.emb_path, ntype), nfile),
                         weights_only=True) for nfile in ntype_files]
        feats = th.cat(feats, dim=0)
        assert feats.shape[0] == num_nodes
        assert feats.shape[1] == args.emb_size
