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

    Generate a test graph with different feature names in different node types
"""

import os
import argparse
import dgl

import numpy as np

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Generate graph")
    argparser.add_argument("--path", type=str, required=True,
                           help="Path to save the generated graph")
    argparser.add_argument("--multi_feats",
                           type=lambda x: (str(x).lower() in ['true', '1']),
                           default=False,
                           help="If true, generate multiple node feature for each node type")
    args = argparser.parse_args()
    for d in os.listdir(args.path):
        part_dir = os.path.join(args.path, d)
        if not os.path.isfile(part_dir):
            data = dgl.data.load_tensors(os.path.join(part_dir, 'node_feat.dgl'))
            new_data = {}
            names = list(data.keys())
            # To ensure that renaming for different node types is deterministic,
            # we should reorder the list of data name.
            names.sort()
            i = 0
            for name in names:
                if 'feat' in name:
                    # Here we want to make sure that features on different node types
                    # have exactly different names.
                    new_data[name + str(i)] = data[name]
                    i += 1
                    if args.multi_feats:
                        new_data[name + str(i)] = data[name]
                        i += 1
                else:
                    new_data[name] = data[name]
            dgl.data.save_tensors(os.path.join(part_dir, 'node_feat.dgl'), new_data)
            print('node features:', new_data.keys())
