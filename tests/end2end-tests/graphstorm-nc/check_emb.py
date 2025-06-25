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
import argparse
import json
import logging
import os

import numpy as np
import pyarrow.parquet as pq
import torch as th

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Check embedding file correctness")
    argparser.add_argument("--emb-path", type=str, required=True,
        help="Path to node embedings to load")
    argparser.add_argument("--graph-config", type=str, required=True,
        help="Config file of the graph")
    argparser.add_argument("--ntypes", type=str, nargs='+', required=True,
        help="Node types to check. The format is 'ntype1 ntype2")
    argparser.add_argument("--emb-size", type=int, required=True,
        help="Output emb dim")
    argparser.add_argument("--expected-row-count", type=int, required=False,
        default=None, help="Expected number of rows in the embedding file")
    argparser.add_argument("--file-format", type=str, default="pytorch",
        choices=["parquet", "torch"],
        help=("File format of the embedding files, can be 'parquet' or 'pytorch', "
              "default: pytorch"))
    args = argparser.parse_args()

    with open(args.graph_config, 'r', encoding="utf-8") as f:
        graph_config = json.load(f)

        # Get all ntypes
        ntypes = args.ntypes
        node_map = {}
        for ntype, nrange in graph_config['node_map'].items():
            node_map[ntype] = 0
            for r in nrange:
                node_map[ntype] += r[1] - r[0]

    # multiple node types
    for ntype, num_nodes in node_map.items():
        if ntype not in ntypes:
            continue
        ntype_files = os.listdir(os.path.join(args.emb_path, ntype))

        if args.file_format == "parquet":
            feats = pq.read_table(
                [
                    os.path.join(
                        os.path.join(args.emb_path, ntype), nfile)
                    for nfile in ntype_files if nfile.startswith("embed")
                ]
            )
            feats = th.tensor(np.array(feats["emb"].to_pylist())).squeeze()
        else:
            # Only work with torch 1.13+
            feats = [th.load(os.path.join(os.path.join(args.emb_path, ntype), nfile),
                            weights_only=True) for nfile in ntype_files]
            feats = th.cat(feats, dim=0)

        # If the caller asks for a specific row count check that,
        # otherwise assume all nodes should have embeddings
        if args.expected_row_count is not None:
            assert feats.shape[0] == args.expected_row_count, \
                f"Expected {args.expected_row_count} rows, got {feats.shape[0]} rows"
        else:
            assert feats.shape[0] == num_nodes, \
                f"Expected {num_nodes} rows, got {feats.shape[0]} rows"

        # Ensure embeddings have correct number of dims
        assert feats.shape[1] == args.emb_size, \
            f"Expected {args.emb_size} columns, got {feats.shape[1]} columns"

        logging.info("Checks for '%s' embeddings passed", ntype)
