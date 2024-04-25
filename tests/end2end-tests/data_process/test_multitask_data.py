"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Test constructed graph data with customized mask
"""
import os
import json
import dgl
import pyarrow.parquet as pq
import numpy as np
import torch as th
import argparse

def read_data_parquet(data_file):
    table = pq.read_table(data_file)
    pd = table.to_pandas()
    return {key: np.array(pd[key]) for key in pd}

argparser = argparse.ArgumentParser("Preprocess graphs")
argparser.add_argument("--graph-format", type=str, required=True,
                       help="The constructed graph format.")
argparser.add_argument("--graph_dir", type=str, required=True,
                       help="The path of the constructed graph.")
argparser.add_argument("--conf_file", type=str, required=True,
                       help="The configuration file.")
argparser.add_argument("--with-reverse-edge",
                       type=lambda x: (str(x).lower() in ['true', '1']),
                       default=False,
                       help="Whether check reverse edges")
args = argparser.parse_args()
out_dir = args.graph_dir
with open(args.conf_file, 'r') as f:
    conf = json.load(f)

args = argparser.parse_args()
out_dir = args.graph_dir
with open(args.conf_file, 'r') as f:
    conf = json.load(f)

if args.graph_format == "DGL":
    g = dgl.load_graphs(os.path.join(out_dir, "test.dgl"))[0][0]
elif args.graph_format == "DistDGL":
    from dgl.distributed.graph_partition_book import _etype_str_to_tuple
    g, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list = \
            dgl.distributed.load_partition(os.path.join(out_dir, 'test.json'), 0)
    g = dgl.to_heterogeneous(g, ntypes_list, [etype[1] for etype in etypes_list])
    for key, val in node_feats.items():
        ntype, name = key.split('/')
        g.nodes[ntype].data[name] = val
    for key, val in edge_feats.items():
        etype, name = key.split('/')
        etype = _etype_str_to_tuple(etype)
        g.edges[etype].data[name] = val
else:
    raise ValueError('Invalid graph format: {}'.format(args.graph_format))

