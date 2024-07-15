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

import dgl
import numpy as np

from numpy.testing import assert_almost_equal

argparser = argparse.ArgumentParser("Compare graphs")
argparser.add_argument("--graph-path1", type=str, required=True,
                       help="The path of the constructed graph.")
argparser.add_argument("--graph-path2", type=str, required=True,
                       help="The path of the constructed graph.")
args = argparser.parse_args()

g1 = dgl.load_graphs(args.graph_path1)[0][0]
g2 = dgl.load_graphs(args.graph_path2)[0][0]
assert g1.ntypes == g2.ntypes
assert g1.etypes == g2.etypes
for ntype in g1.ntypes:
    assert g1.number_of_nodes(ntype) == g2.number_of_nodes(ntype)
    for name in g1.nodes[ntype].data:
        # We should skip '*_mask' because data split is split randomly.
        if 'mask' not in name:
            assert_almost_equal(g1.nodes[ntype].data[name].numpy(),
                                g2.nodes[ntype].data[name].numpy())


for etype in g1.canonical_etypes:
    assert g1.number_of_edges(etype) == g2.number_of_edges(etype)
    for name in g1.edges[etype].data:
        # We should skip '*_mask' because data split is split randomly.
        if 'mask' not in name:
            assert_almost_equal(g1.edges[etype].data[name].numpy(),
                                g2.edges[etype].data[name].numpy())
