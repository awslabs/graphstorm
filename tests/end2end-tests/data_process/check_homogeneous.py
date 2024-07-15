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
import os
import argparse
import dgl
from dgl.distributed.constants import DEFAULT_NTYPE, DEFAULT_ETYPE
from numpy.testing import assert_almost_equal


def check_reverse_edge(args):

    g_orig = dgl.load_graphs(os.path.join(args.orig_graph_path, "graph.dgl"))[0][0]
    g_rev = dgl.load_graphs(os.path.join(args.rev_graph_path, "graph.dgl"))[0][0]
    assert g_orig.ntypes == g_rev.ntypes
    assert g_orig.etypes == g_rev.etypes
    assert g_orig.number_of_nodes(DEFAULT_NTYPE) == g_rev.number_of_nodes(DEFAULT_NTYPE)
    assert 2 * g_orig.number_of_edges(DEFAULT_ETYPE) == g_rev.number_of_edges(DEFAULT_ETYPE)
    for ntype in g_orig.ntypes:
        assert g_orig.number_of_nodes(ntype) == g_rev.number_of_nodes(ntype)
        for name in g_orig.nodes[ntype].data:
            # We should skip '*_mask' because data split is split randomly.
            if 'mask' not in name:
                assert_almost_equal(g_orig.nodes[ntype].data[name].numpy(),
                                    g_rev.nodes[ntype].data[name].numpy())

    # Check edge feature
    g_orig_feat = dgl.data.load_tensors(os.path.join(args.orig_graph_path, "edge_feat.dgl"))
    g_rev_feat = dgl.data.load_tensors(os.path.join(args.rev_graph_path, "edge_feat.dgl"))
    for feat_type in g_orig_feat.keys():
        if "mask" not in feat_type:
            assert_almost_equal(g_orig_feat[feat_type].numpy(),
                                g_rev_feat[feat_type].numpy()[:g_orig.number_of_edges(DEFAULT_ETYPE)])
        else:
            assert_almost_equal(g_rev_feat[feat_type].numpy()[g_orig.number_of_edges(DEFAULT_ETYPE):],
                                [0] * g_orig.number_of_edges(DEFAULT_ETYPE))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Check edge prediction remapping")
    argparser.add_argument("--orig-graph-path", type=str, default="/tmp/movielen_100k_train_val_1p_4t_homogeneous/part0/",
                           help="Path to save the generated data")
    argparser.add_argument("--rev-graph-path", type=str, default="/tmp/movielen_100k_train_val_1p_4t_homogeneous_rev/part0/",
                           help="Path to save the generated data")

    args = argparser.parse_args()

    check_reverse_edge(args)