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

import pytest

import torch as th
from numpy.testing import assert_almost_equal

from graphstorm.model import (LinkPredictDotDecoder,
                              LinkPredictDistMultDecoder,
                              LinkPredictWeightedDotDecoder,
                              LinkPredictWeightedDistMultDecoder,
                              MLPEFeatEdgeDecoder)
from graphstorm.dataloading import (BUILTIN_LP_UNIFORM_NEG_SAMPLER,
                                    BUILTIN_LP_JOINT_NEG_SAMPLER,
                                    EP_DECODER_EDGE_FEAT)
from graphstorm.eval.utils import calc_distmult_pos_score
from graphstorm.eval.utils import calc_dot_pos_score
from graphstorm.model.edge_decoder import _get_edge_weight

from numpy.testing import assert_equal

from data_utils import generate_dummy_hetero_graph

def _check_scores(score, pos_score, neg_scores, etype, num_neg, batch_size):
    # pos scores
    assert score[etype][0].shape[0] == batch_size
    assert len(score[etype][0].shape) == 1
    # neg scores
    assert len(score[etype][1].shape) == 2
    assert score[etype][1].shape[0] == batch_size
    assert score[etype][1].shape[1] == num_neg
    assert_almost_equal(score[etype][0].numpy(), pos_score.numpy(), decimal=5)
    assert_almost_equal(score[etype][1].numpy(), neg_scores.numpy(), decimal=5)


def check_calc_test_scores_uniform_neg(decoder, etypes, h_dim, num_pos, num_neg, device):
    neg_sample_type = BUILTIN_LP_UNIFORM_NEG_SAMPLER
    emb = {
        'a': th.rand((128, h_dim)),
        'b': th.rand((128, h_dim)),
    }

    def gen_edge_pairs():
        pos_src = th.randint(100, (num_pos,))
        pos_dst = th.randint(100, (num_pos,))
        neg_src = th.randint(128, (num_pos, num_neg))
        neg_dst = th.randint(128, (num_pos, num_neg))
        return (pos_src, neg_src, pos_dst, neg_dst)

    with th.no_grad():
        pos_neg_tuple = {
            etypes[0]: gen_edge_pairs(),
            etypes[1]: gen_edge_pairs(),
        }
        pos_src, neg_src, pos_dst, neg_dst = pos_neg_tuple[etypes[0]]
        pos_neg_tuple[etypes[0]] = (pos_src, None, pos_dst, neg_dst)
        pos_src, neg_src, pos_dst, neg_dst = pos_neg_tuple[etypes[1]]
        pos_neg_tuple[etypes[1]] = (pos_src, neg_src, pos_dst, None)
        score = decoder.calc_test_scores(emb, pos_neg_tuple, neg_sample_type, device)
        pos_src, _, pos_dst, neg_dst = pos_neg_tuple[etypes[0]]
        pos_src_emb = emb[etypes[0][0]][pos_src]
        pos_dst_emb = emb[etypes[0][2]][pos_dst]
        rel_emb = decoder.get_relemb(etypes[0])
        pos_score = calc_distmult_pos_score(pos_src_emb, rel_emb, pos_dst_emb)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            neg_dst_emb = emb[etypes[0][2]][neg_dst[i]]
            # (dim) * (dim) * (num_neg, dim)
            ns = calc_distmult_pos_score(pse, rel_emb, neg_dst_emb)
            neg_scores.append(ns)
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etypes[0], num_neg, pos_src.shape[0])

        pos_src, neg_src, pos_dst, _ = pos_neg_tuple[etypes[1]]
        pos_src_emb = emb[etypes[1][0]][pos_src]
        pos_dst_emb = emb[etypes[1][2]][pos_dst]
        rel_emb = decoder.get_relemb(etypes[1])
        pos_score = calc_distmult_pos_score(pos_src_emb, rel_emb, pos_dst_emb)
        neg_scores = []
        for i in range(pos_dst.shape[0]):
            neg_src_emb = emb[etypes[1][0]][neg_src[i]]
            pde = pos_dst_emb[i]
            # (num_neg, dim) * (dim) * (dim)
            ns = calc_distmult_pos_score(neg_src_emb, rel_emb, pde)
            neg_scores.append(ns)
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etypes[1], num_neg, pos_src.shape[0])

        pos_neg_tuple = {
            etypes[0]: gen_edge_pairs(),
        }
        score = decoder.calc_test_scores(emb, pos_neg_tuple, neg_sample_type, device)
        pos_src, neg_src, pos_dst, neg_dst = pos_neg_tuple[etypes[0]]
        pos_src_emb = emb[etypes[0][0]][pos_src]
        pos_dst_emb = emb[etypes[0][2]][pos_dst]
        rel_emb = decoder.get_relemb(etypes[0])
        pos_score = calc_distmult_pos_score(pos_src_emb, rel_emb, pos_dst_emb)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            pde = pos_dst_emb[i]
            neg_src_emb = emb[etypes[0][0]][neg_src[i]]
            neg_dst_emb = emb[etypes[0][2]][neg_dst[i]]
            # (num_neg, dim) * (dim) * (dim)
            ns_0 = calc_distmult_pos_score(neg_src_emb, rel_emb, pde)
            # (dim) * (dim) * (num_neg, dim)
            ns_1 = calc_distmult_pos_score(pse, rel_emb, neg_dst_emb)
            neg_scores.append(th.cat((ns_0, ns_1), dim=-1))
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etypes[0], num_neg*2, pos_src.shape[0])

def check_calc_test_scores_joint_neg(decoder, etypes, h_dim, num_pos, num_neg, device):
    neg_sample_type = BUILTIN_LP_JOINT_NEG_SAMPLER
    emb = {
        'a': th.rand((128, h_dim)),
        'b': th.rand((128, h_dim)),
    }

    def gen_edge_pairs():
        pos_src = th.randint(100, (num_pos,))
        pos_dst = th.randint(100, (num_pos,))
        neg_src = th.randint(128, (num_neg,))
        neg_dst = th.randint(128, (num_neg,))
        return (pos_src, neg_src, pos_dst, neg_dst)

    with th.no_grad():
        pos_neg_tuple = {
            etypes[0]: gen_edge_pairs(),
            etypes[1]: gen_edge_pairs(),
        }
        pos_src, neg_src, pos_dst, neg_dst = pos_neg_tuple[etypes[0]]
        pos_neg_tuple[etypes[0]] = (pos_src, None, pos_dst, neg_dst)
        pos_src, neg_src, pos_dst, neg_dst = pos_neg_tuple[etypes[1]]
        pos_neg_tuple[etypes[1]] = (pos_src, neg_src, pos_dst, None)
        score = decoder.calc_test_scores(emb, pos_neg_tuple, neg_sample_type, device)
        pos_src, _, pos_dst, neg_dst = pos_neg_tuple[etypes[0]]
        pos_src_emb = emb[etypes[0][0]][pos_src]
        pos_dst_emb = emb[etypes[0][2]][pos_dst]
        rel_emb = decoder.get_relemb(etypes[0])
        pos_score = calc_distmult_pos_score(pos_src_emb, rel_emb, pos_dst_emb)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            neg_dst_emb = emb[etypes[0][2]][neg_dst]
            # (dim) * (dim) * (num_neg, dim)
            ns = calc_distmult_pos_score(pse, rel_emb, neg_dst_emb)
            neg_scores.append(ns)
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etypes[0], num_neg, pos_src.shape[0])

        pos_src, neg_src, pos_dst, _ = pos_neg_tuple[etypes[1]]
        pos_src_emb = emb[etypes[1][0]][pos_src]
        pos_dst_emb = emb[etypes[1][2]][pos_dst]
        rel_emb = decoder.get_relemb(etypes[1])
        pos_score = calc_distmult_pos_score(pos_src_emb, rel_emb, pos_dst_emb)
        neg_scores = []
        for i in range(pos_dst.shape[0]):
            neg_src_emb = emb[etypes[1][0]][neg_src]
            pde = pos_dst_emb[i]
            # (num_neg, dim) * (dim) * (dim)
            ns = calc_distmult_pos_score(neg_src_emb, rel_emb, pde)
            neg_scores.append(ns)
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etypes[1], num_neg, pos_src.shape[0])

        pos_neg_tuple = {
            etypes[0]: gen_edge_pairs(),
        }
        score = decoder.calc_test_scores(emb, pos_neg_tuple, neg_sample_type, device)
        pos_src, neg_src, pos_dst, neg_dst = pos_neg_tuple[etypes[0]]
        pos_src_emb = emb[etypes[0][0]][pos_src]
        pos_dst_emb = emb[etypes[0][2]][pos_dst]
        rel_emb = decoder.get_relemb(etypes[0])
        pos_score = calc_distmult_pos_score(pos_src_emb, rel_emb, pos_dst_emb)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            pde = pos_dst_emb[i]
            neg_src_emb = emb[etypes[0][0]][neg_src]
            neg_dst_emb = emb[etypes[0][2]][neg_dst]
            # (num_neg, dim) * (dim) * (dim)
            ns_0 = calc_distmult_pos_score(neg_src_emb, rel_emb, pde)
            # (dim) * (dim) * (num_neg, dim)
            ns_1 = calc_distmult_pos_score(pse, rel_emb, neg_dst_emb)
            neg_scores.append(th.cat((ns_0, ns_1), dim=-1))
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etypes[0], num_neg*2, pos_src.shape[0])

def check_calc_test_scores_dot_uniform_neg(decoder, etype, h_dim, num_pos, num_neg, device):
    neg_sample_type = BUILTIN_LP_UNIFORM_NEG_SAMPLER
    emb = {
        'a': th.rand((128, h_dim)),
        'b': th.rand((128, h_dim)),
    }

    def gen_edge_pairs():
        pos_src = th.randint(100, (num_pos,))
        pos_dst = th.randint(100, (num_pos,))
        neg_src = th.randint(128, (num_pos, num_neg))
        neg_dst = th.randint(128, (num_pos, num_neg))
        return (pos_src, neg_src, pos_dst, neg_dst)

    with th.no_grad():
        pos_src, _, pos_dst, neg_dst = gen_edge_pairs()
        pos_neg_tuple = {etype: (pos_src, None, pos_dst, neg_dst)}
        score = decoder.calc_test_scores(emb, pos_neg_tuple, neg_sample_type, device)
        pos_src_emb = emb[etype[0]][pos_src]
        pos_dst_emb = emb[etype[2]][pos_dst]
        pos_score = calc_dot_pos_score(pos_src_emb, pos_dst_emb)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            neg_dst_emb = emb[etype[2]][neg_dst[i]]
            # (dim) * (num_neg, dim)
            ns = calc_dot_pos_score(pse, neg_dst_emb)
            neg_scores.append(ns)
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etype, num_neg, pos_src.shape[0])

        pos_src, neg_src, pos_dst, _ = gen_edge_pairs()
        pos_neg_tuple = {etype: (pos_src, neg_src, pos_dst, None)}
        score = decoder.calc_test_scores(emb, pos_neg_tuple, neg_sample_type, device)
        pos_src_emb = emb[etype[0]][pos_src]
        pos_dst_emb = emb[etype[2]][pos_dst]
        pos_score = calc_dot_pos_score(pos_src_emb, pos_dst_emb)
        neg_scores = []
        for i in range(pos_dst.shape[0]):
            neg_src_emb = emb[etype[0]][neg_src[i]]
            pde = pos_dst_emb[i]
            # (num_neg, dim) * (dim)
            ns = calc_dot_pos_score(neg_src_emb, pde)
            neg_scores.append(ns)
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etype, num_neg, pos_src.shape[0])

        pos_src, neg_src, pos_dst, neg_dst = gen_edge_pairs()
        pos_neg_tuple = {etype: (pos_src, neg_src, pos_dst, neg_dst)}
        score = decoder.calc_test_scores(emb, pos_neg_tuple, neg_sample_type, device)
        pos_src_emb = emb[etype[0]][pos_src]
        pos_dst_emb = emb[etype[2]][pos_dst]
        pos_score = calc_dot_pos_score(pos_src_emb, pos_dst_emb)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            pde = pos_dst_emb[i]
            neg_src_emb = emb[etype[0]][neg_src[i]]
            neg_dst_emb = emb[etype[2]][neg_dst[i]]
            # (num_neg, dim) * (dim)
            ns_0 = calc_dot_pos_score(neg_src_emb, pde)
            # (dim) * (num_neg, dim)
            ns_1 = calc_dot_pos_score(pse, neg_dst_emb)
            neg_scores.append(th.cat((ns_0, ns_1), dim=-1))
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etype, num_neg*2, pos_src.shape[0])

def check_calc_test_scores_dot_joint_neg(decoder, etype, h_dim, num_pos, num_neg, device):
    neg_sample_type = BUILTIN_LP_JOINT_NEG_SAMPLER
    emb = {
        'a': th.rand((128, h_dim)),
        'b': th.rand((128, h_dim)),
    }

    def gen_edge_pairs():
        pos_src = th.randint(100, (num_pos,))
        pos_dst = th.randint(100, (num_pos,))
        neg_src = th.randint(128, (num_neg,))
        neg_dst = th.randint(128, (num_neg,))
        return (pos_src, neg_src, pos_dst, neg_dst)

    with th.no_grad():
        pos_src, _, pos_dst, neg_dst = gen_edge_pairs()
        pos_neg_tuple = {etype: (pos_src, None, pos_dst, neg_dst)}
        score = decoder.calc_test_scores(emb, pos_neg_tuple, neg_sample_type, device)
        pos_src_emb = emb[etype[0]][pos_src]
        pos_dst_emb = emb[etype[2]][pos_dst]
        pos_score = calc_dot_pos_score(pos_src_emb, pos_dst_emb)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            neg_dst_emb = emb[etype[2]][neg_dst]
            # (dim) * (num_neg, dim)
            ns = calc_dot_pos_score(pse, neg_dst_emb)
            neg_scores.append(ns)
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etype, num_neg, pos_src.shape[0])

        pos_src, neg_src, pos_dst, _ = gen_edge_pairs()
        pos_neg_tuple = {etype: (pos_src, neg_src, pos_dst, None)}
        score = decoder.calc_test_scores(emb, pos_neg_tuple, neg_sample_type, device)
        pos_src_emb = emb[etype[0]][pos_src]
        pos_dst_emb = emb[etype[2]][pos_dst]
        pos_score = calc_dot_pos_score(pos_src_emb, pos_dst_emb)
        neg_scores = []
        for i in range(pos_dst.shape[0]):
            neg_src_emb = emb[etype[0]][neg_src]
            pde = pos_dst_emb[i]
            # (num_neg, dim) * (dim)
            ns = calc_dot_pos_score(neg_src_emb, pde)
            neg_scores.append(ns)
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etype, num_neg, pos_src.shape[0])

        pos_src, neg_src, pos_dst, neg_dst = gen_edge_pairs()
        pos_neg_tuple = {etype: (pos_src, neg_src, pos_dst, neg_dst)}
        score = decoder.calc_test_scores(emb, pos_neg_tuple, neg_sample_type, device)
        pos_src_emb = emb[etype[0]][pos_src]
        pos_dst_emb = emb[etype[2]][pos_dst]
        pos_score = calc_dot_pos_score(pos_src_emb, pos_dst_emb)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            pde = pos_dst_emb[i]
            neg_src_emb = emb[etype[0]][neg_src]
            neg_dst_emb = emb[etype[2]][neg_dst]
            # (num_neg, dim) * (dim)
            ns_0 = calc_dot_pos_score(neg_src_emb, pde)
            # (dim) * (num_neg, dim)
            ns_1 = calc_dot_pos_score(pse, neg_dst_emb)
            neg_scores.append(th.cat((ns_0, ns_1), dim=-1))
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etype, num_neg*2, pos_src.shape[0])

@pytest.mark.parametrize("h_dim", [16, 64])
@pytest.mark.parametrize("num_pos", [8, 32])
@pytest.mark.parametrize("num_neg", [1, 32])
@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_LinkPredictDistMultDecoder(h_dim, num_pos, num_neg, device):
    th.manual_seed(0)
    etypes = [('a', 'r1', 'b'), ('a', 'r2', 'b')]
    decoder = LinkPredictDistMultDecoder(etypes, h_dim)
    # mimic that decoder has been trained.
    decoder.trained_rels[0] = 1
    decoder.trained_rels[1] = 1

    check_calc_test_scores_uniform_neg(decoder, etypes, h_dim, num_pos, num_neg, device)
    check_calc_test_scores_joint_neg(decoder, etypes, h_dim, num_pos, num_neg, device)

@pytest.mark.parametrize("h_dim", [16, 64])
@pytest.mark.parametrize("num_pos", [8, 32])
@pytest.mark.parametrize("num_neg", [1, 32])
@pytest.mark.parametrize("device",["cpu", "cuda:0"])
def test_LinkPredictDotDecoder(h_dim, num_pos, num_neg, device):
    th.manual_seed(1)
    etype = ('a', 'r1', 'b')
    decoder = LinkPredictDotDecoder(h_dim)

    check_calc_test_scores_dot_uniform_neg(decoder, etype, h_dim, num_pos, num_neg, device)
    check_calc_test_scores_dot_joint_neg(decoder, etype, h_dim, num_pos, num_neg, device)

@pytest.mark.parametrize("h_dim", [16, 64])
@pytest.mark.parametrize("feat_dim", [8, 32])
@pytest.mark.parametrize("out_dim", [2, 32])
@pytest.mark.parametrize("num_ffn_layers", [0, 2])
def test_MLPEFeatEdgeDecoder(h_dim, feat_dim, out_dim, num_ffn_layers):
    g = generate_dummy_hetero_graph()
    target_etype = ("n0", "r0", "n1")
    encoder_feat = {
        "n0": th.randn(g.num_nodes("n0"), h_dim),
        "n1": th.randn(g.num_nodes("n1"), h_dim)
    }
    efeat = th.randn(g.num_edges(target_etype), feat_dim)
    g.edges[target_etype].data[EP_DECODER_EDGE_FEAT] = efeat


    decoder = MLPEFeatEdgeDecoder(h_dim,
                                  feat_dim,
                                  out_dim,
                                  multilabel=False,
                                  target_etype=target_etype,
                                  num_ffn_layers=num_ffn_layers)
    with th.no_grad():
        decoder.eval()
        output = decoder(g, encoder_feat)
        u, v = g.edges(etype=target_etype)
        ufeat = encoder_feat["n0"][u]
        ifeat = encoder_feat["n1"][v]
        h = th.cat([ufeat, ifeat], dim=1)
        nn_h = th.matmul(h, decoder.nn_decoder)
        nn_h = decoder.relu(nn_h)

        feat_h = th.matmul(efeat, decoder.feat_decoder)
        feat_h = decoder.relu(feat_h)
        combine_h = th.cat([nn_h, feat_h], dim=1)
        if num_ffn_layers > 0:
            combine_h = decoder.ngnn_mlp(combine_h)
        combine_h = th.matmul(combine_h, decoder.combine_decoder)
        combine_h = decoder.relu(combine_h)
        out = th.matmul(combine_h, decoder.decoder)

        assert_almost_equal(output.cpu().numpy(), out.cpu().numpy())

        prediction = decoder.predict(g, encoder_feat)
        pred = out.argmax(dim=1)
        assert_almost_equal(prediction.cpu().numpy(), pred.cpu().numpy())

def test_get_edge_weight():
    g = generate_dummy_hetero_graph()
    g.edges[("n0", "r0", "n1")].data['weight'] = \
        th.randn(g.num_edges(("n0", "r0", "n1")))
    g.edges[("n0", "r1", "n1")].data['weight'] = \
        th.randn(g.num_edges(("n0", "r1", "n1")))
    g.edges[("n0", "r1", "n1")].data['weight1'] = \
        th.randn((g.num_edges(("n0", "r1", "n1")), 1))
    # edata with wrong shape
    g.edges[("n0", "r1", "n1")].data['weight2'] = \
        th.randn((g.num_edges(("n0", "r1", "n1")), 2))

    weight = _get_edge_weight(g, "weight", ("n0", "r0", "n1"))
    assert_equal(g.edges[("n0", "r0", "n1")].data['weight'].numpy(),
                 weight.numpy())
    # weight1 does not exist in g.edges(("n0", "r0", "n1"))
    weight = _get_edge_weight(g, "weight1", ("n0", "r0", "n1"))
    assert_equal(th.ones(g.num_edges(("n0", "r0", "n1")),).numpy(),
                 weight.numpy())

    weight = _get_edge_weight(g, "weight", ("n0", "r1", "n1"))
    assert_equal(g.edges[("n0", "r1", "n1")].data['weight'].numpy(),
                 weight.numpy())

if __name__ == '__main__':
    test_LinkPredictDistMultDecoder(16, 8, 1, "cpu")
    test_LinkPredictDistMultDecoder(16, 32, 32, "cuda:0")
    test_LinkPredictDotDecoder(16, 8, 1, "cpu")
    test_LinkPredictDotDecoder(16, 32, 32, "cuda:0")

    test_MLPEFeatEdgeDecoder(16,8,2)
    test_MLPEFeatEdgeDecoder(16,32,2)

    test_get_edge_weight()
