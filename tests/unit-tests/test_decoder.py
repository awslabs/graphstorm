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
import math
import pytest

import dgl
import torch as th
from numpy.testing import assert_almost_equal

from graphstorm.model import (LinkPredictDotDecoder,
                              LinkPredictDistMultDecoder,
                              EntityRegression,
                              MLPEFeatEdgeDecoder,
                              LinkPredictContrastiveDotDecoder,
                              LinkPredictContrastiveDistMultDecoder,
                              LinkPredictRotatEDecoder,
                              LinkPredictContrastiveRotatEDecoder)
from graphstorm.dataloading import (BUILTIN_LP_UNIFORM_NEG_SAMPLER,
                                    BUILTIN_LP_JOINT_NEG_SAMPLER)
from graphstorm.eval.utils import (calc_distmult_pos_score,
                                   calc_rotate_pos_score)
from graphstorm.eval.utils import calc_dot_pos_score
from graphstorm.eval.utils import calc_ranking

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
    assert_almost_equal(score[etype][0].cpu().numpy(), pos_score.cpu().numpy(), decimal=5)
    assert_almost_equal(score[etype][1].cpu().numpy(), neg_scores.cpu().numpy(), decimal=5)

def _check_ranking(score, pos_score, neg_scores, etype, num_neg, batch_size):
    assert score[etype][0].shape[0] == batch_size
    assert len(score[etype][0].shape) == 1
    # neg scores
    assert len(score[etype][1].shape) == 2
    assert score[etype][1].shape[0] == batch_size
    assert score[etype][1].shape[1] == num_neg

    p_score = score[etype][0].cpu()
    n_score = score[etype][1].cpu()
    test_ranking = calc_ranking(p_score, n_score)
    ranking = calc_ranking(pos_score.cpu(), neg_scores.cpu())

    assert_almost_equal(test_ranking.numpy(), ranking.numpy())

def check_calc_test_scores_rotate_uniform_neg(decoder, etypes, h_dim, num_pos, num_neg, device):
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
        pos_score = calc_rotate_pos_score(pos_src_emb, pos_dst_emb, rel_emb,
            decoder.emb_init, decoder.gamma)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            neg_dst_emb = emb[etypes[0][2]][neg_dst[i]]
            ns = calc_rotate_pos_score(pse, neg_dst_emb, rel_emb,
                decoder.emb_init, decoder.gamma)
            neg_scores.append(ns)
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etypes[0], num_neg, pos_src.shape[0])

        pos_src, neg_src, pos_dst, _ = pos_neg_tuple[etypes[1]]
        pos_src_emb = emb[etypes[1][0]][pos_src]
        pos_dst_emb = emb[etypes[1][2]][pos_dst]
        rel_emb = decoder.get_relemb(etypes[1])
        pos_score = calc_rotate_pos_score(pos_src_emb, pos_dst_emb, rel_emb,
            decoder.emb_init, decoder.gamma)
        neg_scores = []
        for i in range(pos_dst.shape[0]):
            neg_src_emb = emb[etypes[1][0]][neg_src[i]]
            pde = pos_dst_emb[i]
            ns = calc_rotate_pos_score(neg_src_emb, pde, rel_emb,
                decoder.emb_init, decoder.gamma)
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
        pos_score = calc_rotate_pos_score(pos_src_emb, pos_dst_emb, rel_emb,
            decoder.emb_init, decoder.gamma)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            pde = pos_dst_emb[i]
            neg_src_emb = emb[etypes[0][0]][neg_src[i]]
            neg_dst_emb = emb[etypes[0][2]][neg_dst[i]]
            # (num_neg, dim) * (dim) * (dim)
            ns_0 = calc_rotate_pos_score(neg_src_emb, pde, rel_emb,
                decoder.emb_init, decoder.gamma)
            # (dim) * (dim) * (num_neg, dim)
            ns_1 = calc_rotate_pos_score(pse, neg_dst_emb, rel_emb,
                decoder.emb_init, decoder.gamma)
            neg_scores.append(th.cat((ns_0, ns_1), dim=-1))
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etypes[0], num_neg*2, pos_src.shape[0])


def check_calc_test_scores_rotate_joint_neg(decoder, etypes, h_dim, num_pos, num_neg, device):
    neg_sample_type = BUILTIN_LP_JOINT_NEG_SAMPLER
    emb = {
        'a': th.rand((128, h_dim)),
        'b': th.rand((128, h_dim)),
    }

    def gen_edge_pairs():
        pos_src = th.ones((num_pos,), dtype=int)
        pos_dst = th.randint(100, (num_pos,))
        neg_src = th.randint(128, (num_neg,))
        neg_dst = th.randint(128, (num_neg,))
        neg_src[neg_src==1] = 2
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
        pos_score = calc_rotate_pos_score(pos_src_emb, pos_dst_emb, rel_emb,
            decoder.emb_init, decoder.gamma)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            neg_dst_emb = emb[etypes[0][2]][neg_dst]
            # (dim) * (dim) * (num_neg, dim)
            ns = calc_rotate_pos_score(pse, neg_dst_emb, rel_emb,
                decoder.emb_init, decoder.gamma)
            neg_scores.append(ns)
        neg_scores = th.stack(neg_scores)
        _check_ranking(score, pos_score, neg_scores, etypes[0], num_neg, pos_src.shape[0])

        pos_src, neg_src, pos_dst, _ = pos_neg_tuple[etypes[1]]
        pos_src_emb = emb[etypes[1][0]][pos_src]
        pos_dst_emb = emb[etypes[1][2]][pos_dst]
        rel_emb = decoder.get_relemb(etypes[1])
        pos_score = calc_rotate_pos_score(pos_src_emb, pos_dst_emb, rel_emb,
            decoder.emb_init, decoder.gamma)
        neg_scores = []
        for i in range(pos_dst.shape[0]):
            neg_src_emb = emb[etypes[1][0]][neg_src]
            pde = pos_dst_emb[i]
            # (num_neg, dim) * (dim) * (dim)
            ns = calc_rotate_pos_score(neg_src_emb, pde, rel_emb,
                decoder.emb_init, decoder.gamma)
            neg_scores.append(ns)
        neg_scores = th.stack(neg_scores)
        _check_ranking(score, pos_score, neg_scores, etypes[1], num_neg, pos_src.shape[0])

        pos_neg_tuple = {
            etypes[0]: gen_edge_pairs(),
        }
        score = decoder.calc_test_scores(emb, pos_neg_tuple, neg_sample_type, device)
        pos_src, neg_src, pos_dst, neg_dst = pos_neg_tuple[etypes[0]]
        pos_src_emb = emb[etypes[0][0]][pos_src]
        pos_dst_emb = emb[etypes[0][2]][pos_dst]
        rel_emb = decoder.get_relemb(etypes[0])
        pos_score = calc_rotate_pos_score(pos_src_emb, pos_dst_emb, rel_emb,
            decoder.emb_init, decoder.gamma)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            pde = pos_dst_emb[i]
            neg_src_emb = emb[etypes[0][0]][neg_src]
            neg_dst_emb = emb[etypes[0][2]][neg_dst]

            ns_0 = calc_rotate_pos_score(neg_src_emb, pde, rel_emb,
                decoder.emb_init, decoder.gamma)
            ns_1 = calc_rotate_pos_score(pse, neg_dst_emb, rel_emb,
                decoder.emb_init, decoder.gamma)
            neg_scores.append(th.cat((ns_0, ns_1), dim=-1))
        neg_scores = th.stack(neg_scores)
        _check_ranking(score, pos_score, neg_scores, etypes[0], num_neg*2, pos_src.shape[0])


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
        pos_score = calc_distmult_pos_score(pos_src_emb, pos_dst_emb, rel_emb)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            neg_dst_emb = emb[etypes[0][2]][neg_dst[i]]
            # (dim) * (dim) * (num_neg, dim)
            ns = calc_distmult_pos_score(pse, neg_dst_emb, rel_emb)
            neg_scores.append(ns)
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etypes[0], num_neg, pos_src.shape[0])

        pos_src, neg_src, pos_dst, _ = pos_neg_tuple[etypes[1]]
        pos_src_emb = emb[etypes[1][0]][pos_src]
        pos_dst_emb = emb[etypes[1][2]][pos_dst]
        rel_emb = decoder.get_relemb(etypes[1])
        pos_score = calc_distmult_pos_score(pos_src_emb, pos_dst_emb, rel_emb)
        neg_scores = []
        for i in range(pos_dst.shape[0]):
            neg_src_emb = emb[etypes[1][0]][neg_src[i]]
            pde = pos_dst_emb[i]
            # (num_neg, dim) * (dim) * (dim)
            ns = calc_distmult_pos_score(neg_src_emb, pde, rel_emb)
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
        pos_score = calc_distmult_pos_score(pos_src_emb, pos_dst_emb, rel_emb)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            pde = pos_dst_emb[i]
            neg_src_emb = emb[etypes[0][0]][neg_src[i]]
            neg_dst_emb = emb[etypes[0][2]][neg_dst[i]]
            # (num_neg, dim) * (dim) * (dim)
            ns_0 = calc_distmult_pos_score(neg_src_emb, pde, rel_emb)
            # (dim) * (dim) * (num_neg, dim)
            ns_1 = calc_distmult_pos_score(pse, neg_dst_emb, rel_emb)
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
        pos_score = calc_distmult_pos_score(pos_src_emb, pos_dst_emb, rel_emb)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            neg_dst_emb = emb[etypes[0][2]][neg_dst]
            # (dim) * (dim) * (num_neg, dim)
            ns = calc_distmult_pos_score(pse, neg_dst_emb, rel_emb)
            neg_scores.append(ns)
        neg_scores = th.stack(neg_scores)
        _check_scores(score, pos_score, neg_scores, etypes[0], num_neg, pos_src.shape[0])

        pos_src, neg_src, pos_dst, _ = pos_neg_tuple[etypes[1]]
        pos_src_emb = emb[etypes[1][0]][pos_src]
        pos_dst_emb = emb[etypes[1][2]][pos_dst]
        rel_emb = decoder.get_relemb(etypes[1])
        pos_score = calc_distmult_pos_score(pos_src_emb, pos_dst_emb, rel_emb)
        neg_scores = []
        for i in range(pos_dst.shape[0]):
            neg_src_emb = emb[etypes[1][0]][neg_src]
            pde = pos_dst_emb[i]
            # (num_neg, dim) * (dim) * (dim)
            ns = calc_distmult_pos_score(neg_src_emb, pde, rel_emb)
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
        pos_score = calc_distmult_pos_score(pos_src_emb, pos_dst_emb, rel_emb)
        neg_scores = []
        for i in range(pos_src.shape[0]):
            pse = pos_src_emb[i]
            pde = pos_dst_emb[i]
            neg_src_emb = emb[etypes[0][0]][neg_src]
            neg_dst_emb = emb[etypes[0][2]][neg_dst]
            # (num_neg, dim) * (dim) * (dim)
            ns_0 = calc_distmult_pos_score(neg_src_emb, pde, rel_emb)
            # (dim) * (dim) * (num_neg, dim)
            ns_1 = calc_distmult_pos_score(pse, neg_dst_emb, rel_emb)
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
def test_LinkPredictRotatEDecoder(h_dim, num_pos, num_neg, device):
    th.manual_seed(0)
    etypes = [('a', 'r1', 'b'), ('a', 'r2', 'b')]
    decoder = LinkPredictRotatEDecoder(etypes, h_dim, gamma=4.)
    # mimic that decoder has been trained.
    decoder.trained_rels[0] = 1
    decoder.trained_rels[1] = 1

    check_calc_test_scores_rotate_uniform_neg(decoder, etypes, h_dim, num_pos, num_neg, device)
    check_calc_test_scores_rotate_joint_neg(decoder, etypes, h_dim, num_pos, num_neg, device)

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

def check_forward(decoder, etype, h_dim, num_pos, num_neg, comput_score, device):
    n0_embs = th.rand((1000, h_dim), device=device)
    n1_embs = th.rand((1000, h_dim), device=device)

    src = th.arange(num_pos)
    dst = th.arange(num_pos)
    dst_neg = th.randint(1000, (num_neg,))

    src_randidx = th.randperm(num_pos)
    new_src = src[src_randidx]
    new_dst = dst[src_randidx]

    src_neg = src.reshape(-1, 1).repeat(1, num_neg).reshape(-1,)
    dst_neg = dst_neg.repeat(num_pos)

    new_src_neg = new_src.reshape(-1, 1).repeat(1, num_neg).reshape(-1,)

    pos_g = dgl.heterograph(
        {etype: (new_src, new_dst)},
        num_nodes_dict={
            etype[0]: 1000,
            etype[2]: 1000,
        })

    neg_g = dgl.heterograph(
        {etype: (new_src_neg, dst_neg)},
        num_nodes_dict={
            etype[0]: 1000,
            etype[2]: 1000,
        })

    pos_g = pos_g.to(device)
    neg_g = neg_g.to(device)

    pos_scores = decoder(pos_g, {etype[0]: n0_embs, etype[2]:n1_embs})
    neg_scores = decoder(neg_g, {etype[0]: n0_embs, etype[2]:n1_embs})

    pos_scores_ = comput_score(n0_embs[src.to(device)], n1_embs[dst.to(device)])
    neg_scores_ = comput_score(n0_embs[src_neg.to(device)], n1_embs[dst_neg.to(device)])

    assert_almost_equal(pos_scores[etype].detach().cpu().numpy(),
                        pos_scores_.detach().cpu().numpy())
    assert_almost_equal(neg_scores[etype].detach().cpu().numpy(),
                        neg_scores_.detach().cpu().numpy())

@pytest.mark.parametrize("h_dim", [16, 64])
@pytest.mark.parametrize("num_pos", [8, 32])
@pytest.mark.parametrize("num_neg", [1, 32])
@pytest.mark.parametrize("device",["cpu", "cuda:0"])
def test_LinkPredictContrastiveDotDecoder(h_dim, num_pos, num_neg, device):
    th.manual_seed(1)
    etype = ('a', 'r1', 'b')
    decoder = LinkPredictContrastiveDotDecoder(h_dim)
    def comput_score(src_emb, dst_emb):
        return th.sum(src_emb * dst_emb, dim=-1)

    check_forward(decoder, etype, h_dim, num_pos, num_neg, comput_score, device)

@pytest.mark.parametrize("h_dim", [16, 64])
@pytest.mark.parametrize("num_pos", [8, 32])
@pytest.mark.parametrize("num_neg", [1, 32])
@pytest.mark.parametrize("device",["cpu", "cuda:0"])
def test_LinkPredictContrastiveDistMultDecoder(h_dim, num_pos, num_neg, device):
    th.manual_seed(1)
    etype = ('a', 'r1', 'b')
    decoder = LinkPredictContrastiveDistMultDecoder([etype], h_dim)
    decoder.trained_rels[0] = 1 # trick the decoder
    decoder = decoder.to(device)
    rel_emb = decoder.get_relemb(etype).to(device)
    def comput_score(src_emb, dst_emb):
        return th.sum(src_emb * rel_emb * dst_emb, dim=-1)

    check_forward(decoder, etype, h_dim, num_pos, num_neg, comput_score, device)

@pytest.mark.parametrize("h_dim", [16, 64])
@pytest.mark.parametrize("num_pos", [8, 32])
@pytest.mark.parametrize("num_neg", [1, 32])
@pytest.mark.parametrize("device",["cpu", "cuda:0"])
def test_LinkPredictContrastiveRotatEDecoder(h_dim, num_pos, num_neg, device):
    th.manual_seed(1)
    etype = ('a', 'r1', 'b')
    gamma = 4.
    decoder = LinkPredictContrastiveRotatEDecoder([etype], h_dim, gamma=gamma)
    decoder.trained_rels[0] = 1 # trick the decoder
    decoder = decoder.to(device)
    rel_emb = decoder.get_relemb(etype).to(device)
    emb_init = decoder.emb_init

    def comput_score(src_emb, dst_emb):
        real_head, imag_head = th.chunk(src_emb, 2, dim=-1)
        real_tail, imag_tail = th.chunk(dst_emb, 2, dim=-1)

        phase_rel = rel_emb / (emb_init / th.tensor(math.pi))
        real_rel, imag_rel = th.cos(phase_rel), th.sin(phase_rel)
        real_score = real_head * real_rel - imag_head * imag_rel
        imag_score = real_head * imag_rel + imag_head * real_rel
        real_score = real_score - real_tail
        imag_score = imag_score - imag_tail
        score = th.stack([real_score, imag_score], dim=0)
        score = score.norm(dim=0)

        return gamma - score.sum(-1)

    check_forward(decoder, etype, h_dim, num_pos, num_neg, comput_score, device)

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
    efeat = {target_etype: th.randn(g.num_edges(target_etype), feat_dim)}
    norm = None
    decoder = MLPEFeatEdgeDecoder(h_dim,
                                  feat_dim,
                                  out_dim,
                                  multilabel=False,
                                  target_etype=target_etype,
                                  num_ffn_layers=num_ffn_layers,
                                  norm=norm)
    with th.no_grad():
        decoder.eval()
        output = decoder(g, encoder_feat, efeat)
        u, v = g.edges(etype=target_etype)
        ufeat = encoder_feat["n0"][u]
        ifeat = encoder_feat["n1"][v]
        h = th.cat([ufeat, ifeat], dim=1)
        nn_h = th.matmul(h, decoder.nn_decoder)
        nn_h = decoder.relu(nn_h)

        feat_h = th.matmul(efeat[target_etype], decoder.feat_decoder)
        feat_h = decoder.relu(feat_h)
        combine_h = th.cat([nn_h, feat_h], dim=1)
        if num_ffn_layers > 0:
            combine_h = decoder.ngnn_mlp(combine_h)
        combine_h = th.matmul(combine_h, decoder.combine_decoder)
        combine_h = decoder.relu(combine_h)
        out = th.matmul(combine_h, decoder.decoder)

        assert_almost_equal(output.cpu().numpy(), out.cpu().numpy())

        prediction = decoder.predict(g, encoder_feat, efeat)
        pred = out.argmax(dim=1)
        assert_almost_equal(prediction.cpu().numpy(), pred.cpu().numpy())

    norm = "layer"
    decoder = MLPEFeatEdgeDecoder(h_dim,
                                  feat_dim,
                                  out_dim,
                                  multilabel=False,
                                  target_etype=target_etype,
                                  num_ffn_layers=num_ffn_layers,
                                  norm=norm)
    with th.no_grad():
        decoder.eval()
        output = decoder(g, encoder_feat, efeat)
        u, v = g.edges(etype=target_etype)
        ufeat = encoder_feat["n0"][u]
        ifeat = encoder_feat["n1"][v]
        h = th.cat([ufeat, ifeat], dim=1)
        nn_h = th.matmul(h, decoder.nn_decoder)
        nn_h = decoder.nn_decoder_norm(nn_h)
        nn_h = decoder.relu(nn_h)

        feat_h = th.matmul(efeat[target_etype], decoder.feat_decoder)
        feat_h = decoder.feat_decoder_norm(feat_h)
        feat_h = decoder.relu(feat_h)
        combine_h = th.cat([nn_h, feat_h], dim=1)
        if num_ffn_layers > 0:
            combine_h = decoder.ngnn_mlp(combine_h)
        combine_h = th.matmul(combine_h, decoder.combine_decoder)
        combine_h = decoder.combine_norm(combine_h)
        combine_h = decoder.relu(combine_h)
        out = th.matmul(combine_h, decoder.decoder)

        assert_almost_equal(output.cpu().numpy(), out.cpu().numpy())

        prediction = decoder.predict(g, encoder_feat, efeat)
        pred = out.argmax(dim=1)
        assert_almost_equal(prediction.cpu().numpy(), pred.cpu().numpy())
@pytest.mark.parametrize("in_dim", [16, 64])
@pytest.mark.parametrize("out_dim", [1, 8])
def test_EntityRegression(in_dim, out_dim):
    decoder = EntityRegression(h_dim=in_dim)
    assert decoder.in_dims == in_dim
    assert decoder.out_dims == 1

    decoder = EntityRegression(h_dim=in_dim, out_dim=out_dim)
    assert decoder.in_dims == in_dim
    assert decoder.out_dims == out_dim

if __name__ == '__main__':
    test_LinkPredictRotatEDecoder(16, 8, 1, "cpu")
    test_LinkPredictRotatEDecoder(16, 32, 32, "cuda:0")

    test_LinkPredictContrastiveRotatEDecoder(32, 8, 16, "cpu")
    test_LinkPredictContrastiveRotatEDecoder(16, 32, 32, "cuda:0")

    test_EntityRegression(8, 1)
    test_EntityRegression(8, 8)

    test_LinkPredictContrastiveDistMultDecoder(32, 8, 16, "cpu")
    test_LinkPredictContrastiveDistMultDecoder(16, 32, 32, "cuda:0")
    test_LinkPredictContrastiveDotDecoder(32, 8, 16, "cpu")
    test_LinkPredictContrastiveDotDecoder(16, 32, 32, "cuda:0")

    test_LinkPredictDistMultDecoder(16, 8, 1, "cpu")
    test_LinkPredictDistMultDecoder(16, 32, 32, "cuda:0")
    test_LinkPredictDotDecoder(16, 8, 1, "cpu")
    test_LinkPredictDotDecoder(16, 32, 32, "cuda:0")

    test_MLPEFeatEdgeDecoder(16,8,2,0)
    test_MLPEFeatEdgeDecoder(16,32,2,2)
