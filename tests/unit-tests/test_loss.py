"""
    Copyright 2024 Contributors

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
import torch.nn.functional as F

from numpy.testing import assert_almost_equal

from graphstorm.model.loss_func import (LinkPredictAdvBCELossFunc,
                                        WeightedLinkPredictAdvBCELossFunc)

@pytest.mark.parametrize("num_pos", [1, 8, 32])
@pytest.mark.parametrize("num_neg", [1, 8, 32])
def test_LinkPredictAdvBCELossFunc(num_pos, num_neg):
    pos_score = {
        ("n0", "r0" ,"n1"): th.rand((num_pos,)) + 0.5,
        ("n0", "r1", "n2"): th.rand((num_pos,)) + 0.3
    }

    neg_score = {
        ("n0", "r0" ,"n1"): th.rand((num_neg,)) - 0.4,
        ("n0", "r1", "n2"): th.rand((num_neg,)) - 0.3
    }

    adversarial_temperature = 0.1
    loss_func = LinkPredictAdvBCELossFunc(adversarial_temperature)
    loss = loss_func(pos_score, neg_score)

    p_score = th.cat([pos_score[("n0", "r0" ,"n1")], pos_score[("n0", "r1", "n2")]])
    n_score = th.cat([neg_score[("n0", "r0" ,"n1")], neg_score[("n0", "r1", "n2")]])
    p_loss = -th.log(F.sigmoid(p_score))
    n_loss = -th.log(1 - F.sigmoid(n_score))

    n_loss = th.softmax(n_score * adversarial_temperature, dim=-1) * n_loss
    n_loss = th.mean(th.sum(n_loss, dim=-1))
    p_loss = th.mean(p_loss)
    gt_loss = (n_loss + p_loss) / 2

    assert_almost_equal(loss.numpy(),gt_loss.numpy())

@pytest.mark.parametrize("num_pos", [1, 8, 32])
@pytest.mark.parametrize("num_neg", [1, 8, 32])
def test_WeightedLinkPredictAdvBCELossFunc(num_pos, num_neg):
    pos_score = {
        ("n0", "r0" ,"n1"): (th.rand((num_pos,)) + 0.5, th.rand((num_pos,)) + 2),
        ("n0", "r1", "n2"): (th.rand((num_pos,)) + 0.3, th.rand((num_pos,)) + 2)
    }

    neg_score = {
        ("n0", "r0" ,"n1"): (th.rand((num_neg,)) - 0.4, None),
        ("n0", "r1", "n2"): (th.rand((num_neg,)) - 0.3, None)
    }

    adversarial_temperature = 0.1
    loss_func = WeightedLinkPredictAdvBCELossFunc(adversarial_temperature)
    loss = loss_func(pos_score, neg_score)

    p_score = th.cat([pos_score[("n0", "r0" ,"n1")][0], pos_score[("n0", "r1", "n2")][0]])
    p_weight = th.cat([pos_score[("n0", "r0" ,"n1")][1], pos_score[("n0", "r1", "n2")][1]])
    n_score = th.cat([neg_score[("n0", "r0" ,"n1")][0], neg_score[("n0", "r1", "n2")][0]])

    p_loss = -th.log(F.sigmoid(p_score))
    p_loss = p_loss * p_weight
    n_loss = -th.log(1 - F.sigmoid(n_score))
    n_loss = th.softmax(n_score * adversarial_temperature, dim=-1) * n_loss

    n_loss = th.mean(th.sum(n_loss, dim=-1))
    p_loss = th.mean(p_loss)
    gt_loss = (n_loss + p_loss) / 2

    assert_almost_equal(loss.numpy(),gt_loss.numpy())

if __name__ == '__main__':
    test_LinkPredictAdvBCELossFunc(16, 128)
    test_WeightedLinkPredictAdvBCELossFunc(16, 128)