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
                                        WeightedLinkPredictAdvBCELossFunc,
                                        FocalLossFunc)

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

def test_FocalLossFunc():
    alpha = 0.25
    gamma = 2.

    loss_func = FocalLossFunc(alpha, gamma)
    logits = th.tensor([[0.6330],[0.9946],[0.2322],[0.0115],[0.9159],[0.5752],[0.4491], [0.9231],[0.7170],[0.2761]])
    labels = th.tensor([0, 0, 0, 1, 1, 1, 0, 1, 0, 0])
    # Manually call the torchvision.ops.sigmoid_focal_loss to generate the loss value
    gt_loss = th.tensor(0.1968)
    loss = loss_func(logits, labels)
    assert_almost_equal(loss.numpy(), gt_loss.numpy(), decimal=4)

    alpha = 0.2
    gamma = 1.5
    loss_func = FocalLossFunc(alpha, gamma)
    logits = th.tensor([2.8205, 0.4035, 0.8215, 1.9420, 0.2400, 2.8565, 1.8330, 0.7786, 2.0962, 1.0399])
    labels = th.tensor([0, 0, 1, 0, 1, 1, 1, 1, 0, 0])
    # Manually call the torchvision.ops.sigmoid_focal_loss to generate the loss value
    gt_loss = th.tensor(0.6040)
    loss = loss_func(logits, labels)
    assert_almost_equal(loss.numpy(), gt_loss.numpy(), decimal=4)


if __name__ == '__main__':
    test_FocalLossFunc()

    test_LinkPredictAdvBCELossFunc(16, 128)
    test_WeightedLinkPredictAdvBCELossFunc(16, 128)
