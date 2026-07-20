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
                                        FocalLossFunc,
                                        LinkPredictBPRLossFunc,
                                        WeightedLinkPredictBPRLossFunc,
                                        ShrinkageLossFunc)

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
    # Test case 1: Strong predictions
    alpha = 0.25
    gamma = 2.0

    loss_func = FocalLossFunc(alpha, gamma)
    # Create logits for both classes
    logits = th.tensor([
        [2.0, -2.0],  # Strong prediction for class 0
        [-3.0, 3.0],  # Strong prediction for class 1
        [0.1, -0.1],  # Weak prediction for class 0
        [-0.2, 0.2]   # Weak prediction for class 1
    ])
    labels = th.tensor([0, 1, 0, 1])

    # Get our implementation's loss
    our_loss = loss_func(logits, labels)

    # Get torchvision's loss using the positive class logits
    # To get the results we used:
    # from torchvision.ops import sigmoid_focal_loss
    # tv_loss = sigmoid_focal_loss(
    #     logits[:, 1],  # Take logits for positive class
    #     labels.float(),
    #     alpha=alpha,
    #     gamma=gamma,
    #     reduction='mean'
    # )
    tv_loss = th.tensor(0.0352)

    assert_almost_equal(our_loss.numpy(), tv_loss.numpy(), decimal=4)

    # Test case 2: Original test case
    alpha = 0.2
    gamma = 1.5
    loss_func = FocalLossFunc(alpha, gamma)

    logits_orig = th.tensor([
        [2.8205, -2.8205], [0.4035, -0.4035], [0.8215, -0.8215],
        [1.9420, -1.9420], [0.2400, -0.2400], [2.8565, -2.8565],
        [1.8330, -1.8330], [0.7786, -0.7786], [2.0962, -2.0962],
        [1.0399, -1.0399]
    ])
    labels = th.tensor([0, 0, 1, 0, 1, 1, 1, 1, 0, 0])

    # Get our implementation's loss
    our_loss = loss_func(logits_orig, labels)

    # Get torchvision's loss
    tv_loss = th.tensor(0.1335)

    assert_almost_equal(our_loss.numpy(), tv_loss.numpy(), decimal=4)

@pytest.mark.parametrize("num_pos", [1, 8, 32])
@pytest.mark.parametrize("num_neg", [1, 8, 32])
def test_LinkPredictBPRLossFunc(num_pos, num_neg):
    pos_score = {
        ("n0", "r0" ,"n1"): th.rand((num_pos,)) + 0.5,
        ("n0", "r1", "n2"): th.rand((num_pos,)) + 0.3
    }

    neg_score = {
        ("n0", "r0" ,"n1"): th.rand((num_pos, num_neg)) - 0.4,
        ("n0", "r1", "n2"): th.rand((num_pos, num_neg,)) - 0.3
    }

    loss_func = LinkPredictBPRLossFunc()
    loss = loss_func(pos_score, neg_score)
    p_score_r0 = pos_score[("n0", "r0" ,"n1")].unsqueeze(1).repeat(1, num_neg)
    p_score_r1 = pos_score[("n0", "r1", "n2")].unsqueeze(1).repeat(1, num_neg)
    dist_r0 = p_score_r0 - neg_score[("n0", "r0" ,"n1")]
    dist_r1 = p_score_r1 - neg_score[("n0", "r1", "n2")]

    loss_r0 = - th.log(1/(1+th.exp(-dist_r0)))
    loss_r1 = - th.log(1/(1+th.exp(-dist_r1)))

    gt_loss = (loss_r0.mean() + loss_r1.mean()) / 2
    assert_almost_equal(loss.numpy(),gt_loss.numpy())

@pytest.mark.parametrize("num_pos", [1, 8, 32])
@pytest.mark.parametrize("num_neg", [1, 8, 32])
def test_WeightedLinkPredictBPRLossFunc(num_pos, num_neg):
    pos_score = {
        ("n0", "r0" ,"n1"): (th.rand((num_pos,)) + 0.5, th.rand((num_pos,))/2 + 0.5),
        ("n0", "r1", "n2"): (th.rand((num_pos,)) + 0.3, th.rand((num_pos,))/2 + 0.5)
    }

    neg_score = {
        ("n0", "r0" ,"n1"): (th.rand((num_pos, num_neg)) - 0.4, None),
        ("n0", "r1", "n2"): (th.rand((num_pos, num_neg,)) - 0.3, None)
    }

    loss_func = WeightedLinkPredictBPRLossFunc()
    loss = loss_func(pos_score,
                     neg_score)
    p_score_r0 = pos_score[("n0", "r0" ,"n1")][0].unsqueeze(1).repeat(1, num_neg)
    p_score_r1 = pos_score[("n0", "r1", "n2")][0].unsqueeze(1).repeat(1, num_neg)
    pos_weight_r0 = pos_score[("n0", "r0" ,"n1")][1].unsqueeze(1).repeat(1, num_neg)
    pos_weight_r1 = pos_score[("n0", "r1", "n2")][1].unsqueeze(1).repeat(1, num_neg)
    dist_r0 = p_score_r0 - neg_score[("n0", "r0" ,"n1")][0]
    dist_r1 = p_score_r1 - neg_score[("n0", "r1", "n2")][0]

    loss_r0 = - th.log(1/(1+th.exp(-dist_r0)))
    loss_r1 = - th.log(1/(1+th.exp(-dist_r1)))

    loss_r0 = loss_r0 * pos_weight_r0
    loss_r1 = loss_r1 * pos_weight_r1

    gt_loss = (loss_r0.mean() + loss_r1.mean()) / 2
    assert_almost_equal(loss.numpy(),gt_loss.numpy())

def test_ShrinkageLossFunc():
    alpha = 10.
    gamma = 0.2
    loss_func = ShrinkageLossFunc(alpha, gamma)
    logits = th.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    labels = th.tensor([1., 1., 1., 1., 1., 1., 1., 1., 0, 0.9])
    # Loss is computed following:
    # l = abs(logits - labels)
    # loss = \frac{l^2}{1 + \exp \left( \alpha \cdot (\gamma - l) \right)}
    gt_loss = th.tensor(0.3565)
    loss = loss_func(logits, labels)
    assert_almost_equal(loss.numpy(), gt_loss.numpy(), decimal=4)

    alpha = 2
    gamma = 1.5
    loss_func = ShrinkageLossFunc(alpha, gamma)
    logits = th.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    labels = th.tensor([1., 1., 1., 1., 1., 1., 1., 1., 0, 0.9])
    # Loss is computed following:
    # l = abs(logits - labels)
    # loss = \frac{l^2}{1 + \exp \left( \alpha \cdot (\gamma - l) \right)}
    gt_loss = th.tensor(0.0692)
    loss = loss_func(logits, labels)
    assert_almost_equal(loss.numpy(), gt_loss.numpy(), decimal=4)
