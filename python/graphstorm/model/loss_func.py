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

    Loss functions.
"""
import logging

import torch as th
from torch import nn
import torch.nn.functional as F

from .gs_layer import GSLayer
from .utils import get_rank

class ClassifyLossFunc(GSLayer):
    """ Loss function for classification tasks.

    If multilabel is set to True, the ``torch.nn.BCEWithLogitsLoss``
    is used, otherwise the ``torch.nn.CrossEntropyLoss`` is used.

    Parameters
    ----------
    multilabel : bool
        Whether this is multi-label classification.
    multilabel_weights : Tensor
        The label weights for multi-label classifciation.
        Default: None
    imbalance_class_weights : Tensor
        The class weights for imbalanced classes.
        Default: None
    """
    def __init__(self, multilabel, multilabel_weights=None, imbalance_class_weights=None):
        super(ClassifyLossFunc, self).__init__()
        self.multilabel = multilabel
        if multilabel:
            self.loss_fn = nn.BCEWithLogitsLoss(weight=imbalance_class_weights,
                                                pos_weight=multilabel_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=imbalance_class_weights)

    def forward(self, logits, labels):
        """ The forward function.

        Parameters
        ----------
        logits: torch.Tensor
            The prediction results.
        labels: torch.Tensor
            The training labels.

        Returns
        -------
        loss: Tensor
            The loss value.
        """
        if self.multilabel:
            # BCEWithLogitsLoss wants labels be th.Float
            loss = self.loss_fn(logits, labels.type(th.float32))
            return loss
        else:
            loss = self.loss_fn(logits, labels.long())
            return loss

    @property
    def in_dims(self):
        """ The number of input dimensions.

        Returns
        -------
        int : the number of input dimensions.
        """
        return None

    @property
    def out_dims(self):
        """ The number of output dimensions.

        Returns
        -------
        int : the number of output dimensions.
        """
        return None

class FocalLossFunc(GSLayer):
    """ Focal loss function for classification tasks.

    Copy from torchvision.ops.sigmoid_focal_loss.
    Only with mean reduction.
    See more details on https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html.

    To use focal loss, the classification task must be
    a binary classification task.

    Parameters
    ----------
    alpha: float
        Weighting factor in range (0,1) to balance
        positive vs negative examples. Use ``-1`` to ignore this parameter.
        Default: ``0.25``.
    gamma: float
        Exponent of the modulating factor (1 - p_t) to
        balance easy vs hard examples.
        Default: ``2``.

    .. versionadded:: 0.4.0
        The :py:class:`FocalLossFunc`.
    """
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLossFunc, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        """ The forward function.

        Parameters
        ----------
        logits: torch.Tensor
            The prediction results.
        labels: torch.Tensor
            The training labels.

        Returns
        -------
        loss: Tensor
            The loss value.
        """
        # We need to reshape logits into a 1D float tensor
        # and cast labels into a float tensor.
        inputs = logits.squeeze()
        targets = labels.float()

        pred = th.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pred_t = pred * targets + (1 - pred) * (1 - targets)
        loss = ce_loss * ((1 - pred_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        loss = loss.mean()
        return loss

    @property
    def in_dims(self):
        """ The number of input dimensions.

        Returns
        -------
        int : the number of input dimensions.
        """
        return None

    @property
    def out_dims(self):
        """ The number of output dimensions.

        Returns
        -------
        int : the number of output dimensions.
        """
        return None

class RegressionLossFunc(GSLayer):
    """ Loss function for regression tasks.

    The ``torch.nn.MSELoss`` is used.
    """
    def __init__(self):
        super(RegressionLossFunc, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, logits, labels):
        """ The forward function.

        Parameters
        ----------
        logits: torch.Tensor
            The prediction results.
        labels: torch.Tensor
            The training labels.

        Returns
        -------
        loss: Tensor
            The loss value.
        """
        # Make sure the lable is a float tensor
        loss = self.loss_fn(logits, labels.float())
        return loss

    @property
    def in_dims(self):
        """ The number of input dimensions.

        Returns
        -------
        int : the number of input dimensions.
        """
        return None

    @property
    def out_dims(self):
        """ The number of output dimensions.

        Returns
        -------
        int : the number of output dimensions.
        """
        return None

class LinkPredictBCELossFunc(GSLayer):
    r""" Loss function for link prediction tasks using binary
    cross entropy loss.

    The ``torch.nn.functional.binary_cross_entropy_with_logits`` is used
    to compute the loss. The loss function is defined as:

    .. math::

        loss = - y \cdot \log score + (1 - y) \cdot \log (1 - score)

    where ``y`` is 1 for a positive edge and 0 for a negative edge.
    ``score`` is the score value of the edges computed by the score function.
    """

    def forward(self, pos_score, neg_score):
        """ The forward function.

        Parameters
        ----------
        pos_score: dict of Tensor
            The scores for positive edges of each edge type.
        neg_score: dict of Tensor
            The scores for negative edges of each edge type.

        Returns
        -------
        loss: Tensor
            The loss value.
        """
        p_score = []
        n_score = []
        for key, p_s in pos_score.items():
            n_s = neg_score[key]
            p_score.append(p_s)
            n_score.append(n_s)

        p_score = th.cat(p_score)
        n_score = th.cat(n_score)
        score = th.cat([p_score, n_score])
        label = th.cat([th.ones_like(p_score), th.zeros_like(n_score)])
        loss = F.binary_cross_entropy_with_logits(score, label)
        return loss

    @property
    def in_dims(self):
        """ The number of input dimensions.

        Returns
        -------
        int : the number of input dimensions.
        """
        return None

    @property
    def out_dims(self):
        """ The number of output dimensions.

        Returns
        -------
        int : the number of output dimensions.
        """
        return None

class WeightedLinkPredictBCELossFunc(GSLayer):
    r""" Loss function for link prediction tasks using binary
    cross entropy loss with weights.

    The ``torch.nn.functional.binary_cross_entropy_with_logits`` is used
    to compute the loss. The loss function is defined as:

    .. math::

        loss = - w\_e [ y \cdot \log score + (1 - y) \cdot \log (1 - score) ]

    where ``y`` is 1 for a positive edge and 0 for a negative edge.
    ``score`` is the score value of ``e`` computed by the score function,
    ``w_e`` is the weight of an edge ``e``, which is defined as:

    .. math::

        w\_e = \left \{
        \begin{array}{lc}
            1,  & \text{ if } e \in G, \\
            0,  & \text{ if } e \notin G
        \end{array}
        \right.

    where ``G`` is the training graph.
    """

    def forward(self, pos_score, neg_score):
        """ The forward function.

        Parameters
        ----------
        pos_score: dict of tuple of Tensor
            The (scores, edge weight) for positive edges of each edge type.
        neg_score: dict of tuple of Tensor
            The (scores, edge weight) for negative edges of each edge type.

        Returns
        -------
        loss: Tensor
            The loss value.
        """
        p_score = []
        p_weight = []
        n_score = []
        for key, p_s in pos_score.items():
            assert len(p_s) == 2, \
                "Pos score must include score and weight " \
                "Please use LinkPredictWeightedDistMultDecoder or " \
                "LinkPredictWeightedDotDecoder"
            n_s = neg_score[key]
            p_s, p_w = p_s
            n_s, _ = n_s # neg_weight is always all 1
            p_score.append(p_s)
            p_weight.append(p_w)
            n_score.append(n_s)
        p_score = th.cat(p_score)
        p_weight = th.cat(p_weight)
        n_score = th.cat(n_score)

        score = th.cat([p_score, n_score])
        label = th.cat([th.ones_like(p_score), th.zeros_like(n_score)])
        weight = th.cat([p_weight, th.ones_like(n_score)])
        loss = F.binary_cross_entropy_with_logits(score, label, weight=weight)
        return loss

    @property
    def in_dims(self):
        """ The number of input dimensions.

        Returns
        -------
        int : the number of input dimensions.
        """
        return None

    @property
    def out_dims(self):
        """ The number of output dimensions.

        Returns
        -------
        int : the number of output dimensions.
        """
        return None

class LinkPredictAdvBCELossFunc(LinkPredictBCELossFunc):
    r""" Binary cross entropy loss function for link prediction tasks
    with adversarial loss for negative samples.

    The loss of positive edges is defined as:

    .. math::

        loss_{pos} = - \log score

    where ``score`` is the score value of the positive edges computed by the score function.

    The loss of negative edges is defined as:

    .. math::

        loss_{neg} = \log (1 - score) \\
        loss_{neg} = \mathrm{softmax}(score * adversarial\_temperature) * loss_{neg}

    where ``score`` is the score value of the negative edges
    computed by the score function and ``adversarial_temperature``
    is a hyper-parameter.

    The final loss is defined as:

    .. math::

        loss = \dfrac{\mathrm{avg}(loss_{pos}) + \mathrm{avg}(loss_{neg})}{2}

    Parameters
    ----------
    adversarial_temperature: float
        Temperature value for adversarial loss.
    """
    def __init__(self, adversarial_temperature):
        super(LinkPredictAdvBCELossFunc, self).__init__()
        assert isinstance(adversarial_temperature, float), \
            "The adversarial_temperature must be a float, but we get " \
            f"{adversarial_temperature} with type {type(adversarial_temperature)}"
        self.adversarial_temperature = adversarial_temperature
        self._debug_print = True

    def _compute_loss(self, score, label):
        """ Compute BCE loss
        """
        loss = -(label * th.log(F.sigmoid(score)) +
            (1 - label) * th.log(1 - F.sigmoid(score)))

        # For debugging abnormal loss values.
        if self._debug_print and get_rank() == 0:
            logging.debug("Cross entropy loss for link prediction \
                          with input score as %s and loss as %s.",
                          score.detach(), loss.detach())
        return loss

    def _compute_adversarial_loss(self, score, label):
        loss = self._compute_loss(score, label)
        loss = th.softmax(score * self.adversarial_temperature,
                                 dim=-1).detach() * loss
        # For debugging abnormal loss values.
        if self._debug_print and get_rank() == 0:
            logging.debug("Adversarial cross entropy loss for negative samples \
                          with input score as %s and loss as %s. \
                          For abnormal loss values, it is suggested to enable \
                          lp embed normalizer by setting --lp-embed-normalizer l2_norm \
                          and tune the hyperparameters gamma or adversarial_temperature.",
                          score.detach(), loss.detach())
            # turn off the debug warning
            self._debug_print = False

        loss = th.sum(loss, dim=-1)
        return loss

    def forward(self, pos_score, neg_score):
        """ The forward function.

        Parameters
        ----------
        pos_score: dict of Tensor
            The scores for positive edges of each edge type.
        neg_score: dict of Tensor
            The scores for negative edges of each edge type.

        Returns
        -------
        loss: Tensor
            The loss value.
        """
        p_score = []
        n_score = []
        for key, p_s in pos_score.items():
            n_s = neg_score[key]
            p_score.append(p_s)
            n_score.append(n_s)

        p_score = th.cat(p_score)
        n_score = th.cat(n_score)
        p_labels = th.ones_like(p_score)
        n_labels = th.zeros_like(n_score)

        pos_loss = self._compute_loss(p_score, p_labels)
        # adversarial negative loss
        neg_loss = self._compute_adversarial_loss(n_score, n_labels)

        pos_loss = th.mean(pos_loss)
        neg_loss = th.mean(neg_loss)
        loss = (neg_loss + pos_loss) / 2

        return loss

class WeightedLinkPredictAdvBCELossFunc(LinkPredictAdvBCELossFunc):
    r""" Binary cross entropy loss function for link prediction tasks
    with adversarial loss for negative samples and weights on positive samples.

    The loss function of a positive edge is defined as:

    .. math::

        loss_{pos} = - w * \log score

    where ``score`` is the score value of the positive edges computed
    by the score function, and ``w`` is the weight of each positive edge.
    The loss of the negative edges is the same as **LinkPredictAdvBCELossFunc**.

    The final loss is defined as:

    .. math::

        loss = \dfrac{\mathrm{avg}(loss_{pos}) + \mathrm{avg}(loss_{neg})}{2}
    """

    def forward(self, pos_score, neg_score):
        """ The forward function.

        Parameters
        ----------
        pos_score: dict of tuple of Tensor
            The (scores, edge weight) for positive edges of each edge type.
        neg_score: dict of tuple of Tensor
            The (scores, edge weight) for negative edges of each edge type.

        Returns
        -------
        loss: Tensor
            The loss value.
        """
        p_score = []
        p_weight = []
        n_score = []
        for key, p_s in pos_score.items():
            assert len(p_s) == 2, \
                "Pos score must include score and weight " \
                "Please use LinkPredictWeightedDistMultDecoder or " \
                "LinkPredictWeightedDotDecoder"
            n_s = neg_score[key]
            p_s, p_w = p_s
            n_s, _ = n_s # neg_weight is always all 1
            p_score.append(p_s)
            p_weight.append(p_w)
            n_score.append(n_s)
        p_score = th.cat(p_score)
        p_weight = th.cat(p_weight)
        n_score = th.cat(n_score)

        p_labels = th.ones_like(p_score)
        n_labels = th.zeros_like(n_score)

        # positive loss multiply edge weight
        pos_loss = self._compute_loss(p_score, p_labels)
        pos_loss = pos_loss * p_weight
        # adversarial negative loss
        neg_loss = self._compute_adversarial_loss(n_score, n_labels)

        pos_loss = th.mean(pos_loss)
        neg_loss = th.mean(neg_loss)
        loss = (neg_loss + pos_loss) / 2
        return loss

class LinkPredictContrastiveLossFunc(GSLayer):
    r""" Contrastive Loss function for link prediction tasks.

    The positive and negative scores are computed through a
    score function as:

        .. math::

            score = f(<src, rel, dst>)

    And we treat a score as the distance between ``src`` and
    ``dst`` nodes under relation ``rel``.

    In contrastive loss, we assume one positive pair <src, dst>
    has K corresponding negative pairs <src, neg_dst1>,
    <src, neg_dst2> .... <src, neg_dstk>. When we compute the
    loss of <src, dst>, we follow the following equation:

        .. math::

            loss = -log(exp(pos\_score)/\sum_{i=0}^K exp(score_i))

    where score includes both positive score of <src, dst> and
    negative scores of <src, neg_dst0>, ... <src, neg_dstk>.

    Parameters
    ----------
    temp: float
        Temperature value.
    """
    def __init__(self, temp=1.0):
        super(LinkPredictContrastiveLossFunc, self).__init__()
        self._temp = temp
        self._debug_print = True

    def forward(self, pos_score, neg_score):
        """ The forward function.

        Parameters
        ----------
        pos_score: dict of Tensor
            The scores for positive edges of each edge type.
        neg_score: dict of Tensor
            The scores for negative edges of each edge type.

        Returns
        -------
        loss: Tensor
            The loss value.
        """
        pscore = []
        nscore = []
        for key, p_s in pos_score.items():
            assert key in neg_score, \
                f"Negative scores of {key} must exists"
            n_s = neg_score[key]

            # Both p_s and n_s are soreted according to source nid
            # (which are same in pos_graph and neg_graph)
            pscore.append(p_s)
            nscore.append(n_s.reshape(p_s.shape[0], -1))
        pscore = th.cat(pscore, dim=0)
        nscore = th.cat(nscore, dim=0)

        pscore = th.div(pscore, self._temp)
        nscore = th.div(nscore, self._temp)
        score = th.cat([pscore.unsqueeze(1), nscore], dim=1)

        exp_logits = th.exp(score)
        log_prob = pscore - th.log(exp_logits.sum(1))
        loss = -log_prob.mean()

        # For debugging abnormal loss values.
        if self._debug_print and get_rank() == 0:
            logging.debug("Contrastive loss with pos score as %s \
                          and neg score as %s and loss as %s.\
                          For abnormal loss values, it is suggested \
                          to tune the hyperparameter gamma or \
                          contrastive_loss_temperature.",
                          pscore.detach(), nscore.detach(), -log_prob.detach())
            self._debug_print = False

        return loss

    @property
    def in_dims(self):
        """ The number of input dimensions.

        Returns
        -------
        int : the number of input dimensions.
        """
        return None

    @property
    def out_dims(self):
        """ The number of output dimensions.

        Returns
        -------
        int : the number of output dimensions.
        """
        return None
