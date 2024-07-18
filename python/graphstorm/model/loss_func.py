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
import torch as th
from torch import nn
import torch.nn.functional as F

from .gs_layer import GSLayer

class ClassifyLossFunc(GSLayer):
    """ Loss function for classification.

    Parameters
    ----------
    multilabel : bool
        Whether this is multi-label classification.
    multilabel_weights : Tensor
        The label weights for multi-label classifciation.
    imbalance_class_weights : Tensor
        The class weights for imbalanced classes.
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
        """
        if self.multilabel:
            # BCEWithLogitsLoss wants labels be th.Float
            return self.loss_fn(logits, labels.type(th.float32))
        else:
            return self.loss_fn(logits, labels.long())

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
    """ Loss function for regression
    """
    def __init__(self):
        super(RegressionLossFunc, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, logits, labels):
        """ The forward function.
        """
        # Make sure the lable is a float tensor
        return self.loss_fn(logits, labels.float())

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
    """ Loss function for link prediction.
    """

    def forward(self, pos_score, neg_score):
        """ The forward function.

            Parameters
            ----------
            pos_score: dict of Tensor
                The scores for positive edges of each edge type.
            neg_score: dict of Tensor
                The scores for negative edges of each edge type.
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
        return F.binary_cross_entropy_with_logits(score, label)

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
    """ Loss function for link prediction.
    """

    def forward(self, pos_score, neg_score):
        """ The forward function.

            Parameters
            ----------
            pos_score: dict of tuple of Tensor
                The (scores, edge weight) for positive edges of each edge type.
            neg_score: dict of tuple of Tensor
                The (scores, edge weight) for negative edges of each edge type.
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
        return F.binary_cross_entropy_with_logits(score, label, weight=weight)

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

class LinkPredictContrastiveLossFunc(GSLayer):
    r""" Contrastive Loss function for link prediction.

        The positive and negative scores are computed through a
        score function as:

            score = f(<src, rel, dst>)

        And we treat a score as the distance between `src` and
        `dst` nodes under relation `rel`.

        In contrastive loss, we assume one positive pair <src, dst>
        has K corresponding negative pairs <src, neg_dst1>,
        <src, neg_dst2> .... <src, neg_dstk> When we compute the
        loss of <src, dst>, we follow the following equation:

            .. math::
            loss = -log(exp(pos\_score)/\sum_{i=0}^N exp(score_i))

        where score includes both positive score of <src, dst> and
        negative scores of <src, neg_dst0>, ... <src, neg_dstk>

        Parameters
        ----------
        temp: float
            Temperature value
    """
    def __init__(self, temp=1.0):
        super(LinkPredictContrastiveLossFunc, self).__init__()
        self._temp = temp

    def forward(self, pos_score, neg_score):
        """ The forward function.

            Parameters
            ----------
            pos_score: dict of Tensor
                The scores for positive edges of each edge type.
            neg_score: dict of Tensor
                The scores for negative edges of each edge type.
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

        return -log_prob.mean()

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
