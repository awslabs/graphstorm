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

class LinkPredictLossFunc(GSLayer):
    """ Loss function for link prediction.
    """

    def forward(self, pos_score, neg_score):
        """ The forward function.
        """
        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)])
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

class WeightedLinkPredictLossFunc(GSLayer):
    """ Loss function for link prediction.
    """

    def forward(self, pos_score, neg_score):
        """ The forward function.
        """
        assert len(pos_score) == 2, \
            "Pos score must include score and weight " \
            "Please use LinkPredictWeightedDistMultDecoder or " \
            "LinkPredictWeightedDotDecoder"
        pos_score, pos_weight = pos_score
        neg_score, _ = neg_score # neg_weight is always all 1
        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)])
        weight = th.cat([pos_weight, th.ones_like(neg_score)])
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
    """ Contrastive Loss function for link prediction.

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
            pos_score: dict of tensors
                A dictionary of etype -> pos scores.
            neg_score: dict of tensors
                A dictionary of etype -> neg scores.
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
