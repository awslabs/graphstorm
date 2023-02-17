"""Loss functions.
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
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=multilabel_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=imbalance_class_weights)

    def forward(self, logits, labels):
        """ The forward function.
        """
        if self.multilabel:
            # BCEWithLogitsLoss wants labels be th.Float
            return self.loss_fn(logits, labels.type(th.float32))
        else:
            return self.loss_fn(logits, labels)

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
