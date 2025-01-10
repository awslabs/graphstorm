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

    Node prediction decoders.
"""
import logging

import torch as th
from torch import nn

from .gs_layer import GSLayer

class EntityClassifier(GSLayer):
    """ Decoder for node classification tasks.

    Parameters
    ----------
    in_dim: int
        The input dimension size.
    num_classes: int
        The number of classes to predict.
    multilabel: bool
        Whether this is a multi-label classification decoder.
    dropout: float
        Dropout rate. Default: 0.
    norm: str
        Normalization methods. Not used, but reserved for complex node classifier
        implementation. Default: None.
    use_bias: bool
        Whether the node decoder uses a bias parameter. Default: True.

    .. versionchanged:: 0.4.0
        Add a new argument "use_bias" so users can control whether decoders have bias.
    """
    def __init__(self,
                 in_dim,
                 num_classes,
                 multilabel,
                 dropout=0,
                 norm=None,
                 use_bias=True):
        super(EntityClassifier, self).__init__()
        self._in_dim = in_dim
        self._num_classes = num_classes
        self._multilabel = multilabel
        self._dropout = dropout
        # TODO(xiangsx): The norm is not used here.
        self._norm = norm
        self._use_bias = use_bias

        self._init_model()

    def _init_model(self):
        """ Init decoder model
        """
        self.decoder = nn.Parameter(th.Tensor(self._in_dim, self._num_classes))
        nn.init.xavier_uniform_(self.decoder,
                                gain=nn.init.calculate_gain('relu'))
        if self._use_bias:
            self.bias = nn.Parameter(th.zeros(self._num_classes))
        # TODO(zhengda): The dropout is not used here.
        # This is the last layer of the model so we risk dropping gradients if we set things to 0.
        self.dropout = nn.Dropout(self._dropout)
        if self._norm is not None:
            logging.warning("Embedding normalization (batch norm or layer norm) "
                            "is not supported in EntityClassifier")

    def forward(self, inputs):
        ''' Node classification decoder forward computation.

        Parameters
        ----------
        inputs: Tensor
            The input embeddings.

        Returns
        -------
        out: Tensor of the prediction logits.
        '''
        out = th.matmul(inputs, self.decoder)
        if self._use_bias:
            out = out + self.bias

        return out

    def predict(self, inputs):
        """ Node classification prediction computation.

        Parameters
        ----------
        inputs: Tensor
            The input embeddings.

        Returns
        --------
        out: Tensor of argmax of the prediction results, or the maximum of the prediction results
        if ``multilabel`` is ``True``.
        """
        logits = th.matmul(inputs, self.decoder)
        if self._use_bias:
            logits = logits + self.bias
        if self._multilabel:
            out = (th.sigmoid(logits) > .5).long()
        else:
            if self.out_dims == 1:
                out = (th.sigmoid(logits) > .5).long()
            else:
                out = logits.argmax(dim=1)
        return out

    def predict_proba(self, inputs):
        """ Node classification prediction computation and return normalized prediction
        results.

        Parameters
        ----------
        inputs: Tensor
            The input embeddings.

        Returns
        -------
        out: Tensor of normalized prediction results.
        """
        logits = th.matmul(inputs, self.decoder)
        if self._use_bias:
            logits = logits + self.bias

        if self._multilabel:
            out = th.sigmoid(logits)
        else:
            if self.out_dims == 1:
                out = th.sigmoid(logits)
            else:
                out = th.softmax(logits, 1)
        return out

    @property
    def in_dims(self):
        """ Return the input dimension size, which is given in class initialization.
        """
        return self._in_dim

    @property
    def out_dims(self):
        """ Return the output dimensions size, which is given in class initialization.
        """
        return self._num_classes

class EntityRegression(GSLayer):
    """ Decoder for node regression tasks.

    Parameters
    ----------
    h_dim: int
        The input dimension size.
    dropout: float
        Dropout rate. Default: 0.
    out_dim: int
        The output dimension size. Default: 1 for regression tasks.
    norm: str, optional
        Normalization methods. Not used, but reserved for complex node regression
        implementation. Default: None.
    use_bias: bool
        Whether the node decoder uses a bias parameter. Default: True.

    .. versionchanged:: 0.4.0
        Add a new argument "use_bias" so users can control whether decoders have bias.
    """
    def __init__(self,
                 h_dim,
                 dropout=0,
                 out_dim=1,
                 norm=None,
                 use_bias=True):
        super(EntityRegression, self).__init__()
        self._h_dim = h_dim
        self._out_dim = out_dim
        self._dropout = dropout
        # TODO(xiangsx): The norm is not used here.
        self._norm = norm
        self._use_bias = use_bias

        self._init_model()

    def _init_model(self):
        self.decoder = nn.Parameter(th.Tensor(self._h_dim, self._out_dim))
        nn.init.xavier_uniform_(self.decoder)
        if self._use_bias:
            self.bias = nn.Parameter(th.zeros(self._out_dim))
        # TODO(zhengda): The dropout is not used.
        # This is the last layer of the model so we risk dropping gradients if we set things to 0.
        self.dropout = nn.Dropout(self._dropout)

        if self._norm is not None:
            logging.warning("Embedding normalization (batch norm or layer norm) "
                            "is not supported in EntityRegression")

    def forward(self, inputs):
        """ Node regression decoder forward computation.

        Parameters
        ----------
        inputs: Tensor
            The input embeddings.

        Returns
        -------
        out: Tensor of the prediction results.
        """
        out = th.matmul(inputs, self.decoder)
        if self._use_bias:
            out = out + self.bias

        return out

    def predict(self, inputs):
        """ Node regression prediction computation.

        Parameters
        ----------
        inputs: Tensor
            The input embeddings.

        Returns
        -------
        out: Tensor of the prediction results.
        """
        out = th.matmul(inputs, self.decoder)
        if self._use_bias:
            out = out + self.bias

        return out

    def predict_proba(self, inputs):
        """ For node regression task, it returns the same results as the
        ``predict()`` function.

        Parameters
        ----------
        inputs: Tensor
            The input embeddings.

        Returns
        -------
        out: Tensor of the prediction results.
        """
        out = self.predict(inputs)

        return out

    @property
    def in_dims(self):
        """ Return the input dimension size, which is given in class initialization.
        """
        return self._h_dim

    @property
    def out_dims(self):
        """ Return the output dimension size, which should be ``1`` for regression tasks.
        """
        return self._out_dim
