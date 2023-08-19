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
import torch as th
from torch import nn

from .gs_layer import GSLayer

class EntityClassifier(GSLayer):
    ''' Classifier for node entities.

    Parameters
    ----------
    in_dim : int
        The input dimension
    num_classes : int
        The number of classes to predict
    multilabel : bool
        Whether this is multi-label classification.
    dropout : float
        The dropout
    '''
    def __init__(self,
                 in_dim,
                 num_classes,
                 multilabel,
                 dropout=0):
        super(EntityClassifier, self).__init__()
        self._in_dim = in_dim
        self._num_classes = num_classes
        self._multilabel = multilabel
        self.decoder = nn.Parameter(th.Tensor(in_dim, num_classes))
        nn.init.xavier_uniform_(self.decoder,
                                gain=nn.init.calculate_gain('relu'))
        # TODO(zhengda): The dropout is not used here.
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        ''' The forward function.

        Parameters
        ----------
        inputs : tensor
            The input features

        Returns
        -------
        Tensor : the logits
        '''
        return th.matmul(inputs, self.decoder)

    def predict(self, inputs):
        """ Make prediction on input data.

        Parameters
        ----------
        inputs : tensor
            The input features

        Returns
        -------
        Tensor : maximum of the predicted results
        """
        logits = th.matmul(inputs, self.decoder)
        return (th.sigmoid(logits) > .5).long() if self._multilabel else logits.argmax(dim=1)

    def predict_proba(self, inputs):
        """ Make prediction on input data.

        Parameters
        ----------
        inputs : tensor
            The input features

        Returns
        -------
        Tensor : all normalized predicted results
        """
        logits = th.matmul(inputs, self.decoder)
        return th.sigmoid(logits) if self._multilabel else th.softmax(logits, 1)

    @property
    def in_dims(self):
        """ The number of input dimensions.
        """
        return self._in_dim

    @property
    def out_dims(self):
        """ The number of output dimensions.
        """
        return self._num_classes

class EntityRegression(GSLayer):
    ''' Regression on entity nodes

    Parameters
    ----------
    h_dim : int
        The hidden dimensions
    dropout : float
        The dropout
    '''
    def __init__(self,
                 h_dim,
                 dropout=0):
        super(EntityRegression, self).__init__()
        self.h_dim = h_dim
        self.decoder = nn.Parameter(th.Tensor(h_dim, 1))
        nn.init.xavier_uniform_(self.decoder)
        # TODO(zhengda): The dropout is not used.
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        ''' The forward function.
        '''
        return th.matmul(inputs, self.decoder)

    def predict(self, inputs):
        """ The prediction function.

        Parameters
        ----------
        inputs : tensor
            The input features

        Returns
        -------
        Tensor : the predicted results
        """
        return th.matmul(inputs, self.decoder)

    def predict_proba(self, inputs):
        """ Make prediction on input data.
            For regression task, it is same as predict

        Parameters
        ----------
        inputs : tensor
            The input features

        Returns
        -------
        Tensor : all normalized predicted results
        """
        return self.predict(inputs)

    @property
    def in_dims(self):
        """ The number of input dimensions.
        """
        return self.h_dim

    @property
    def out_dims(self):
        """ The number of output dimensions.
        """
        return 1
