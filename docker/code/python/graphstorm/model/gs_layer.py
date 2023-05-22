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

    This defines the abstract class of a GraphStorm layer.
"""

import abc

import torch as th
from torch import nn

class GSLayerBase:
    """Abastract class of a GraphStorm layer.

    A GraphStorm GNN model splits a model into multiple components:
    * input encoder for node features,
    * input encoder for edge features,
    * GNN encoder,
    * decoder,
    * loss function.

    A user can can customize each of the components by implementing GSLayer or GSLayerNoParam.
    """

    @property
    @abc.abstractmethod
    def in_dims(self):
        """ The number of input dimensions.

        Returns
        -------
        int : the number of input dimensions.
        """

    @property
    @abc.abstractmethod
    def out_dims(self):
        """ The number of output dimensions.

        Returns
        -------
        int : the number of output dimensions.
        """

class GSLayer(GSLayerBase, nn.Module):    # pylint: disable=abstract-method
    """ The abstract class of a GraphStorm layer with model parameters

    If a GraphStorm layer has model parameters, it should inherit `GSLayer`.
    """

    @property
    def device(self):
        """ The device where the model runs.

        Here we assume all model parameters are on the same device.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            # If there are no parameters in the layer, we assume it's on CPU.
            return th.device("cpu")

class GSLayerNoParam(GSLayerBase):    # pylint: disable=abstract-method
    """ The abstract class of a GraphStorm layer without model parameters.

    If a GraphStorm layer does not have model parameters, it should inherit `GSLayerNoParam`.
    """

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):     # pylint: disable=unused-argument
        """ The forward function.
        """
