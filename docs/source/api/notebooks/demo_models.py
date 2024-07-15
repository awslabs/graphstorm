"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Demonstration models for using GraphStorm APIs
"""


import graphstorm as gs
from graphstorm.model import (GSgnnNodeModel,
                              GSNodeEncoderInputLayer,
                              RelationalGCNEncoder,
                              RelationalGATEncoder,
                              HGTEncoder,
                              EntityClassifier,
                              ClassifyLossFunc,
                              GSgnnLinkPredictionModel,
                              LinkPredictDotDecoder,
                              LinkPredictDistMultDecoder,
                              LinkPredictBCELossFunc)

class RgcnNCModel(GSgnnNodeModel):
    """ A simple RGCN model for node classification using Graphstorm APIs

    This RGCN model extends GraphStorm's GSgnnNodeModel, and it has the standard GraphStorm
    model architecture:
    1. an input layer that converts input node features to the embeddings with hidden dimensions
    2. a GNN encoder layer that performs the message passing work
    3. a decoder layer, i.e., EntityClassifier, that transfors node representations into logits
    for classification, and
    4. a loss function that matches classification tasks.

    Then the model also initialize its own optimizer object.

    Arguments
    ----------
    g: DistGraph
        A DGL DistGraph.
    num_hid_layers: int
        The number of gnn layers.
    node_feat_field: dict of list of strings
        The list features for each node type to be used in the model.
    hid_size: int
        The dimension of hidden layers.
    num_classes: int
        The target number of classes for classification.
    multilabel: boolean
        Indicator of if this is a multilabel task.
    """
    def __init__(self,
                 g,
                 num_hid_layers,
                 node_feat_field,
                 hid_size,
                 num_classes,
                 multilabel=False):
        super(RgcnNCModel, self).__init__(alpha_l2norm=0.)

        # extract feature size
        feat_size = gs.get_node_feat_size(g, node_feat_field)

        # set an input layer encoder
        encoder = GSNodeEncoderInputLayer(g=g, feat_size=feat_size, embed_size=hid_size)
        self.set_node_input_encoder(encoder)

        # set a GNN encoder
        gnn_encoder = RelationalGCNEncoder(g=g,
                                           h_dim=hid_size,
                                           out_dim=hid_size,
                                           num_hidden_layers=num_hid_layers-1)
        self.set_gnn_encoder(gnn_encoder)

        # set a decoder specific to node classification task
        decoder = EntityClassifier(in_dim=hid_size,
                                   num_classes=num_classes,
                                   multilabel=multilabel)
        self.set_decoder(decoder)

        # classification loss function
        self.set_loss_func(ClassifyLossFunc(multilabel=multilabel))

        # initialize model's optimizer
        self.init_optimizer(lr=0.001,
                            sparse_optimizer_lr=0.01,
                            weight_decay=0)


class RgatNCModel(GSgnnNodeModel):
    """ A simple RGAT model for node classification using Graphstorm APIs

    This RGAT model extends GraphStorm's GSgnnNodeModel, and it has the standard GraphStorm
    model architecture:
    1. an input layer that converts input node features to the embeddings with hidden dimensions
    2. a GNN encoder layer that performs the message passing work
    3. a decoder layer, i.e., EntityClassifier, that transfors node representations into logits
    for classification, and
    4. a loss function that matches classification tasks.

    Then the model also initialize its own optimizer object.

    Arguments
    ----------
    g: DistGraph
        A DGL DistGraph.
    num_heads: int
        The number of attention heads.
    num_hid_layers: int
        The number of gnn layers.
    node_feat_field: dict of list of strings
        The list features for each node type to be used in the model.
    hid_size: int
        The dimension of hidden layers.
    num_classes: int
        The target number of classes for classification.
    multilabel: boolean
        Indicator of if this is a multilabel task.
    """
    def __init__(self,
                 g,
                 num_heads,
                 num_hid_layers,
                 node_feat_field,
                 hid_size,
                 num_classes,
                 multilabel=False):
        super(RgatNCModel, self).__init__(alpha_l2norm=0.)

        # extract feature size
        feat_size = gs.get_node_feat_size(g, node_feat_field)

        # set an input layer encoder
        encoder = GSNodeEncoderInputLayer(g=g, feat_size=feat_size, embed_size=hid_size)
        self.set_node_input_encoder(encoder)

        # set a GNN encoder
        gnn_encoder = RelationalGATEncoder(g=g,
                                           num_heads=num_heads,
                                           h_dim=hid_size,
                                           out_dim=hid_size,
                                           num_hidden_layers=num_hid_layers-1)
        self.set_gnn_encoder(gnn_encoder)

        # set a decoder specific to node classification task
        decoder = EntityClassifier(in_dim=hid_size,
                                   num_classes=num_classes,
                                   multilabel=multilabel)
        self.set_decoder(decoder)

        # classification loss function
        self.set_loss_func(ClassifyLossFunc(multilabel=multilabel))

        # initialize model's optimizer
        self.init_optimizer(lr=0.001,
                            sparse_optimizer_lr=0.01,
                            weight_decay=0)


class RgcnLPModel(GSgnnLinkPredictionModel):
    """ A simple RGCN model for link prediction using Graphstorm APIs

    This RGCN model extends GraphStorm's GSgnnLinkPredictionModel, and it has the similar
    model architecture as the node model, but has a different decoder layer and loss function:
    1. an input layer that converts input node features to the embeddings with hidden dimensions
    2. a GNN encoder layer that performs the message passing work
    3. a decoder layer that transfors edge representations into logits for link prediction, and
    4. a loss function that matches to link prediction tasks.

    The model also initialize its own optimizer object.

    Arguments
    ----------
    g: DistGraph
        A DGL DistGraph.
    num_hid_layers: int
        The number of gnn layers.
    node_feat_field: dict of list of strings
        The list features for each node type to be used in the model.
    hid_size: int
        The dimension of hidden layers.
    """
    def __init__(self,
                 g,
                 num_hid_layers,
                 node_feat_field,
                 hid_size):
        super(RgcnLPModel, self).__init__(alpha_l2norm=0.)

        # extract feature size
        feat_size = gs.get_node_feat_size(g, node_feat_field)

        # set an input layer encoder
        encoder = GSNodeEncoderInputLayer(g=g, feat_size=feat_size, embed_size=hid_size)
        self.set_node_input_encoder(encoder)

        # set a GNN encoder
        gnn_encoder = RelationalGCNEncoder(g=g,
                                           h_dim=hid_size,
                                           out_dim=hid_size,
                                           num_hidden_layers=num_hid_layers-1)
        self.set_gnn_encoder(gnn_encoder)

        # set a decoder specific to link prediction task
        decoder = LinkPredictDotDecoder(hid_size)
        self.set_decoder(decoder)

        # link prediction loss function
        self.set_loss_func(LinkPredictBCELossFunc())

        # initialize model's optimizer
        self.init_optimizer(lr=0.001,
                            sparse_optimizer_lr=0.01,
                            weight_decay=0)


class HgtLPModel(GSgnnLinkPredictionModel):
    """ A simple HGT model for link prediction using Graphstorm APIs

    This HGT model extends GraphStorm's GSgnnLinkPredictionModel, and it has the similar
    model architecture as the node model, but has a different decoder layer and loss function:
    1. an input layer that converts input node features to the embeddings with hidden dimensions
    2. a GNN encoder layer that performs the message passing work
    3. a decoder layer that transfors edge representations into logits for link prediction, and
    4. a loss function that matches to link prediction tasks.

    The model also initialize its own optimizer object.

    Arguments
    ----------
    g: DistGraph
        A DGL DistGraph.
    num_heads: int
        The number of attention heads.
    num_hid_layers: int
        The number of gnn layers.
    node_feat_field: dict of list of strings
        The list features for each node type to be used in the model.
    hid_size: int
        The dimension of hidden layers.
    """
    def __init__(self,
                 g,
                 num_heads,
                 num_hid_layers,
                 node_feat_field,
                 hid_size):
        super(HgtLPModel, self).__init__(alpha_l2norm=0.)

        # extract feature size
        feat_size = gs.get_node_feat_size(g, node_feat_field)

        # set an input layer encoder
        encoder = GSNodeEncoderInputLayer(g=g, feat_size=feat_size, embed_size=hid_size)
        self.set_node_input_encoder(encoder)

        # set a GNN encoder
        gnn_encoder = HGTEncoder(g=g,
                                 num_heads=num_heads,
                                 hid_dim=hid_size,
                                 out_dim=hid_size,
                                 num_hidden_layers=num_hid_layers-1)
        self.set_gnn_encoder(gnn_encoder)

        # set a decoder specific to link prediction task
        decoder = LinkPredictDistMultDecoder(etypes=g.canonical_etypes,
                                             h_dim=hid_size)
        self.set_decoder(decoder)

        # link prediction loss function
        self.set_loss_func(LinkPredictBCELossFunc())

        # initialize model's optimizer
        self.init_optimizer(lr=0.001,
                            sparse_optimizer_lr=0.01,
                            weight_decay=0)
