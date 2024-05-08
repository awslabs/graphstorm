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

    Generate example graph data using built-in datasets for node classifcation,
    node regression, edge classification and edge regression.

    Demonstration models for using GraphStorm APIs
"""


from graphstorm.model import (GSgnnNodeModel,
                              GSNodeEncoderInputLayer,
                              RelationalGCNEncoder,
                              EntityClassifier,
                              ClassifyLossFunc)


class RgcnNCModel(GSgnnNodeModel):
    """ TODO add descriptions
    """
    def __init__(self,
                 g,
                 num_hid_layers,
                 feat_size,
                 hid_size,
                 num_classes,
                 multilabel=False):
        super(RgcnNCModel, self).__init__(alpha_l2norm=0.)

        # set an input layer encoder
        encoder = GSNodeEncoderInputLayer(g=g, feat_size=feat_size, embed_size=hid_size)
        self.set_node_input_encoder(encoder)

        # set a GNN encoder
        gnn_encoder = RelationalGCNEncoder(g=g,
                                           h_dim=hid_size,
                                           out_dim=hid_size,
                                           num_hidden_layers=num_hid_layers-1)
        self.set_gnn_encoder(gnn_encoder)

        # set a decoder specific to node-classification task
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
