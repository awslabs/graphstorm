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


import argparse
import graphstorm as gs
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.model.gnn_encoder_base import GraphConvEncoder
from graphstorm.model import (GSgnnNodeModel,
                              GSNodeEncoderInputLayer,
                              RelationalGATEncoder,
                              EntityClassifier,
                              ClassifyLossFunc)


class Ara_GatLayer(nn.Module):
    """ One layer of ARA_GAT
    """
    def __init__(self, in_dim, out_dim, num_heads, rel_names, bias=True,
                 activation=None, self_loop=False, dropout=0.0, norm=None):
        super(Ara_GatLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.leaky_relu = nn.LeakyReLU(0.2)

        # GAT module for each relation type
        self.rel_gats = nn.ModuleDict()
        for rel in rel_names:
            self.rel_gats[str(rel)] = dgl.nn.GATConv(in_dim, out_dim//num_heads,    # should be divible
                                                     num_heads, allow_zero_in_degree=True)

        # across-relation attention weight set
        self.acr_attn_weights = nn.Parameter(th.Tensor(out_dim, 1))
        nn.init.normal_(self.acr_attn_weights)

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_dim))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_dim, out_dim))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        # dropout
        self.dropout = nn.Dropout(dropout)

        # normalization for each node type
        ntypes = set()
        for rel in rel_names:
            ntypes.add(rel[0])
            ntypes.add(rel[2])

        if norm == "batch":
            self.norm = nn.ParameterDict({ntype:nn.BatchNorm1d(out_dim) for ntype in ntypes})
        elif norm == "layer":
            self.norm = nn.ParameterDict({ntype:nn.LayerNorm(out_dim) for ntype in ntypes})
        else:
            self.norm = None

    def forward(self, g, inputs):
        """
        g: DGL.block
            A DGL block
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()

        # loop each edge type to fulfill GAT computation within each edge type
        for src_type, e_type, dst_type in g.canonical_etypes:

            # extract subgraph of each edge type
            sub_graph = g[src_type, e_type, dst_type]

            # check if no edges exist for this edge type
            if sub_graph.num_edges() == 0:
                continue

            # extract source and destination node features
            src_feat = inputs[src_type]
            dst_feat = inputs[dst_type][ :sub_graph.num_dst_nodes()]

            # GAT in one relation type
            agg = self.rel_gats[str((src_type, e_type, dst_type))](sub_graph, (src_feat, dst_feat))
            agg = agg.view(agg.shape[0], -1)

            # store aggregations in destination nodes
            sub_graph.dstdata['agg_' + str((src_type, e_type, dst_type))] = self.leaky_relu(agg)

        h = {}
        for n_type in g.dsttypes:
            if g.num_dst_nodes(n_type) == 0:
                continue

            # cross relation attention enhancement as outputs
            agg_list = []
            for k, v in g.dstnodes[n_type].data.items():
                if k.startswith('agg_'):
                    agg_list.append(v)

            # cross-relation attention
            if agg_list:
                acr_agg = th.stack(agg_list, dim=1)

                acr_att = th.matmul(acr_agg, self.acr_attn_weights)
                acr_sfm = th.softmax(acr_att, dim=1)

                # cross-relation weighted aggregation
                acr_sum = (acr_agg * acr_sfm).sum(dim=1)
            elif not self.self_loop:
                raise ValueError(f'Some nodes in the {n_type} type have no in-degree.' + \
                                 'Please check the data or set \"self_loop=True\"')

            # process new features
            if self.self_loop:
                if agg_list:
                    h_n = acr_sum + th.matmul(inputs[n_type][ :g.num_dst_nodes(n_type)], self.loop_weight)
                else:
                    h_n = th.matmul(inputs[n_type][ :g.num_dst_nodes(n_type)], self.loop_weight)
            if self.bias:
                h_n = h_n + self.h_bias
            if self.activation:
                h_n = self.activation(h_n)
            if self.norm:
                h_n = self.norm[n_type](h_n)
            h_n = self.dropout(h_n)

            h[n_type] = h_n

        return h


class Ara_GatEncoder(GraphConvEncoder):
    """ Across Relation Attention GAT Encoder by extending Graphstorm APIs
    """
    def __init__(self, g, h_dim, out_dim, num_heads, num_hidden_layers=1,
                 dropout=0, use_self_loop=True, norm='batch'):
        super(Ara_GatEncoder, self).__init__(h_dim, out_dim, num_hidden_layers)

        # h2h
        for _ in range(num_hidden_layers):
            self.layers.append(Ara_GatLayer(h_dim, h_dim, num_heads, g.canonical_etypes,
                                            activation=F.relu, self_loop=use_self_loop, dropout=dropout, norm=norm))
        # h2o
        self.layers.append(Ara_GatLayer(h_dim, out_dim, num_heads, g.canonical_etypes,
                                        activation=F.relu, self_loop=use_self_loop, norm=norm))

    def forward(self, blocks, h):
        """ accept block list and feature dictionary as inputs
        """
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h


class RgatNCModel(GSgnnNodeModel):
    """ A customized RGAT model for node classification using Graphstorm APIs
    """
    def __init__(self, g, num_heads, num_hid_layers, node_feat_field, hid_size, num_classes, multilabel=False,
                 encoder_type='ara'    # option for different rgat encoders
                ):
        super(RgatNCModel, self).__init__(alpha_l2norm=0.)

        # extract feature size
        feat_size = gs.get_node_feat_size(g, node_feat_field)

        # set an input layer encoder
        encoder = GSNodeEncoderInputLayer(g=g, feat_size=feat_size, embed_size=hid_size)
        self.set_node_input_encoder(encoder)

        # set the option of using either customized RGAT or built-in RGAT encoder
        if encoder_type == 'ara':
            gnn_encoder = Ara_GatEncoder(g, hid_size, hid_size, num_heads,
                                         num_hidden_layers=num_hid_layers-1)
        elif encoder_type == 'rgat':
            gnn_encoder = RelationalGATEncoder(g, hid_size, hid_size, num_heads,
                                               num_hidden_layers=num_hid_layers-1)
        else:
            raise Exception(f'Not supported encoders \"{encoder_type}\".')
        self.set_gnn_encoder(gnn_encoder)

        # set a decoder specific to node classification task
        decoder = EntityClassifier(in_dim=hid_size, num_classes=num_classes, multilabel=multilabel)
        self.set_decoder(decoder)

        # classification loss function
        self.set_loss_func(ClassifyLossFunc(multilabel=multilabel))

        # initialize model's optimizer
        self.init_optimizer(lr=0.001, sparse_optimizer_lr=0.01, weight_decay=0)


def fit(gs_args, cust_args):
    # Utilize GraphStorm's GSConfig class to accept arguments
    config = GSConfig(gs_args)

    gs.initialize(ip_config=config.ip_config, backend=config.backend, local_rank=config.local_rank)
    acm_data = gs.dataloading.GSgnnData(part_config=config.part_config)

    train_dataloader = gs.dataloading.GSgnnNodeDataLoader(
        dataset=acm_data,
        target_idx=acm_data.get_node_train_set(ntypes=config.target_ntype),
        node_feats=config.node_feat_name,
        label_field=config.label_field,
        fanout=config.fanout,
        batch_size=config.batch_size,
        train_task=True)
    val_dataloader = gs.dataloading.GSgnnNodeDataLoader(
        dataset=acm_data,
        target_idx=acm_data.get_node_val_set(ntypes=config.target_ntype),
        node_feats=config.node_feat_name,
        label_field=config.label_field,
        fanout=config.eval_fanout,
        batch_size=config.eval_batch_size,
        train_task=False)
    test_dataloader = gs.dataloading.GSgnnNodeDataLoader(
        dataset=acm_data,
        target_idx=acm_data.get_node_test_set(ntypes=config.target_ntype),
        node_feats=config.node_feat_name,
        label_field=config.label_field,
        fanout=config.eval_fanout,
        batch_size=config.eval_batch_size,
        train_task=False)

    model = RgatNCModel(g=acm_data.g,
                        num_heads=config.num_heads, 
                        num_hid_layers=config.num_layers,
                        node_feat_field=config.node_feat_name,
                        hid_size=config.hidden_size,
                        num_classes=config.num_classes,
                        encoder_type=cust_args.rgat_encoder_type)   # here use the customized argument instead of GSConfig

    evaluator = gs.eval.GSgnnClassificationEvaluator(eval_frequency=config.eval_frequency)

    trainer = gs.trainer.GSgnnNodePredictionTrainer(model)
    trainer.setup_evaluator(evaluator)
    trainer.setup_device(gs.utils.get_device())

    trainer.fit(train_loader=train_dataloader,
                val_loader=val_dataloader,
                test_loader=test_dataloader,
                num_epochs=config.num_epochs,
                save_model_path=config.save_model_path)


if __name__ == '__main__':
    # Leverage GraphStorm's argument parser to accept configuratioin yaml file
    arg_parser = get_argument_parser()

    # parse all arguments and split GraphStorm's built-in arguments from the customized ones
    gs_args, unknown_args = arg_parser.parse_known_args()
    print(f'GS arguments: {gs_args}')

    # create a new argument parser dedicated for customized arguments
    cust_parser = argparse.ArgumentParser(description="Customized Arguments")
    # add customized arguments
    cust_parser.add_argument('--rgat-encoder-type', type=str, default="ara",
                             help="The RGAT encoder type. Default is ara, a customized RGAT module.")
    cust_args = cust_parser.parse_args(unknown_args)
    print(f'Customized arguments: {cust_args}')

    # use both argument sets in our main function
    fit(gs_args, cust_args)
