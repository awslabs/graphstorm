""" HGT node classification trainer in GSF
"""
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

import graphstorm as gs
from graphstorm.config import GSConfig
from graphstorm import model as gsmodel
from graphstorm.trainer import GSgnnNodePredictionTrainer
from graphstorm.inference import GSgnnNodePredictionInferrer
from graphstorm.dataloading import GSgnnData
from graphstorm.dataloading import GSgnnNodeDataLoader
from graphstorm.eval import GSgnnClassificationEvaluator
from graphstorm.tracker import GSSageMakerTaskTracker
from graphstorm.utils import get_device

from dgl.nn.functional import edge_softmax


class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,            # input dimension
                 out_dim,           # output dimension
                 node_dict,         # node type and id in order, e.g., {'author': 0, 'paper': 1, 'subject': 2}
                 edge_dict,         # edge type and id in order, e.g., {'writing': 0, 'cited': 1, 'citing': 2}
                 num_heads,           # number of attention heads
                 dropout = 0.2,     # dropout rate, defaut is 0.2
                 use_norm = False   # Use normalization or not, default is False
                 ):
        super(HGTLayer, self).__init__()
        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_ntypes    = len(node_dict)
        self.num_etypes    = len(edge_dict)
        self.num_heads       = num_heads
        self.d_k           = out_dim // num_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm    = use_norm

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()

        for _ in range(self.num_ntypes):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_etypes, self.num_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_etypes, num_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_etypes, num_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_ntypes))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, graph, h):
        if graph.is_block:
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in graph.canonical_etypes:
                # extract each relation as a sub graph
                sub_graph = graph[srctype, etype, dsttype]

                # check if no edges exist for this can_etype
                if sub_graph.num_edges() == 0:
                    continue

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.num_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.num_heads, self.d_k)
                q = q_linear(h[dsttype][:sub_graph.num_dst_nodes()]).view(-1, self.num_heads, self.d_k)

                e_id = self.edge_dict[(srctype, etype, dsttype)]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata[f'v_{e_id :d}'] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            edge_fn = {}
            for etype, e_id in edge_dict.items():
                if etype not in graph.canonical_etypes:
                    continue
                else:
                    edge_fn[etype] = (fn.u_mul_e(f'v_{e_id :d}', 't', 'm'), fn.sum('m', 't'))
            graph.multi_update_all(edge_fn, cross_reducer = 'mean')

            new_h = {}
            for ntype in graph.dsttypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                if graph.num_dst_nodes(ntype) == 0:
                    continue

                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = graph.dstnodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype][: graph.num_dst_nodes(ntype)] * (1-alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out

            return new_h


class HGT(gsmodel.GSgnnNodeModelBase):
    def __init__(self,
                 node_id_dict,      # node type and id in order,
                                    #   e.g., {'author': 0, 'paper': 1, 'subject': 2}
                 edge_id_dict,      # edge type and id in order,
                                    #   e.g., {'writing': 0, 'cited': 1, 'citing': 2}
                 n_inp_dict,        # dictionary for input node feature dimensions,
                                    #   0 means featureless nodes
                 n_hid,             # hidden dimension
                 n_out,             # output dimension
                 num_layers,          # number of gnn layers
                 num_heads,           # number of attention
                 target_ntype,     # the node type to be predict
                 use_norm = True,   # use normalization or not, default is True
                 alpha_l2norm = 0,
                 lr = 0.001
                 ):
        super(HGT, self).__init__()
        self.node_dict = node_id_dict
        self.edge_dict = edge_id_dict
        self.num_layers = num_layers
        self.target_ntype=target_ntype
        self.alpha_l2norm = alpha_l2norm
        self.lr = lr

        # set adapt weights according to node id and feature dimension dictionary
        self.adapt_ws = nn.ModuleDict()
        featureless_ntype_cnt = 0
        self.ntype_id_map = {}
        for ntype, n_inp_dim in n_inp_dict.items():
            if n_inp_dim == 0:
                self.ntype_id_map[self.node_dict[ntype]] = featureless_ntype_cnt
                self.adapt_ws[ntype] = None
                featureless_ntype_cnt += 1
            else:
                self.adapt_ws[ntype] = nn.Linear(n_inp_dim, n_hid)
        # define an Embedding layers for featureless nodes
        if featureless_ntype_cnt > 0:
            self.ntype_embed = nn.Embedding(featureless_ntype_cnt, n_hid)

        # hgt layers
        self.gcs = nn.ModuleList()
        for _ in range(num_layers):
            self.gcs.append(HGTLayer(n_hid,
                                     n_hid,
                                     node_id_dict,
                                     edge_id_dict,
                                     num_heads,
                                     use_norm = use_norm))
        # output layer for classification
        self.out = nn.Linear(n_hid, n_out)

        # use GSF components
        self._loss_fn = gsmodel.ClassifyLossFunc(multilabel=False)

    def forward(self, blocks, node_feats, edge_feats, labels, input_nodes):
        # input layer
        h = {}
        for ntype in blocks[0].ntypes:
            if self.adapt_ws[ntype] is None:
                n_id = self.node_dict[ntype]
                emb_id = self.ntype_id_map[n_id]
                device = self.ntype_embed.device
                embeding = self.ntype_embed(torch.Tensor([emb_id]).long().to(device))
                n_embed = embeding.expand(blocks[0].num_nodes(ntype), -1)
            else:
                n_embed = self.adapt_ws[ntype](node_feats[ntype])

            h[ntype] = F.gelu(n_embed)
        # gnn layers
        for i in range(self.num_layers):
            h = self.gcs[i](blocks[i], h)
        # output layer
        for ntype, emb in h.items():
            h[ntype] = self.out(emb)

        # prediction loss computation
        pred_loss = self._loss_fn(h[self.target_ntype], labels[self.target_ntype])

        reg_loss = torch.tensor(0.).to(pred_loss.device)
        # L2 regularization of dense parameters
        for d_para in self.parameters():
            reg_loss += d_para.square().sum()

        reg_loss = self.alpha_l2norm * reg_loss

        return pred_loss + reg_loss

    def predict(self, blocks, node_feats, _, input_nodes, return_proba):
        # input layer
        h = {}
        for ntype in blocks[0].ntypes:
            if self.adapt_ws[ntype] is None:
                n_id = self.node_dict[ntype]
                emb_id = self.ntype_id_map[n_id]
                device = self.ntype_embed.device
                embeding = self.ntype_embed(torch.Tensor([emb_id]).long().to(device))
                n_embed = embeding.expand(blocks[0].num_nodes(ntype), -1)
            else:
                n_embed = self.adapt_ws[ntype](node_feats[ntype])

            h[ntype] = F.gelu(n_embed)
        # gnn layers
        for i in range(self.num_layers):
            h = self.gcs[i](blocks[i], h)
        # output layer
        for ntype, emb in h.items():
            h[ntype] = self.out(emb)

        if return_proba:
            return h[self.target_ntype].argmax(dim=1), torch.softmax(h[self.target_ntype], 1)
        else:
            return h[self.target_ntype].argmax(dim=1), h[self.target_ntype]

    def restore_model(self, restore_model_path):
        pass

    def save_model(self, model_path):
        pass

    def create_optimizer(self):
        # Here we don't set up an optimizer for sparse embeddings.
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main(args):
    config = GSConfig(args)
    gs.initialize(ip_config=args.ip_config, backend="gloo",
                  local_rank=config.local_rank)

    # Process node_feat_field to define GraphStorm dataset
    node_feat_fields = {}
    node_feat_types = args.node_feat.split('-')
    for node_feat_type in node_feat_types:
        node_type, feat_names = node_feat_type.split(':')
        node_feat_fields[node_type] = feat_names.split(',')

    # Define the GraphStorm training dataset
    train_data = GSgnnData(config.part_config)

    # Create input arguments for the HGT model
    node_dict = {}
    edge_dict = {}
    for ntype in train_data.g.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in train_data.g.canonical_etypes:
        edge_dict[etype] = len(edge_dict)

    nfeat_dims = {}
    for ntype, _ in node_dict.items():
        if train_data.g.nodes[ntype].data.get('feat') is not None:
            nfeat_dims[ntype] = train_data.g.nodes[ntype].data['feat'].shape[-1]
        else:
            nfeat_dims[ntype] = 0

    num_layers = len(config.fanout)

    # Define the HGT model
    model = HGT(node_dict, edge_dict,
                n_inp_dict=nfeat_dims,
                n_hid=config.hidden_size,
                n_out=config.num_classes,
                num_layers=num_layers,
                num_heads=args.num_heads,
                target_ntype=config.target_ntype,
                use_norm=True,
                alpha_l2norm=config.alpha_l2norm,
                lr=config.lr)

    # Create a trainer for the node classification task.
    trainer = GSgnnNodePredictionTrainer(model, topk_model_to_save=config.topk_model_to_save)
    trainer.setup_device(device=get_device())

    train_idxs = train_data.get_node_train_set(config.target_ntype)
    # Define the GraphStorm train dataloader
    dataloader = GSgnnNodeDataLoader(train_data, train_idxs, fanout=config.fanout,
                                     batch_size=config.batch_size,
                                     node_feats=node_feat_fields,
                                     label_field=config.label_field,
                                     train_task=True)

    eval_ntype = config.eval_target_ntype
    val_idxs = train_data.get_node_val_set(eval_ntype)
    test_idxs = train_data.get_node_test_set(eval_ntype)
    # Optional: Define the evaluation dataloader
    eval_dataloader = GSgnnNodeDataLoader(train_data, val_idxs, fanout=config.fanout,
                                          batch_size=config.eval_batch_size,
                                          node_feats=node_feat_fields,
                                          label_field=config.label_field,
                                          train_task=False)

    # Optional: Define the evaluation dataloader
    test_dataloader = GSgnnNodeDataLoader(train_data, test_idxs, fanout=config.fanout,
                                          batch_size=config.eval_batch_size,
                                          node_feats=node_feat_fields,
                                          label_field=config.label_field,
                                          train_task=False)

    # Optional: set up a evaluator
    evaluator = GSgnnClassificationEvaluator(config.eval_frequency,
                                             config.eval_metric,
                                             config.multilabel,
                                             config.use_early_stop,
                                             config.early_stop_burnin_rounds,
                                             config.early_stop_rounds,
                                             config.early_stop_strategy)
    trainer.setup_evaluator(evaluator)
    # Optional: set up a task tracker to show the progress of training.
    tracker = GSSageMakerTaskTracker(config.eval_frequency)
    trainer.setup_task_tracker(tracker)

    # Start the training process.
    trainer.fit(train_loader=dataloader,
                num_epochs=config.num_epochs,
                val_loader=eval_dataloader,
                test_loader=test_dataloader,
                save_model_path=config.save_model_path,
                use_mini_batch_infer=True)

    # After training, get the best model from the trainer.
    best_model_path = trainer.get_best_model_path()
    model.restore_model(best_model_path)

    # Create a dataset for inference.
    infer_data = GSgnnData(config.part_config)

    # Create an inference for a node task.
    infer = GSgnnNodePredictionInferrer(model)
    infer.setup_device(device=get_device())
    infer.setup_evaluator(evaluator)
    infer.setup_task_tracker(tracker)
    infer_idxs = infer_data.get_node_infer_set(eval_ntype)
    dataloader = GSgnnNodeDataLoader(infer_data,infer_idxs,
                                    fanout=config.fanout, batch_size=100,
                                    node_feats=node_feat_fields,
                                    label_field=config.label_field,
                                    train_task=False)

    # Run inference on the inference dataset and save the GNN embeddings in the specified path.
    infer.infer(dataloader, save_embed_path=config.save_embed_path,
                save_prediction_path=config.save_prediction_path,
                use_mini_batch_infer=True)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Training HGT model with the GraphStorm Framework")
    argparser.add_argument("--yaml-config-file", type=str, required=True,
                           help="The GraphStorm YAML configuration file path.")
    argparser.add_argument("--node-feat", type=str, required=True,
                           help="The name of the node features. \
                                 Format is nodetype1:featname1,featname2-nodetype2:featname1,...")
    argparser.add_argument("--num-heads", type=int, default=4,
                           help="The number of heads for HGT's self-attention module")
    argparser.add_argument("--part-config", type=str, required=True,
                           help="The partition config file. \
                                 For customized models, MUST have this argument!!")
    argparser.add_argument("--ip-config", type=str, required=True,
                           help="The IP config file for the cluster. \
                                 For customized models, MUST have this argument!!")
    argparser.add_argument("--verbose",
                           type=lambda x: (str(x).lower() in ['true', '1']),
                           default=argparse.SUPPRESS,
                          help="Print more information. \
                                For customized models, MUST have this argument!!")
    argparser.add_argument("--local-rank", type=int,
                           help="The rank of the trainer. \
                                 For customized models, MUST have this argument!!")

    # Ignore unknown args to make script more robust to input arguments
    args, _ = argparser.parse_known_args()
    print(args)
    main(args)
