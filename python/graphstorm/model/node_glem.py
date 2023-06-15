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

    GLEM model for node prediction task in GraphStorm.
"""
import abc
import torch as th
import dgl

from .gnn import GSgnnModel, GSgnnModelBase, GSOptimizer
from .node_gnn import GSgnnNodeModel, GSgnnNodeModelBase

class GLEM(GSgnnNodeModelBase):
    def __init__(self, 
                 alpha_l2norm,
                 em_order_gnn_first=True,
                 inference_using_gnn=True,
                 pl_weight=0.5
                 ):
        super(GLEM, self).__init__()
        self.alpha_l2norm = alpha_l2norm
        self.em_order_gnn_first = em_order_gnn_first
        self.inference_using_gnn = inference_using_gnn
        self.pl_weight = pl_weight
        self.lm = GSgnnNodeModel(alpha_l2norm)
        self.gnn = GSgnnNodeModel(alpha_l2norm)
        # self.lm.node_input_encoder: graphstorm.model.lm_embed.GSLMNodeEncoderInputLayer
        # self.lm.gnn_encoder: None
        # self.lm.decoder: EntityClassifier

    def init_optimizer(self, lr, sparse_optimizer_lr, weight_decay, lm_lr=None):
        # optimizer will be stored in self.lm._optimizer, self.gnn._optimizer
        sparse_params = self.gnn.get_sparse_params()
        if len(sparse_params) > 0:
            emb_optimizer = dgl.distributed.optim.SparseAdam(sparse_params, lr=sparse_optimizer_lr)
            sparse_opts = [emb_optimizer]
        else:
            sparse_opts = []

        dense_params = self.gnn.get_dense_params()
        if len(dense_params) > 0:
            optimizer = th.optim.Adam(self.gnn.get_dense_params(), lr=lr,
                                      weight_decay=weight_decay)
            dense_opts = [optimizer]
        else:
            dense_opts = []

        # Combine params in the LM transformers and LM decoder head
        lm_params = self.lm.get_lm_params() + self.lm.get_dense_params()
        if len(lm_params) > 0:
            lm_optimizer = th.optim.Adam(lm_params, \
                                         lr=lm_lr if lm_lr is not None else lr,
                                         weight_decay=weight_decay)
            lm_opts = [lm_optimizer]
        else:
            lm_opts = []
        self._optimizer = GSOptimizer(dense_opts=dense_opts,
                                      lm_opts=lm_opts,
                                      sparse_opts=sparse_opts)        

    def create_optimizer(self):
        return self._optimizer
    
    def set_node_input_encoder(self, encoder):
        """Set the node input LM encoder for lm, shared with gnn. 
        """
        self.lm.set_node_input_encoder(encoder)

    def set_gnn_encoder(self, gnn_encoder):
        """Set gnn encoder only for gnn.
        """
        self.gnn.set_gnn_encoder(gnn_encoder)

    @property
    def gnn_encoder(self):
        return self.gnn.gnn_encoder

    def set_decoder(self, decoder):
        """Set the same decoder for both, since lm needs to be able to predict node labels during training
        """
        self.lm.set_decoder(decoder)
        self.gnn.set_decoder(decoder)

    def set_loss_func(self, loss_fn):
        """Set the final loss function based on the node task 
        """
        self.lm.set_loss_func(loss_fn)
        self.gnn.set_loss_func(loss_fn)

    def freeze_params(self, part='lm'):
        if part == 'lm':
            params = self.lm.parameters()
        elif part == 'gnn':
            params = self.gnn.parameters()
        for param in params:
            param.requires_grad = False
    
    def unfreeze_params(self, part='lm'):
        if part == 'lm':
            params = self.lm.parameters()
        elif part == 'gnn':
            params = self.gnn.parameters()
        for param in params:
            param.requires_grad = True

    def forward(self, blocks, node_feats, edge_feats, labels, input_nodes, use_gnn=True):
        """ Forward pass for GLEM model.
        Parameters
        ----------        
        blocks : List[dgl.heterograph.DGLBlock]
        node_feats : Dict[ntype: tensors.shape [bs, feat_dim]]
        edge_feats : None
        labels : Dict[target_ntype: tensor.shape [bs]]
        input_nodes : {target_ntype: tensor.shape [bs], other_ntype: []}
        use_gnn : bool
            If True, use GNN's decoder, otherwise, use LM's decoder 
        """
        if use_gnn:
            total_loss = self.forward_gnn(blocks, node_feats, edge_feats, labels, input_nodes)
        else:
            total_loss = self.forward_lm(blocks, node_feats, edge_feats, labels, input_nodes)
        return total_loss

    def forward_gnn(self, blocks, node_feats, _, labels, input_nodes):
        """Forward pass for node prediction using GNN
        """
        emb = self._embed_nodes(blocks, node_feats, _, input_nodes)
        labels = self._process_labels(labels)
        logits = self.gnn.decoder(emb)

        batch_size = labels.size(0)
        n_gold_nodes = batch_size // 2
        # compute pseudo labels from LM
        logits_lm = self.lm.decoder(emb[n_gold_nodes:])
        pseudo_labels = logits_lm.argmax(-1)
        
        # GLEM loss
        loss = compute_loss(self.gnn.loss_func, logits, labels[:n_gold_nodes], pseudo_labels, pl_weight=self.pl_weight)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(loss.device)
        # L2 regularization of dense parameters
        for d_para in self.gnn.get_dense_params():
            reg_loss += d_para.square().sum()
        # weighted addition to the total loss
        return loss + self.alpha_l2norm * reg_loss

    
    def forward_lm(self, blocks, node_feats, _, labels, input_nodes):
        """Forward pass for node prediction using LM"""
        emb = self._embed_nodes(blocks, node_feats, _, input_nodes)
        labels = self._process_labels(labels)
        logits = self.lm.decoder(emb)
        batch_size = labels.size(0)
        n_gold_nodes = batch_size // 2
        # compute pseudo labels from GNN
        logits_gnn = self.gnn.decoder(emb[n_gold_nodes:])
        pseudo_labels = logits_gnn.argmax(-1)

        # GLEM loss
        loss = compute_loss(self.gnn.loss_func, logits, labels[:n_gold_nodes], pseudo_labels, pl_weight=self.pl_weight)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(loss.device)
        # L2 regularization of dense parameters
        for d_para in self.gnn.get_dense_params():
            reg_loss += d_para.square().sum()
        # weighted addition to the total loss
        return loss + self.alpha_l2norm * reg_loss

    
    def predict(self, blocks, node_feats, edge_feats, input_nodes, return_proba):        
        emb = self._embed_nodes(blocks, node_feats, edge_feats, input_nodes)
        if self.inference_using_gnn:
            decoder = self.gnn.decoder
        else:
            decoder = self.lm.decoder
        if return_proba:
            preds = decoder.predict_proba(emb)
        else:
            preds = decoder.predict(emb)
        return preds, emb
    
    def _embed_nodes(self, blocks, node_feats, _, input_nodes=None):
        """ Embed and encode nodes with LM, followed by GNN encoder for GLEM model
        """
        # Get the projected LM embeddings without GNN message passing
        encode_embs = self.lm.comput_input_embed(input_nodes, node_feats)
        # GNN message passing:
        encode_embs = self.gnn.gnn_encoder(blocks, encode_embs)
        target_ntype = list(encode_embs.keys())[0]
        emb = encode_embs[target_ntype]        
        return emb
    
    def _process_labels(self, labels):
        # TODO(zhengda) we only support node prediction on one node type now
        assert len(labels) == 1, "We only support prediction on one node type for now."
        target_ntype = list(labels.keys())[0]
        return labels[target_ntype]


def compute_loss(loss_func, logits, labels, pseudo_labels=None, pl_weight=0.5):
    """
    Combine two types of losses: (1-alpha)*MLE (CE loss on gold labels) + alpha*Pl_loss (CE loss on pseudo labels)
    semi-supervised objective for training LM or GNN individually in E-step, M-step, respectively.

    If pseudo_labels is provided, assuming the first segment are for gold, and later segment are for pseudo in logits.
    """
    if pseudo_labels is not None:
        assert logits.size(0) == labels.size(0) + pseudo_labels.size(0), "logits prediction miss matches the provided gold and pseudo labels"
        n_golds = labels.size(0)

        deal_nan = lambda x: 0 if th.isnan(x) else x
        mle_loss = deal_nan(loss_func(logits[:n_golds], labels))
        pl_loss = deal_nan(loss_func(logits[n_golds:], pseudo_labels))
        loss = pl_weight * pl_loss + (1 - pl_weight) * mle_loss
    else:
        loss = loss_func(logits, labels)
    return loss
