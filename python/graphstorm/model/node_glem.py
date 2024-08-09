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
import os
from copy import deepcopy
import torch as th
import dgl

from .gnn import GSOptimizer
from .node_gnn import GSgnnNodeModel, GSgnnNodeModelBase

# GLEM supports configuring the parameter grouping of the following:
GLEM_CONFIGURABLE_PARAMETER_NAMES = {
    "gnn_param_group": set(["pure_lm", "sparse_embed", "node_input_projs", "node_proj_matrix"]),
    "lm_param_group": set(["pure_lm", "node_input_projs", "node_proj_matrix"])
    }

class GLEMOptimizer(GSOptimizer):
    """ An optimizer specific for GLEM, implementing on/off switches of sparse optimizer.
    """
    def _clear_traces(self):
        """ Clear the traces in sparse optimizers.
        """
        for optimizer in self.sparse_opts:
            for emb in optimizer._params:
                emb.reset_trace()

    def zero_grad(self, optimize_sparse_params=True):
        """ Setting the gradient to zero
        """
        all_opts = self.dense_opts + self.lm_opts
        if optimize_sparse_params:
            all_opts += self.sparse_opts
        for optimizer in all_opts:
            optimizer.zero_grad()
        if not optimize_sparse_params:
            # force reset trace for sparse opt when sparse emb are frozen
            # under this condition, emb still collects traces with grad being None's.
            # we need to do this to ensure the gradient update step after unfreezing
            # can be performed correctly.
            self._clear_traces()

    def step(self, optimize_sparse_params=True):
        """ Moving the optimizer
        """
        all_opts = self.dense_opts + self.lm_opts
        if optimize_sparse_params:
            all_opts += self.sparse_opts
        for optimizer in all_opts:
            optimizer.step()

class GLEM(GSgnnNodeModelBase):
    """
    GLEM model (https://arxiv.org/abs/2210.14709) for node-level tasks.

    Parameters
    ----------
    alpha_l2norm: float
        Coef for l2 norm of unused weights
    target_ntype: str
        Target node type
    em_order_gnn_first: bool
        In the EM training, set true to train GNN first, train LM first otherwise
    inference_using_gnn: bool
        Set true to use GNN in inference time, otherwise inference with LM
    pl_weight: float
        Weight for the pseudo-likelihood loss in GLEM's loss function.
    num_pretrain_epochs: int
        Number of pretraining epochs to train LM and GNN independently without
        pseudo-likelihood loss.
    lm_param_group: List[str]
        names of parameters that will be optimized when training LM.
    gnn_param_group: List[str]
        names of parameters that will be optimized when training GNN.
    """
    def __init__(self,
                 alpha_l2norm,
                 target_ntype,
                 em_order_gnn_first=False,
                 inference_using_gnn=True,
                 pl_weight=0.5,
                 num_pretrain_epochs=5,
                 lm_param_group=None,
                 gnn_param_group=None
                 ):
        super(GLEM, self).__init__()
        self.alpha_l2norm = alpha_l2norm
        self.target_ntype = target_ntype
        self.em_order_gnn_first = em_order_gnn_first
        self.inference_using_gnn = inference_using_gnn
        self.pl_weight = pl_weight
        self.num_pretrain_epochs = num_pretrain_epochs
        self.lm = GSgnnNodeModel(alpha_l2norm)
        self.gnn = GSgnnNodeModel(alpha_l2norm)
        # `training_lm` has three states, controled by `.toggle()`:
        # None: model is loaded for inference, inference logic is decided by `inference_using_gnn`
        # True: lm is being trained
        # False: gnn is being trained
        self.training_lm = None
        if lm_param_group is None:
            lm_param_group = ["pure_lm", "node_proj_matrix"]
        if gnn_param_group is None:
            gnn_param_group = ["node_input_projs"]
        assert set(lm_param_group).issubset(GLEM_CONFIGURABLE_PARAMETER_NAMES['lm_param_group'])
        assert set(gnn_param_group).issubset(GLEM_CONFIGURABLE_PARAMETER_NAMES['gnn_param_group'])
        self.param_names_groups = {'lm': lm_param_group, 'gnn': gnn_param_group}
        # set up default flag for `training_sparse_embed` based on whether its trainable at
        # either stages:
        self.training_sparse_embed = 'sparse_embed' in lm_param_group + gnn_param_group

    @property
    def named_params(self):
        """Mapping parameter name to the actual list of parameters."""
        return {
            'pure_lm': self.lm.get_lm_params(),
            'sparse_embed': self.lm.get_sparse_params(),
            'node_input_projs': list(self.lm.node_input_encoder.input_projs.parameters()),
            'node_proj_matrix': list(self.lm.node_input_encoder.proj_matrix.parameters()),
        }

    def trainable_parameters(self, part):
        """To access the trainable torch parameters from lm or gnn part of the model."""
        if part == 'lm':
            params = list(self.lm.decoder.parameters())
        else:
            params = list(self.gnn.gnn_encoder.parameters()) + list(self.gnn.decoder.parameters())
        for param_name in self.param_names_groups[part]:
            if param_name != 'sparse_embed':
                params.extend(self.named_params[param_name])
        return params

    @property
    def inference_route_is_gnn(self):
        """This flag decides which inference route to perform: gnn (True) or lm (False).
        This is decided based on the values of `training_lm` and `inference_using_gnn`.
        There are two inference routes for GLEM:
        - False (lm): lm.node_input_encoder->lm.decoder
        - True (gnn): lm.node_input_encoder->gnn.gnn_encoder->gnn.decoder
        """
        if self.training_lm is None:
            # GLEM is loaded for inference only, decide based `inference_using_gnn`
            return self.inference_using_gnn
        # GLEM is being trained: use gnn route if not training lm
        return not self.training_lm

    def init_optimizer(self, lr, sparse_optimizer_lr=None, weight_decay=0, lm_lr=None):
        """Initialize optimzer, which will be stored in self.lm._optimizer, self.gnn._optimizer
        """
        sparse_params = self.lm.get_sparse_params()
        # check and set the sparse optimizer learning rate
        if sparse_optimizer_lr is None:
            sparse_optimizer_lr = lr
        if len(sparse_params) > 0:
            emb_optimizer = dgl.distributed.optim.SparseAdam(sparse_params, lr=sparse_optimizer_lr)
            sparse_opts = [emb_optimizer]
        else:
            sparse_opts = []

        dense_params = self.gnn.get_dense_params() + self.lm.get_dense_params()
        if len(dense_params) > 0:
            optimizer = th.optim.Adam(dense_params, lr=lr,
                                      weight_decay=weight_decay)
            dense_opts = [optimizer]
        else:
            dense_opts = []

        lm_params = self.lm.get_lm_params()
        if len(lm_params) > 0:
            lm_optimizer = th.optim.Adam(lm_params, \
                                         lr=lm_lr if lm_lr is not None else lr,
                                         weight_decay=weight_decay)
            lm_opts = [lm_optimizer]
        else:
            lm_opts = []
        self._optimizer = GLEMOptimizer(dense_opts=dense_opts,
                                      lm_opts=lm_opts,
                                      sparse_opts=sparse_opts)

    def create_optimizer(self):
        """Create the optimizer that optimizes the model."""
        return self._optimizer

    def save_dense_model(self, model_path):
        self.lm.save_dense_model(os.path.join(model_path, 'LM'))
        self.gnn.save_dense_model(os.path.join(model_path, 'GNN'))

    def save_sparse_model(self, model_path):
        self.lm.save_sparse_model(os.path.join(model_path, 'LM'))
        self.gnn.save_sparse_model(os.path.join(model_path, 'GNN'))

    def restore_dense_model(self, restore_model_path,
                            model_layer_to_load=None):
        self.lm.restore_dense_model(os.path.join(restore_model_path, 'LM'),
                                    model_layer_to_load)
        self.gnn.restore_dense_model(os.path.join(restore_model_path, 'GNN'),
                                     ['gnn', 'decoder'])

    def restore_sparse_model(self, restore_model_path):
        self.lm.restore_sparse_model(os.path.join(restore_model_path, 'LM'))

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
        """Alias for accessing the gnn_encoder. Hide gnn_encoder if the inference route is lm.
        This property is only used for model inference and evaluation."""
        return self.gnn.gnn_encoder if self.inference_route_is_gnn else None

    @property
    def node_input_encoder(self):
        """Alias for accessing the node_input_encoder.
        This property is only used for model inference and evaluation."""
        return self.lm.node_input_encoder

    @property
    def decoder(self):
        """Alias for accessing the decoder.
        This property is only used for model inference and evaluation."""
        return self.gnn.decoder if self.inference_route_is_gnn else self.lm.decoder

    def set_decoder(self, decoder):
        """Set the same decoder for both, since lm needs to be able to
        predict node labels during training
        """
        self.lm.set_decoder(deepcopy(decoder))
        self.gnn.set_decoder(decoder)

    def set_loss_func(self, loss_fn):
        """Set the final loss function based on the node task
        """
        self.lm.set_loss_func(loss_fn)
        self.gnn.set_loss_func(loss_fn)

    def toggle_params(self, part='lm', freeze=True):
        """Freeze or unfreeze parameters in lm or gnn"""
        if 'sparse_embed' in self.param_names_groups[part]:
            self.training_sparse_embed = not freeze
        for param in self.trainable_parameters(part):
            param.requires_grad = not freeze

    def toggle(self, part='lm', data=None):
        """The method toggles training between lm and gnn. It uses `toggle_params` to
        freeze/unfreeze model parameters and `(un)freeze_input_encoder` to control the
        caching of LM embeddings"""
        if part == 'lm':
            self.training_lm = True
            self.toggle_params('gnn', True)
            self.toggle_params('lm', False)
            # when training lm, do not use the cached LM
            self.lm.unfreeze_input_encoder()
        elif part == 'gnn':
            self.training_lm = False
            self.toggle_params('lm', True)
            self.toggle_params('gnn', False)
            # when training gnn, always cache LM embeddings
            self.lm.freeze_input_encoder(data)
        else:
            raise ValueError(f"Unknown model part: {part}")

    def forward(self, blocks, node_feats, edge_feats, labels, input_nodes, use_gnn=True,
                no_pl=False, blocks_u=None, node_feats_u=None, edge_feats_u=None,
                input_nodes_u=None):
        """ Forward pass for GLEM model.
        Parameters
        ----------
        blocks : List[dgl.heterograph.DGLBlock] labeled message flow graphs
        node_feats : Dict[ntype: tensors.shape [bs, feat_dim]]
        edge_feats : None
        labels : Dict[target_ntype: tensor.shape [bs]]
        input_nodes : {target_ntype: tensor.shape [bs], other_ntype: []}
        use_gnn : bool
            If True, use GNN's decoder, otherwise, use LM's decoder
        no_pl : bool
            If True, do not calculate pseudo likelihood, use MLE loss only
        blocks_u : List[dgl.heterograph.DGLBlock] unlabeled message flow graphs
        node_feats_u : Dict[ntype: tensors.shape [bs, feat_dim]] unlabeled node features
        edge_feats_u : None
        input_nodes_u : {target_ntype: tensor.shape [bs], other_ntype: []} unlabeled nodes
        """
        if blocks_u is None or no_pl:
            # no unlabeled data provided
            if use_gnn:
                total_loss = self.forward_gnn(blocks, node_feats, edge_feats, labels, input_nodes,
                                            no_pl=no_pl)
            else:
                total_loss = self.forward_lm(blocks, node_feats, edge_feats, labels, input_nodes,
                                            no_pl=no_pl)
        else:
            if use_gnn:
                total_loss = self.forward_gnn_semisup(blocks, node_feats, edge_feats, labels,
                                                      input_nodes, blocks_u, node_feats_u,
                                                      edge_feats_u, input_nodes_u)
            else:
                total_loss = self.forward_lm_semisup(blocks, node_feats, edge_feats, labels,
                                                     input_nodes, blocks_u, node_feats_u,
                                                     edge_feats_u, input_nodes_u)
        return total_loss

    def forward_gnn(self, blocks, node_feats, _, labels, input_nodes, no_pl=False):
        """Forward pass for node prediction using GNN
        """
        emb_lm, emb_gnn = self._embed_nodes(blocks, node_feats, _, input_nodes, do_gnn_encode=True)
        labels = self._process_labels(labels)
        logits = self.gnn.decoder(emb_gnn)
        if no_pl:
            loss = self.gnn.loss_func(logits, labels)
        else:
            batch_size = labels.size(0)
            n_gold_nodes = batch_size // 2
            # compute pseudo labels from LM
            logits_lm = self.lm.decoder(emb_lm[n_gold_nodes:])
            pseudo_labels = logits_lm.argmax(-1)

            # GLEM loss
            loss = compute_loss(self.gnn.loss_func, logits, labels[:n_gold_nodes], pseudo_labels,
                                pl_weight=self.pl_weight)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(loss.device)
        # L2 regularization of dense parameters
        for d_para in self.gnn.get_dense_params():
            reg_loss += d_para.square().sum()
        # weighted addition to the total loss
        return loss + self.alpha_l2norm * reg_loss

    def forward_gnn_semisup(self, blocks, node_feats, edge_feats, labels, input_nodes, blocks_u,
                            node_feats_u, edge_feats_u, input_nodes_u):
        """Forward pass for node prediction using GNN with unlabeled nodes
        """
        _, emb_gnn = self._embed_nodes(blocks, node_feats, edge_feats, input_nodes,
                                       do_gnn_encode=True)
        labels = self._process_labels(labels)

        emb_lm_u, emb_gnn_u = self._embed_nodes(blocks_u, node_feats_u, edge_feats_u,
                                                input_nodes_u, do_gnn_encode=True)
        # compute pseudo labels from LM
        logits_lm = self.lm.decoder(emb_lm_u)
        pseudo_labels = logits_lm.argmax(-1)
        logits = self.gnn.decoder(th.cat([emb_gnn, emb_gnn_u]))
        # GLEM loss
        loss = compute_loss(self.gnn.loss_func, logits, labels, pseudo_labels,
                            pl_weight=self.pl_weight)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(loss.device)
        # L2 regularization of dense parameters
        for d_para in self.gnn.get_dense_params():
            reg_loss += d_para.square().sum()
        # weighted addition to the total loss
        return loss + self.alpha_l2norm * reg_loss

    def forward_lm(self, blocks, node_feats, _, labels, input_nodes, no_pl=False):
        """Forward pass for node prediction using LM"""
        emb_lm, emb_gnn = self._embed_nodes(blocks, node_feats, _, input_nodes,
                                            do_gnn_encode=not no_pl)
        labels = self._process_labels(labels)
        logits = self.lm.decoder(emb_lm)
        if no_pl:
            loss = self.lm.loss_func(logits, labels)
        else:
            batch_size = labels.size(0)
            n_gold_nodes = batch_size // 2
            # compute pseudo labels from GNN
            logits_gnn = self.gnn.decoder(emb_gnn[n_gold_nodes:])
            pseudo_labels = logits_gnn.argmax(-1)

            # GLEM loss
            loss = compute_loss(self.gnn.loss_func, logits, labels[:n_gold_nodes], pseudo_labels,
                                pl_weight=self.pl_weight)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(loss.device)
        # L2 regularization of dense parameters
        for d_para in self.gnn.get_dense_params():
            reg_loss += d_para.square().sum()
        # weighted addition to the total loss
        return loss + self.alpha_l2norm * reg_loss

    def forward_lm_semisup(self, blocks, node_feats, edge_feats, labels, input_nodes, blocks_u,
                           node_feats_u, edge_feats_u, input_nodes_u):
        """Forward pass for node prediction using LM with unlabeled nodes
        """
        emb_lm, _ = self._embed_nodes(blocks, node_feats, edge_feats, input_nodes,
                                      do_gnn_encode=False)
        labels = self._process_labels(labels)
        emb_lm_u, emb_gnn_u = self._embed_nodes(blocks_u, node_feats_u, edge_feats_u, input_nodes_u,
                                                do_gnn_encode=True)
        logits = self.lm.decoder(th.cat([emb_lm, emb_lm_u]))
        # compute pseudo labels from GNN
        logits_gnn = self.gnn.decoder(emb_gnn_u)
        pseudo_labels = logits_gnn.argmax(-1)
        # GLEM loss
        loss = compute_loss(self.gnn.loss_func, logits, labels, pseudo_labels,
                            pl_weight=self.pl_weight)
        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(loss.device)
        # L2 regularization of dense parameters
        for d_para in self.gnn.get_dense_params():
            reg_loss += d_para.square().sum()
        # weighted addition to the total loss
        return loss + self.alpha_l2norm * reg_loss

    def predict(self, blocks, node_feats, edge_feats, input_nodes, return_proba):
        """Make prediction on the nodes with the LM or GNN.
        The model's `inference_route_is_gnn` flag determines how inference is performed.
        If inference_route_is_gnn, message-passing GNN is used on the LM features,
        Otherwise, LM's decoder is used for inference, no message-passing involved.

        Parameters
        ----------
        blocks : list of DGLBlock
            The message passing graph for computing GNN embeddings.
        node_feats : dict of Tensors
            The node features of the message passing graphs.
        edge_feats : dict of Tensors
            The edge features of the message passing graphs.
        input_nodes: dict of Tensors
            The input nodes of a mini-batch.
        return_proba : bool
            Whether or not to return all the predicted results or only the maximum one

        Returns
        -------
        Tensor : GNN prediction results. Return all the results when return_proba is true
            otherwise return the maximum result.
        Tensor : the GNN embeddings.
        """
        emb_lm, emb_gnn = self._embed_nodes(blocks, node_feats, edge_feats, input_nodes,
                                do_gnn_encode=self.inference_route_is_gnn)
        if self.inference_route_is_gnn:
            decoder = self.gnn.decoder
            emb = emb_gnn
        else:
            decoder = self.lm.decoder
            emb = emb_lm
        if return_proba:
            preds = decoder.predict_proba(emb)
        else:
            preds = decoder.predict(emb)
        return preds, emb

    # pylint: disable=unused-argument
    def inplace_normalize_node_embs(self, embs):
        """ Do inplace node embedding normalization.

            This function is called by do_full_graph_inference().

            Parameters
            ----------
            embs: dict of Tensor
                Node embeddings.
        """
        # GLEM node model does not need normalization
        # do nothing.
        return

    def _get_seed_nodes(self, input_nodes, node_feats, blocks):
        """ Get seed nodes and features from input nodes and labels of the seed nodes.
        Parameters
        ----------
        input_nodes : {target_ntype: tensor.shape [bs], other_ntype: []}
        node_feats : {ntype: tensor}
        blocks : list[dgl.Block]
        """
        target_ntype = self.target_ntype
        n_seed_nodes = blocks[-1].num_dst_nodes()
        seed_nodes = {target_ntype: input_nodes[target_ntype][:n_seed_nodes]}
        seed_feats = {}
        if target_ntype in node_feats:
            seed_feats = {target_ntype: node_feats[target_ntype][:n_seed_nodes]}
        return seed_nodes, seed_feats

    def _embed_nodes(self, blocks, node_feats, _, input_nodes=None, do_gnn_encode=True):
        """ Embed and encode nodes with LM, optionally followed by GNN encoder for GLEM model
        """
        target_ntype = self.target_ntype
        if do_gnn_encode:
            # Get the projected LM embeddings without GNN message passing
            encode_embs = self.lm.comput_input_embed(input_nodes, node_feats)
            # GNN message passing
            encode_embs_gnn = self.gnn.gnn_encoder(blocks, encode_embs)
            n_seed_nodes = blocks[-1].num_dst_nodes()

            # Call emb normalization.
            # the default behavior is doing nothing.
            encode_emb = self.normalize_node_embs(
                {target_ntype:encode_embs[target_ntype][:n_seed_nodes]})[target_ntype]
            encode_emb_gnn = self.normalize_node_embs(
                {target_ntype:encode_embs_gnn[target_ntype]})[target_ntype]
            return encode_emb, encode_emb_gnn
        else:
            # Get the projected LM embeddings for seed nodes and corresponding node features:
            seed_nodes, seed_feats = self._get_seed_nodes(input_nodes, node_feats, blocks)
            encode_embs = self.lm.comput_input_embed(seed_nodes, seed_feats)
            # Call emb normalization.
            # the default behavior is doing nothing.
            encode_emb = self.normalize_node_embs(
                {target_ntype:encode_embs[target_ntype]})[target_ntype]
            return encode_emb, None

    def _process_labels(self, labels):
        # TODO(zhengda) we only support node prediction on one node type now
        assert len(labels) == 1, "We only support prediction on one node type for now."
        target_ntype = list(labels.keys())[0]
        return labels[target_ntype]


def compute_loss(loss_func, logits, labels, pseudo_labels=None, pl_weight=0.5):
    """Combine two types of losses: (1-alpha)*MLE (CE loss on gold labels) + alpha*Pl_loss
    (CE loss on pseudo labels) semi-supervised objective for training LM or GNN individually
    in E-step, M-step, respectively.
    If pseudo_labels is provided, assuming the first segment are for gold, and later segment are
    for pseudo in logits.
    """
    if pseudo_labels is not None:
        assert logits.size(0) == labels.size(0) + pseudo_labels.size(0), \
            "logits prediction miss matches the provided gold and pseudo labels"
        n_golds = labels.size(0)

        def deal_nan(x):
            return 0 if th.isnan(x) else x
        mle_loss = deal_nan(loss_func(logits[:n_golds], labels))
        pl_loss = deal_nan(loss_func(logits[n_golds:], pseudo_labels))
        loss = pl_weight * pl_loss + (1 - pl_weight) * mle_loss
    else:
        loss = loss_func(logits, labels)
    return loss
