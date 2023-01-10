"""GNN model for node prediction task in GraphStorm
"""
import torch as th

from .gnn import GSgnnModel

class GSgnnNodeModel(GSgnnModel):
    """ GraphStorm GNN model for node prediction tasks

    Parameters
    ----------
    alpha_l2norm : float
        The alpha for L2 normalization.
    """
    def __init__(self, alpha_l2norm):
        super(GSgnnNodeModel, self).__init__()
        self.alpha_l2norm = alpha_l2norm

    def forward(self, blocks, input_feats, input_nodes, labels):
        """ The forward function for node prediction.

        Parameters
        ----------
        blocks : list of DGLBlock
            The message passing graph for computing GNN embeddings.
        input_feats : dict of Tensors
            The input features of the message passing graphs.
        input_nodes : dict of Tensors
            The input nodes of the message passing graphs.
        labels: dict of Tensor
            The labels of the predicted nodes.

        Returns
        -------
        The loss of prediction.
        """
        alpha_l2norm = self.alpha_l2norm
        gnn_embs = self.compute_embed_step(blocks, input_feats, input_nodes)
        # TODO(zhengda) we only support node prediction on one node type now
        assert len(labels) == 1, "We only support prediction on one node type for now."
        target_ntype = list(labels.keys())[0]
        assert target_ntype in gnn_embs
        emb = gnn_embs[target_ntype]
        labels = labels[target_ntype]
        logits = self.decoder(emb)
        pred_loss = self.loss_func(logits, labels)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(pred_loss.device)
        # L2 regularization of dense parameters
        for d_para in self.get_dense_params():
            reg_loss += d_para.square().sum()

        # weighted addition to the total loss
        return pred_loss + alpha_l2norm * reg_loss

    def predict(self, blocks, input_feats, input_nodes):
        """ Make prediction on the nodes with GNN.

        Parameters
        ----------
        blocks : list of DGLBlock
            The message passing graph for computing GNN embeddings.
        input_feats : dict of Tensors
            The input features of the message passing graphs.
        input_nodes : dict of Tensors
            The input nodes of the message passing graphs.

        Returns
        -------
        Tensor : the prediction results.
        Tensor : the GNN embeddings.
        """
        gnn_embs = self.compute_embed_step(blocks, input_feats, input_nodes)
        # TODO(zhengda) we only support node prediction on one node type.
        assert len(gnn_embs) == 1, 'There are {} node types: {}'.format(len(gnn_embs),
                                                                        list(gnn_embs.keys()))
        target_ntype = list(gnn_embs.keys())[0]
        return self.decoder.predict(gnn_embs[target_ntype]), gnn_embs[target_ntype]

def node_mini_batch_gnn_predict(model, loader, return_label=False):
    """ Perform mini-batch prediction on a GNN model.

    Parameters
    ----------
    model : GSgnnModel
        The GraphStorm GNN model
    loader : GSgnnNodeDataLoader
        The GraphStorm dataloader
    return_label : bool
        Whether or not to return labels.

    Returns
    -------
    Tensor : GNN prediction results.
    Tensor : GNN embeddings.
    Tensor : labels if return_labels is True
    """
    device = model.device
    data = loader.data
    preds = []
    embs = []
    labels = []
    model.eval()
    with th.no_grad():
        for input_nodes, seeds, blocks in loader:
            input_feats = data.get_node_feats(input_nodes, device)
            blocks = [block.to(device) for block in blocks]
            pred, emb = model.predict(blocks, input_feats, input_nodes)
            preds.append(pred.cpu())
            embs.append(emb.cpu())

            if return_label:
                lbl = data.get_labels(seeds)
                assert len(lbl) == 1
                labels.append(list(lbl.values())[0])
    model.train()
    preds = th.cat(preds)
    embs = th.cat(embs)
    if return_label:
        labels = th.cat(labels)
        return preds, embs, labels
    else:
        return preds, embs

def node_mini_batch_predict(model, emb, loader, return_label=False):
    """ Perform mini-batch prediction.

    Parameters
    ----------
    model : GSgnnModel
        The GraphStorm GNN model
    emb : dict of Tensor
        The GNN embeddings
    loader : GSgnnNodeDataLoader
        The GraphStorm dataloader
    return_label : bool
        Whether or not to return labels.

    Returns
    -------
    Tensor : GNN prediction results.
    Tensor : labels if return_labels is True
    """
    device = model.device
    data = loader.data
    preds = []
    labels = []
    # TODO(zhengda) I need to check if the data loader only returns target nodes.
    model.eval()
    with th.no_grad():
        for input_nodes, seeds, _ in loader:
            assert len(input_nodes) == 1, "Currently we only support one node type"
            ntype = list(input_nodes.keys())[0]
            in_nodes = input_nodes[ntype]
            pred = model.decoder.predict(emb[ntype][in_nodes].to(device))
            preds.append(pred.cpu())
            if return_label:
                lbl = data.get_labels(seeds)
                labels.append(lbl[ntype])
    model.train()
    if return_label:
        preds = th.cat(preds)
        labels = th.cat(labels)
        return preds, labels
    else:
        return th.cat(preds)
