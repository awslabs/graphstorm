"""GNN model for edge prediction s in GraphStorm
"""
import torch as th
import dgl

from .gnn import GSgnnModel

class GSgnnEdgeModel(GSgnnModel):
    """ GraphStorm GNN model for edge prediction tasks

    Parameters
    ----------
    alpha_l2norm : float
        The alpha for L2 normalization.
    """
    def __init__(self, alpha_l2norm):
        super(GSgnnEdgeModel, self).__init__()
        self.alpha_l2norm = alpha_l2norm

    def forward(self, blocks, batch_graph, input_feats, input_nodes, labels):
        """ The forward function for edge prediction.

        Parameters
        ----------
        blocks : list of DGLBlock
            The message passing graph for computing GNN embeddings.
        batch_graph : a DGLGraph
            The graph where we run edge classification.
        input_feats : dict of Tensor
            The input features of the message passing graphs.
        input_nodes : dict of Tensor
            The input nodes of the message passing graphs.
        labels: dict of Tensor
            The labels of the predicted edges.

        Returns
        -------
        The loss of prediction.
        """
        alpha_l2norm = self.alpha_l2norm
        gnn_embs = self.compute_embed_step(blocks, input_feats, input_nodes)
        # TODO(zhengda) we only support prediction on one edge type now
        assert len(labels) == 1, "We only support prediction on one edge type for now."
        target_etype = list(labels.keys())[0]

        logits = self.decoder(batch_graph, gnn_embs)
        pred_loss = self.loss_func(logits, labels[target_etype])

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(pred_loss.device)
        # L2 regularization of dense parameters
        for d_para in self.get_dense_params():
            reg_loss += d_para.square().sum()

        # weighted addition to the total loss
        return pred_loss + alpha_l2norm * reg_loss

    def predict(self, blocks, batch_graph, input_feats, input_nodes):
        """ The forward function for edge prediction.

        Parameters
        ----------
        blocks : list of DGLBlock
            The message passing graph for computing GNN embeddings.
        batch_graph : a DGLGraph
            The graph where we run edge classification.
        input_feats : dict
            The input features of the message passing graphs.
        input_nodes : dict
            The input nodes of the message passing graphs.

        Returns
        -------
        The prediction results.
        """
        gnn_embs = self.compute_embed_step(blocks, input_feats, input_nodes)
        return self.decoder.predict(batch_graph, gnn_embs)

def edge_mini_batch_gnn_predict(model, loader, return_label=False):
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
    Tensor : labels if return_labels is True
    """
    device = model.device
    data = loader.data
    preds = []
    labels = []
    model.eval()
    with th.no_grad():
        for input_nodes, batch_graph, blocks in loader:
            input_feats = data.get_node_feats(input_nodes, device)
            blocks = [block.to(device) for block in blocks]
            pred = model.predict(blocks, batch_graph, input_feats, input_nodes)
            preds.append(pred.cpu())

            if return_label:
                # retrieving seed edge id from the graph to find labels
                # TODO(zhengda) expand code for multiple edge types
                assert len(batch_graph.etypes) == 1
                predict_etype = batch_graph.canonical_etypes[0]
                # TODO(zhengda) the data loader should return labels directly.
                seeds = batch_graph.edges[predict_etype].data[dgl.EID]
                lbl = data.get_labels({predict_etype: seeds})
                assert len(lbl) == 1
                labels.append(lbl[predict_etype])
    model.train()
    preds = th.cat(preds)
    if return_label:
        return preds, th.cat(labels)
    else:
        return preds

def edge_mini_batch_predict(model, emb, loader, return_label=False):
    """ Perform mini-batch prediction.

    This function usually follows full-grain GNN embedding inference. After having
    the GNN embeddings, we need to perform mini-batch computation to make predictions
    on the GNN embeddings.

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
    # find the target src and dst ntypes
    model.eval()
    decoder = model.decoder
    data = loader.data
    with th.no_grad():
        # save preds and labels together in order not to shuffle
        # the order when gather tensors from other trainers
        preds_list = []
        labels_list = []
        device = model.device
        for input_nodes, batch_graph, _ in loader:
            assert len(batch_graph.etypes) == 1
            etype = batch_graph.canonical_etypes[0]
            batch_embs = {}
            for ntype, in_nodes in input_nodes.items():
                batch_embs[ntype] = emb[ntype][in_nodes].to(device)
            batch_graph = batch_graph.to(device)
            # TODO(zhengda) how to deal with edge features?
            preds_list.append(decoder.predict(batch_graph, batch_embs))
            # TODO(zhengda) we need to have the data loader reads everything,
            # instead of reading labels here.
            if return_label:
                labels = data.get_labels({etype: batch_graph.edges[etype].data[dgl.EID]})
                labels_list.append(labels[etype])
        # can't use torch.stack here becasue the size of last tensor is different
        preds = th.cat(preds_list)
    th.distributed.barrier()

    model.train()
    if return_label:
        return preds, th.cat(labels_list)
    else:
        return preds
