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

    Utils for graph computation
"""

import time
import logging

from .gnn import GSgnnModel
from .embed import compute_node_input_embeddings
from .lp_gnn import GSgnnLinkPredictionModelInterface
from .gnn_encoder_base import dist_minibatch_inference
from ..dataloading.dataset import prepare_batch_input
from ..utils import get_rank, barrier

def do_mini_batch_inference(model, data, batch_size=1024,
                            fanout=None, edge_mask=None, infer_ntypes=None,
                            task_tracker=None):
    """ Do mini batch inference

    It may use some of the edges indicated by `edge_mask` to compute GNN embeddings.

    Parameters
    ----------
    model: torch model
        GNN model
    data : GSgnnData
        The GraphStorm dataset
    batch_size : int
        The batch size for inferencing a GNN layer
    fanout: list of int
        The fanout for computing the GNN embeddings in a GNN layer.
    edge_mask : str
        The edge mask that indicates what edges are used to compute GNN embeddings.
    infer_ntypes: list of str
        Node types that need to compute node embeddings.
    task_tracker: GSTaskTrackerAbc
        Task tracker

    Returns
    -------
    dict of th.Tensor : node embeddings.
    """
    if get_rank() == 0:
        logging.debug("Perform mini-batch inference on the full graph.")
    t1 = time.time() # pylint: disable=invalid-name
    barrier()
    if model.gnn_encoder is None:
        # Only graph aware but not GNN models
        embeddings = compute_node_input_embeddings(data.g,
                                                   batch_size,
                                                   model.node_input_encoder,
                                                   task_tracker=task_tracker,
                                                   feat_field=data.node_feat_field,
                                                   target_ntypes=infer_ntypes)
        model.eval()
    elif model.node_input_encoder.require_cache_embed():
        # If the input encoder has heavy computation, we should compute
        # the embeddings and cache them.
        input_embeds = compute_node_input_embeddings(data.g,
                                                     batch_size,
                                                     model.node_input_encoder,
                                                     task_tracker=task_tracker,
                                                     feat_field=data.node_feat_field)
        model.eval()
        device = model.gnn_encoder.device
        def get_input_embeds(input_nodes):
            if not isinstance(input_nodes, dict):
                assert len(data.g.ntypes) == 1
                input_nodes = {data.g.ntypes[0]: input_nodes}
            return {ntype: input_embeds[ntype][ids].to(device) \
                    for ntype, ids in input_nodes.items()}
        embeddings = dist_minibatch_inference(data.g,
                                                model.gnn_encoder,
                                                get_input_embeds,
                                                batch_size, fanout,
                                                edge_mask=edge_mask,
                                                target_ntypes=infer_ntypes,
                                                task_tracker=task_tracker)
    else:
        model.eval()
        device = model.gnn_encoder.device
        def get_input_embeds(input_nodes):
            if not isinstance(input_nodes, dict):
                assert len(data.g.ntypes) == 1
                input_nodes = {data.g.ntypes[0]: input_nodes}
            feats = prepare_batch_input(data.g, input_nodes, dev=device,
                                        feat_field=data.node_feat_field)
            return model.node_input_encoder(feats, input_nodes)
        embeddings = dist_minibatch_inference(data.g,
                                                model.gnn_encoder,
                                                get_input_embeds,
                                                batch_size, fanout,
                                                edge_mask=edge_mask,
                                                target_ntypes=infer_ntypes,
                                                task_tracker=task_tracker)

    if isinstance(model, GSgnnLinkPredictionModelInterface):
        # If it is link prediction, try to normalize the node embeddings
        embeddings = model.normalize_node_embs(embeddings)

    model.train()
    if get_rank() == 0:
        logging.debug("computing GNN embeddings: %.4f seconds", time.time() - t1)
    return embeddings

def do_full_graph_inference(model, data, batch_size=1024, fanout=None, edge_mask=None,
                            task_tracker=None):
    """ Do fullgraph inference

    It may use some of the edges indicated by `edge_mask` to compute GNN embeddings.

    Parameters
    ----------
    model: torch model
        GNN model
    data : GSgnnData
        The GraphStorm dataset
    batch_size : int
        The batch size for inferencing a GNN layer
    fanout: list of int
        The fanout for computing the GNN embeddings in a GNN layer.
    edge_mask : str
        The edge mask that indicates what edges are used to compute GNN embeddings.
    task_tracker: GSTaskTrackerAbc
        Task tracker

    Returns
    -------
    dict of th.Tensor : node embeddings.
    """
    if get_rank() == 0:
        logging.debug("Perform full-graph inference with batch size %d and fanout %s.",
                      batch_size, str(fanout))
    assert isinstance(model, GSgnnModel) or type(model).__name__ == 'GLEM',\
        "Only GSgnnModel and GLEM support full-graph inference."
    t1 = time.time() # pylint: disable=invalid-name
    # full graph evaluation
    barrier()
    if model.gnn_encoder is None:
        # Only graph aware but not GNN models
        embeddings = compute_node_input_embeddings(data.g,
                                                   batch_size,
                                                   model.node_input_encoder,
                                                   task_tracker=task_tracker,
                                                   feat_field=data.node_feat_field)
        model.eval()
    elif model.node_input_encoder.require_cache_embed():
        # If the input encoder has heavy computation, we should compute
        # the embeddings and cache them.
        input_embeds = compute_node_input_embeddings(data.g,
                                                     batch_size,
                                                     model.node_input_encoder,
                                                     task_tracker=task_tracker,
                                                     feat_field=data.node_feat_field)
        model.eval()
        device = model.gnn_encoder.device
        def get_input_embeds(input_nodes):
            if not isinstance(input_nodes, dict):
                assert len(data.g.ntypes) == 1
                input_nodes = {data.g.ntypes[0]: input_nodes}
            res = {}
            # If the input node layer doesn't generate embeddings for a node type,
            # we ignore it. This behavior is the same as reading node features below.
            for ntype, ids in input_nodes.items():
                if ntype in input_embeds:
                    res[ntype] = input_embeds[ntype][ids].to(device)
            return res
        embeddings = model.gnn_encoder.dist_inference(data.g, get_input_embeds,
                                                    batch_size, fanout, edge_mask=edge_mask,
                                                    task_tracker=task_tracker)
    else:
        model.eval()
        device = model.gnn_encoder.device
        def get_input_embeds(input_nodes):
            if not isinstance(input_nodes, dict):
                assert len(data.g.ntypes) == 1
                input_nodes = {data.g.ntypes[0]: input_nodes}
            feats = prepare_batch_input(data.g, input_nodes, dev=device,
                                        feat_field=data.node_feat_field)
            return model.node_input_encoder(feats, input_nodes)

        embeddings = model.gnn_encoder.dist_inference(data.g, get_input_embeds,
                                                    batch_size, fanout, edge_mask=edge_mask,
                                                    task_tracker=task_tracker)

    if isinstance(model, GSgnnLinkPredictionModelInterface):
        # If it is link prediction, try to normalize the node embeddings
        # Called when model.eval()
        embeddings = model.normalize_node_embs(embeddings)
    model.train()

    if get_rank() == 0:
        logging.debug("computing GNN embeddings: %.4f seconds", time.time() - t1)
    return embeddings
