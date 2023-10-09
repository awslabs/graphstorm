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

    Inferrer wrapper for embedding generation.
"""
import logging
from graphstorm.config import  (BUILTIN_TASK_NODE_CLASSIFICATION,
                                BUILTIN_TASK_NODE_REGRESSION,
                                BUILTIN_TASK_EDGE_CLASSIFICATION,
                                BUILTIN_TASK_EDGE_REGRESSION,
                                BUILTIN_TASK_LINK_PREDICTION)
from .graphstorm_infer import GSInferrer
from ..model.utils import save_embeddings as save_gsgnn_embeddings
from ..model.utils import save_relation_embeddings
from ..model.edge_decoder import LinkPredictDistMultDecoder
from ..model.gnn import do_full_graph_inference, do_mini_batch_inference
from ..model.node_gnn import node_mini_batch_gnn_predict

from ..utils import sys_tracker, get_rank, get_world_size, barrier, create_dist_tensor


class GSgnnEmbGenInferer(GSInferrer):
    """ Embedding Generation inffer inferrer.

    This is a high-level inferrer wrapper that can be used directly
    to generate embedding for inferer.

    Parameters
    ----------
    model : GSgnnNodeModel
        The GNN model with different task.
    """

    def infer(self, data, task_type, save_embed_path, loader,
            use_mini_batch_infer=False,
            node_id_mapping_file=None,
            return_proba=True,
            save_embed_format="pytorch"):
        """ Do Embedding Generating

        Generate node embeddings and save.

        Parameters
        ----------
        data: GSgnnData
            The GraphStorm dataset
        task_type : str
            task_type must be one of graphstorm builtin task types
        save_embed_path : str
            The path where the GNN embeddings will be saved.
        loader : GSEdgeDataLoader/GSNodeDataLoader
            The mini-batch sampler for built-in graphstorm task.
        use_mini_batch_infer : bool
            Whether or not to use mini-batch inference when computing node embedings.
        node_id_mapping_file: str
            Path to the file storing node id mapping generated by the
            graph partition algorithm.
        save_embed_format : str
            Specify the format of saved embeddings.
        """

        device = self.device
        # deal with uninitialized case first
        if use_mini_batch_infer and \
                task_type in {BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION}:
            assert save_embed_path is None, \
                "Unable to save the node embeddings when using mini batch inference " \
                "when doing edge task." \
                "It is not guaranteed that mini-batch prediction will cover all the nodes."

        if task_type in {BUILTIN_TASK_NODE_REGRESSION, BUILTIN_TASK_NODE_CLASSIFICATION}:
            assert len(loader.data.eval_ntypes) == 1, \
                "GraphStorm only support single target node type for training and inference"

        assert save_embed_path is not None

        sys_tracker.check('start embedding generation')
        self._model.eval()

        if task_type == BUILTIN_TASK_LINK_PREDICTION:
            # for embedding generation, it is preferred to use full graph
            if use_mini_batch_infer:
                embs = do_mini_batch_inference(self._model, data, fanout=loader.fanout,
                                               edge_mask=None,
                                               task_tracker=self.task_tracker)
            else:
                embs = do_full_graph_inference(self._model, data, fanout=loader.fanout,
                                               edge_mask=None,
                                               task_tracker=self.task_tracker)
        elif task_type in {BUILTIN_TASK_NODE_REGRESSION, BUILTIN_TASK_NODE_CLASSIFICATION}:
            # only generate embeddings on the target node type
            ntype = loader.data.eval_ntypes[0]
            if use_mini_batch_infer:
                inter_embs = node_mini_batch_gnn_predict(self._model, loader, return_proba,
                                                  return_label=False)[1]
                inter_embs = {ntype: inter_embs[ntype]} if isinstance(inter_embs, dict) \
                    else {ntype: inter_embs}
                g = loader.data.g
                ntype_emb = create_dist_tensor((g.num_nodes(ntype), inter_embs[ntype].shape[1]),
                                               dtype=inter_embs[ntype].dtype,
                                               name=f'gen-emb-{ntype}',
                                               part_policy=g.get_node_partition_policy(ntype),
                                               persistent=True)
                ntype_emb[loader.target_nidx[ntype]] = inter_embs[ntype]
                embs = {ntype: ntype_emb}
            else:
                embs = do_full_graph_inference(self._model, data, fanout=loader.fanout,
                                               task_tracker=self.task_tracker)
                ntype_emb = embs[ntype]
                embs = {ntype: ntype_emb}
        elif task_type in {BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION}:
            # Currently it is not allowed to do mini-batch inference
            # and save embedding on edge tasks
            embs = do_full_graph_inference(self._model, loader.data, fanout=loader.fanout,
                                           task_tracker=self.task_tracker)
            target_ntypes = set()
            for etype in loader.data.eval_etypes:
                target_ntypes.add(etype[0])
                target_ntypes.add(etype[2])

            embs = {ntype: embs[ntype] for ntype in sorted(target_ntypes)}
        else:
            raise TypeError("Not supported for task type: ", task_type)

        if get_rank() == 0:
            logging.info("save embeddings to %s", save_embed_path)

        save_gsgnn_embeddings(save_embed_path, embs, get_rank(),
            get_world_size(),
            device=device,
            node_id_mapping_file=node_id_mapping_file,
            save_embed_format=save_embed_format)
        barrier()
        sys_tracker.check('save embeddings')

        # save relation embedding if any
        if get_rank() == 0:
            decoder = self._model.decoder
            if isinstance(decoder, LinkPredictDistMultDecoder):
                if save_embed_path is not None:
                    save_relation_embeddings(save_embed_path, decoder)
