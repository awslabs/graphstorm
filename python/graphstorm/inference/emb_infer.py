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
from ..model.gnn import do_full_graph_inference, do_mini_batch_inference

from ..utils import sys_tracker, get_rank, get_world_size, barrier


class GSgnnEmbGenInferer(GSInferrer):
    """ Embedding Generation inferrer.

    This is a high-level inferrer wrapper that can be used directly
    to generate embedding for inferer.

    Parameters
    ----------
    model : GSgnnNodeModel
        The GNN model with different task.
    """

    def nc_emb(self, g, model, use_mini_batch_infer, fanout):
        """ Embedding Generation for node task.
        It will only generate embeddings on the target node type.

        Parameters
        ----------
        g: GSgnnData
            The GraphStorm dataset
        model : GSgnnNodeModel
            The GNN model on edge prediction/classification task
        use_mini_batch_infer : bool
            Whether to use mini-batch inference when computing node embeddings.
        fanout: list of int
            The fanout of each GNN layers used in inference.
        """
        if use_mini_batch_infer:
            embs = do_mini_batch_inference(model, g, fanout=fanout,
                                           edge_mask=None,
                                           task_tracker=self.task_tracker,
                                           infer_ntypes=g.infer_idxs)
        else:
            embs = do_full_graph_inference(model, g, fanout=fanout,
                                           task_tracker=self.task_tracker)
        return embs

    def ec_emb(self, g, model, use_mini_batch_infer, fanout):
        """ Embedding Generation for edge task.
        It will only generate embeddings on the target node type
        defined in the target edge type.

        Parameters
        ----------
        g: GSgnnData
            The GraphStorm dataset
        model : GSgnnNodeModel
            The GNN model on edge prediction/classification task
        use_mini_batch_infer : bool
            Whether to use mini-batch inference when computing node embeddings.
        fanout: list of int
            The fanout of each GNN layers used in inference.
        """
        infer_ntypes = []
        for etype in g.infer_idxs:
            if etype[0] not in infer_ntypes:
                infer_ntypes.append(etype[0])
            if etype[2] not in infer_ntypes:
                infer_ntypes.append(etype[2])

        if use_mini_batch_infer:
            embs = do_mini_batch_inference(model, g, fanout=fanout,
                                           edge_mask=None,
                                           task_tracker=self.task_tracker,
                                           infer_ntypes=infer_ntypes)
        else:
            embs = do_full_graph_inference(model, g, fanout=fanout,
                                           task_tracker=self.task_tracker)
        return embs

    def lp_emb(self, g, model, use_mini_batch_infer, fanout):
        """ Embedding Generation for link prediction task.
        It will only generate embeddings on whole graph.

        Parameters
        ----------
        g: GSgnnData
            The GraphStorm dataset
        model : GSgnnNodeModel
            The GNN model on edge prediction/classification task
        use_mini_batch_infer : bool
            Whether to use mini-batch inference when computing node embeddings.
        fanout: list of int
            The fanout of each GNN layers used in inference.
        """
        if use_mini_batch_infer:
            embs = do_mini_batch_inference(model, g, fanout=fanout,
                                           edge_mask=None,
                                           task_tracker=self.task_tracker)
        else:
            embs = do_full_graph_inference(model, g, fanout=fanout,
                                           edge_mask=None,
                                           task_tracker=self.task_tracker)
        return embs

    def infer(self, data, task_type, save_embed_path, eval_fanout,
            use_mini_batch_infer=False,
            node_id_mapping_file=None,
            save_embed_format="pytorch"):
        """ Do Embedding Generating

        Generate node embeddings and save into disk.

        Parameters
        ----------
        data: GSgnnData
            The GraphStorm dataset
        task_type : str
            task_type must be one of graphstorm builtin task types
        save_embed_path : str
            The path where the GNN embeddings will be saved.
        use_mini_batch_infer : bool
            Whether to use mini-batch inference when computing node embeddings.
        node_id_mapping_file: str
            Path to the file storing node id mapping generated by the
            graph partition algorithm.
        save_embed_format : str
            Specify the format of saved embeddings.
        """

        device = self.device

        assert save_embed_path is not None, \
            "It requires save embed path for gs_gen_node_embedding"

        sys_tracker.check('start embedding generation')
        self._model.eval()

        if task_type == BUILTIN_TASK_LINK_PREDICTION:
            embs = self.lp_emb(data, self._model, use_mini_batch_infer, eval_fanout)
        elif task_type in {BUILTIN_TASK_NODE_REGRESSION, BUILTIN_TASK_NODE_CLASSIFICATION}:
            embs = self.nc_emb(data, self._model, use_mini_batch_infer, eval_fanout)
        elif task_type in {BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION}:
            embs = self.ec_emb(data, self._model, use_mini_batch_infer, eval_fanout)
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
