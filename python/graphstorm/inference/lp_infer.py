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

    Inferrer wrapper for link predicion.
"""
import time

from .graphstorm_infer import GSInferrer
from ..model.utils import save_embeddings as save_gsgnn_embeddings
from ..model.utils import save_relation_embeddings
from ..model.edge_decoder import LinkPredictDistMultDecoder
from ..model.gnn import do_full_graph_inference, do_mini_batch_inference
from ..model.lp_gnn import lp_mini_batch_predict

from ..utils import sys_tracker, get_world_size, barrier

class GSgnnLinkPredictionInferrer(GSInferrer):
    """ Link prediction inferrer.

    This is a highlevel inferrer wrapper that can be used directly
    to do link prediction model inference.

    Parameters
    ----------
    model : GSgnnNodeModel
        The GNN model for node prediction.
    rank : int
        The rank.
    """

    # TODO(zhengda) We only support full-graph inference for now.
    def infer(self, data, loader, save_embed_path,
            edge_mask_for_gnn_embeddings='train_mask',
            use_mini_batch_infer=False,
            node_id_mapping_file=None):
        """ Do inference

        The inference can do two things:
        1. (Optional) Evaluate the model performance on a test set if given
        2. Generate node embeddings

        Parameters
        ----------
        data: GSgnnData
            The GraphStorm dataset
        loader : GSgnnLinkPredictionTestDataLoader
            The mini-batch sampler for link prediction task.
        save_embed_path : str
            The path where the GNN embeddings will be saved.
        edge_mask_for_gnn_embeddings : str
            The mask that indicates the edges used for computing GNN embeddings. By default,
            the dataloader uses the edges in the training graphs to compute GNN embeddings to
            avoid information leak for link prediction.
        use_mini_batch_infer : bool
            Whether or not to use mini-batch inference when computing node embedings.
        node_id_mapping_file: str
            Path to the file storing node id mapping generated by the
            graph partition algorithm.
        """
        sys_tracker.check('start inferencing')
        self._model.eval()
        if use_mini_batch_infer:
            embs = do_mini_batch_inference(self._model, data, fanout=loader.fanout,
                                           edge_mask=edge_mask_for_gnn_embeddings,
                                           task_tracker=self.task_tracker)
        else:
            embs = do_full_graph_inference(self._model, data, fanout=loader.fanout,
                                           edge_mask=edge_mask_for_gnn_embeddings,
                                           task_tracker=self.task_tracker)
        sys_tracker.check('compute embeddings')
        device = self.device
        if save_embed_path is not None:
            save_gsgnn_embeddings(save_embed_path, embs, self.rank,
                get_world_size(),
                device=device,
                node_id_mapping_file=node_id_mapping_file)
        barrier()
        sys_tracker.check('save embeddings')

        if self.evaluator is not None:
            test_start = time.time()
            test_rankings = lp_mini_batch_predict(self._model, embs, loader, device)
            val_mrr, test_mrr = self.evaluator.evaluate(None, test_rankings, 0)
            sys_tracker.check('run evaluation')
            if self.rank == 0:
                self.log_print_metrics(val_score=val_mrr,
                                       test_score=test_mrr,
                                       dur_eval=time.time() - test_start,
                                       total_steps=0)

        barrier()
        # save relation embedding if any
        if self.rank == 0:
            decoder = self._model.decoder
            if isinstance(decoder, LinkPredictDistMultDecoder):
                if save_embed_path is not None:
                    save_relation_embeddings(save_embed_path, decoder)
