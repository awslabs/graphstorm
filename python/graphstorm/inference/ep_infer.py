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

    Inferrer wrapper for edge classification and regression.
"""
import os
import time

from .graphstorm_infer import GSInferrer
from ..model.utils import save_embeddings as save_gsgnn_embeddings
from ..model.utils import save_edge_prediction_results
from ..model.utils import shuffle_nids
from ..model.gnn import do_full_graph_inference
from ..model.edge_gnn import edge_mini_batch_predict, edge_mini_batch_gnn_predict

from ..utils import sys_tracker, get_world_size, get_rank, barrier, create_dist_tensor

class GSgnnEdgePredictionInferrer(GSInferrer):
    """ Edge classification/regression inferrer.

    This is a high-level inferrer wrapper that can be used directly
    to do edge classification/regression model inference.

    Parameters
    ----------
    model : GSgnnNodeModel
        The GNN model for node prediction.
    """

    # pylint: disable=unused-argument
    def infer(self, loader, save_embed_path, save_prediction_path=None,
            use_mini_batch_infer=False,
            node_id_mapping_file=None,
            edge_id_mapping_file=None,
            return_proba=True,
            save_embed_format="pytorch"):
        """ Do inference

        The inference can do three things:

        1. (Optional) Evaluate the model performance on a test set if given.
        2. Generate node embeddings.
        3. Comput inference results for edges with target edge type.

        Parameters
        ----------
        loader : GSEdgeDataLoader
            The mini-batch sampler for edge prediction task.
        save_embed_path : str
            The path where the GNN embeddings will be saved.
        save_prediction_path : str
            The path where the prediction results will be saved.
        use_mini_batch_infer : bool
            Whether or not to use mini-batch inference.
        node_id_mapping_file: str
            Path to the file storing node id mapping generated by the
            graph partition algorithm.
        return_proba: bool
            Whether to return all the predictions or the maximum prediction.
        save_embed_format : str
            Specify the format of saved embeddings.
        """
        do_eval = self.evaluator is not None
        if do_eval:
            assert loader.data.labels is not None, \
                "A label field must be provided for edge classification " \
                "or regression inference when evaluation is required."

        if use_mini_batch_infer:
            assert save_embed_path is None, \
                "Unable to save the node embeddings when using mini batch inference." \
                "It is not guaranteed that mini-batch prediction will cover all the nodes."
        sys_tracker.check('start inferencing')
        self._model.eval()

        if use_mini_batch_infer:
            res = edge_mini_batch_gnn_predict(self._model,
                                              loader,
                                              return_proba,
                                              return_label=do_eval)
        else:
            embs = do_full_graph_inference(self._model, loader.data, fanout=loader.fanout,
                                           task_tracker=self.task_tracker)
            sys_tracker.check('compute embeddings')
            res = edge_mini_batch_predict(self._model, embs, loader, return_proba,
                                          return_label=do_eval)
        preds = res[0]
        labels = res[1] if do_eval else None
        sys_tracker.check('compute prediction')

        # Only save the embeddings related to target edge types.
        infer_data = loader.data
        # TODO support multiple etypes
        assert len(infer_data.eval_etypes) == 1, \
            "GraphStorm only support single target edge type for training and inference"

        # do evaluation first
        if do_eval:
            test_start = time.time()
            pred = preds[infer_data.eval_etypes[0]]
            label = labels[infer_data.eval_etypes[0]] if labels is not None else None
            val_score, test_score = self.evaluator.evaluate(pred, pred, label, label, 0)
            sys_tracker.check('run evaluation')
            if get_rank() == 0:
                self.log_print_metrics(val_score=val_score,
                                       test_score=test_score,
                                       dur_eval=time.time() - test_start,
                                       total_steps=0)
        device = self.device
        if save_embed_path is not None:
            target_ntypes = set()
            for etype in infer_data.eval_etypes:
                target_ntypes.add(etype[0])
                target_ntypes.add(etype[2])

            # The order of the ntypes must be sorted
            embs = {ntype: embs[ntype] for ntype in sorted(target_ntypes)}
            save_gsgnn_embeddings(save_embed_path, embs, get_rank(),
                get_world_size(),
                device=device,
                node_id_mapping_file=node_id_mapping_file,
                save_embed_format=save_embed_format)
            barrier()
            sys_tracker.check('save embeddings')

        if save_prediction_path is not None:
            g = loader.data.g
            shuffled_preds = {}
            for etype, pred in preds.items():
                pred_src_nids, pred_dst_nids = \
                    g.find_edges(loader.target_eidx[etype], etype=etype)

                if node_id_mapping_file is not None:
                    pred_src_nids = shuffle_nids(g, etype[0], pred_src_nids,
                                                node_id_mapping_file, get_rank())
                    pred_dst_nids = shuffle_nids(g, etype[2], pred_dst_nids,
                                                node_id_mapping_file, get_rank())
                shuffled_preds[etype] = (pred, pred_src_nids, pred_dst_nids)
            save_edge_prediction_results(shuffled_preds, save_prediction_path)

        barrier()
        sys_tracker.check('save predictions')
