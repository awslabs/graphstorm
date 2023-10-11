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

    Inferrer wrapper for node classification and regression.
"""
import time
import logging

from .graphstorm_infer import GSInferrer
from ..model.utils import save_shuffled_node_embeddings
from ..model.utils import save_node_prediction_results
from ..model.utils import NodeIDShuffler
from ..model.gnn import do_full_graph_inference
from ..model.node_gnn import node_mini_batch_gnn_predict
from ..model.node_gnn import node_mini_batch_predict

from ..utils import sys_tracker, get_rank, barrier

class GSgnnNodePredictionInferrer(GSInferrer):
    """ Node classification/regression inferrer.

    This is a high-level inferrer wrapper that can be used directly
    to do node classification/regression model inference.

    Parameters
    ----------
    model : GSgnnNodeModel
        The GNN model for node prediction.
    """

    def infer(self, loader, save_embed_path, save_prediction_path=None,
              use_mini_batch_infer=False,
              node_id_mapping_file=None,
              return_proba=True,
              save_embed_format="pytorch"):
        """ Do inference

        The inference does three things:

        1. (Optional) Evaluate the model performance on a test set if given.
        2. Generate node embeddings.
        3. Comput inference results for nodes with target node type.

        Parameters
        ----------
        loader : GSNodeDataLoader
            The mini-batch sampler for node prediction task.
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
                "A label field must be provided for node classification " \
                "or regression inference when evaluation is required."

        sys_tracker.check('start inferencing')
        self._model.eval()
        ntypes = loader.data.eval_ntypes

        if use_mini_batch_infer:
            res = node_mini_batch_gnn_predict(self._model, loader, return_proba,
                                              return_label=do_eval)
            preds = res[0]
            embs = res[1]
            labels = res[2] if do_eval else None
        else:
            embs = do_full_graph_inference(self._model, loader.data, fanout=loader.fanout,
                                           task_tracker=self.task_tracker)
            res = node_mini_batch_predict(self._model, embs, loader, return_proba,
                                          return_label=do_eval)
            preds = res[0]
            labels = res[1] if do_eval else None

            if save_embed_path is not None:
                # Only embeddings of the target nodes will be saved.
                embs = {ntype: embs[ntype][loader.target_nidx[ntype]] \
                        for ntype in ntypes}
            else:
                # release embs
                del embs
        sys_tracker.check('compute embeddings')

        # do evaluation first
        # do evaluation if any
        if do_eval:
            # iterate all the target ntypes
            for ntype in ntypes:
                pred = preds[ntype]
                label = labels[ntype]
                test_start = time.time()
                val_score, test_score = self.evaluator.evaluate(pred, pred, label, label, 0)
                sys_tracker.check('run evaluation')
                if get_rank() == 0:
                    self.log_print_metrics(val_score=val_score,
                                        test_score=test_score,
                                        dur_eval=time.time() - test_start,
                                        total_steps=0)

        nid_shuffler = None
        g = loader.data.g
        if save_embed_path is not None:
            # We are going to save the node embeddings of loader.target_nidx[ntype]
            nid_shuffler = NodeIDShuffler(g, node_id_mapping_file, ntypes) \
                if node_id_mapping_file else None
            shuffled_embs = {}
            for ntype in ntypes:
                if get_rank() == 0:
                    logging.info("save embeddings pf {ntype} to %s", save_embed_path)

                # only save embeddings of target_nidx
                assert ntype in embs, \
                    f"{ntype} is not in the set of evaluation ntypes {ntypes}"
                emb_nids = loader.target_nidx[ntype]

                if node_id_mapping_file is not None:
                    emb_nids = nid_shuffler.shuffle_nids(ntype, emb_nids)
                shuffled_embs[ntype] = (embs[ntype], emb_nids)
            save_shuffled_node_embeddings(shuffled_embs, save_embed_path, save_embed_format)

            barrier()
            sys_tracker.check('save embeddings')

        if save_prediction_path is not None:
            # save_embed_path may be None. In that case, we need to init nid_shuffler
            if nid_shuffler is None:
                nid_shuffler = NodeIDShuffler(g, node_id_mapping_file, list(preds.keys)) \
                    if node_id_mapping_file else None
            shuffled_preds = {}
            for ntype, pred in preds.items():
                assert ntype in ntypes, \
                    f"{ntype} is not in the set of evaluation ntypes {loader.data.eval_ntypes}"

                pred_nids = loader.target_nidx[ntype]
                if node_id_mapping_file is not None:
                    pred_nids = nid_shuffler.shuffle_nids(ntype, pred_nids)
                shuffled_preds[ntype] = (pred, pred_nids)

            save_node_prediction_results(shuffled_preds, save_prediction_path)
        barrier()
        sys_tracker.check('save predictions')
