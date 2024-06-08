"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    GNN inferer for multi-task learning in GraphStorm
"""
import os
import time
import logging
import torch as th

from ..config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                      BUILTIN_TASK_NODE_REGRESSION,
                      BUILTIN_TASK_EDGE_CLASSIFICATION,
                      BUILTIN_TASK_EDGE_REGRESSION,
                      BUILTIN_TASK_RECONSTRUCT_NODE_FEAT)
from .graphstorm_infer import GSInferrer
from ..model.utils import save_full_node_embeddings as save_gsgnn_embeddings
from ..model.utils import (save_node_prediction_results,
                           save_edge_prediction_results,
                           save_relation_embeddings)
from ..model.utils import NodeIDShuffler
from ..model import do_full_graph_inference, do_mini_batch_inference
from ..model.multitask_gnn import multi_task_mini_batch_predict
from ..model.node_gnn import run_node_mini_batch_predict
from ..model.lp_gnn import run_lp_mini_batch_predict
from ..model.gnn_encoder_base import GSgnnGNNEncoderInterface

from ..model.edge_decoder import LinkPredictDistMultDecoder

from ..utils import sys_tracker, get_rank, barrier

class GSgnnMultiTaskLearningInferer(GSInferrer):
    """ Multi task inferrer.

    This is a high-level inferrer wrapper that can be used directly
    to do multi task model inference.

    Parameters
    ----------
    model : GSgnnMultiTaskModel
        The GNN model for prediction.
    """

    # pylint: disable=unused-argument
    def infer(self, data,
              mt_test_loader,
              save_embed_path=None,
              save_prediction_path=None,
              use_mini_batch_infer=False,
              node_id_mapping_file=None,
              edge_id_mapping_file=None,
              return_proba=True,
              save_embed_format="pytorch",
              infer_batch_size=1024):
        """ Do inference

        The inference can do three things:

        1. Generate node embeddings.
        2. Comput inference results for each tasks
        3. (Optional) Evaluate the model performance on a test set if given.

        Parameters
        ----------
        data: GSgnnData
            Graph data.
        mt_test_loader:  tuple of GSgnnMultiTaskDataLoaders
            A tuple of mini-batch samplers for inference.
            In format of (test_dataloader, lp_test_dataloader,
            recon_nfeat_test_dataloader). The second dataloader
            contains test dataloaders for link predicction tasks.
            The third dataloader contains test dataloaders for
            node feature reconstruction tasks. When evaluating
            these tasks, different message passing strategies
            will be applied. The first dataloader contains
            all other dataloaders.
        save_embed_path: str
            The path to save the node embeddings.
        save_prediction_path: str
            The path to save the prediction resutls.
        use_mini_batch_infer: bool
            Whether or not to use mini-batch inference when computing node embedings.
        node_id_mapping_file: str
            Path to the file storing node id mapping generated by the
            graph partition algorithm.
        edge_id_mapping_file: str
            Path to the file storing edge id mapping generated by the
            graph partition algorithm.
        return_proba: bool
            Whether to return all the predictions or the maximum prediction.
        save_embed_format : str
            Specify the format of saved embeddings.
        infer_batch_size: int
            Specify the inference batch size when computing node embeddings
            with mini batch inference.
        """
        do_eval = self.evaluator is not None
        sys_tracker.check('start inferencing')
        self._model.eval()
        mt_loader, lp_test_loader, recon_nfeat_test_loader = mt_test_loader

        fanout = None
        for task_fanout in mt_loader.fanout:
            if task_fanout is not None:
                fanout = task_fanout
                break

        def gen_embs():
            # Generate node embeddings.
            if use_mini_batch_infer:
                embs = do_mini_batch_inference(
                    self._model, data, batch_size=infer_batch_size,
                    fanout=fanout, task_tracker=self.task_tracker)
            else:
                embs = do_full_graph_inference(
                    self._model, data, fanout=fanout,
                    task_tracker=self.task_tracker)
            return embs
        embs = gen_embs()
        sys_tracker.check('compute embeddings')
        device = self.device

        pre_results = \
            multi_task_mini_batch_predict(self._model,
                                          emb=embs,
                                          loader=mt_loader,
                                          device=device,
                                          return_proba=return_proba,
                                          return_label=do_eval)

        if lp_test_loader is not None:
            # We also need to compute test scores for link prediction tasks.
            dataloaders = lp_test_loader.dataloaders
            task_infos = lp_test_loader.task_infos

            with th.no_grad():
                for dataloader, task_info in zip(dataloaders, task_infos):
                    if dataloader is None:
                        pre_results[task_info.task_id] = None

                    if use_mini_batch_infer:
                        lp_test_embs = do_mini_batch_inference(
                            self._model, data, batch_size=infer_batch_size,
                            fanout=fanout,
                            edge_mask=task_info.task_config.train_mask,
                            task_tracker=self.task_tracker)
                    else:
                        lp_test_embs = do_full_graph_inference(
                            self._model, data, fanout=fanout,
                            edge_mask=task_info.task_config.train_mask,
                            task_tracker=self.task_tracker)
                    decoder = self._model.task_decoders[task_info.task_id]
                    ranking = run_lp_mini_batch_predict(decoder, lp_test_embs, dataloader, device)
                    pre_results[task_info.task_id] = ranking
        if recon_nfeat_test_loader is not None:
            # We also need to compute test scores for node feature reconstruction tasks.
            dataloaders = lp_test_loader.dataloaders
            task_infos = lp_test_loader.task_infos

            with th.no_grad():
                for dataloader, task_info in zip(dataloaders, task_infos):
                    if dataloader is None:
                        pre_results[task_info.task_id] = (None, None)

                    if isinstance(self.gnn_encoder, GSgnnGNNEncoderInterface):
                        if self.has_sparse_params():
                            # When there are learnable embeddings, we can not
                            # just simply skip the last layer self-loop.
                            # Keep the self-loop and print a warning
                            # we will use the computed embs directly
                            logging.warning("When doing %s inference, we need to "
                                            "avoid adding self loop in the last GNN layer "
                                            "to avoid the potential node "
                                            "feature leakage issue. "
                                            "When there are learnable embeddings on "
                                            "nodes, GraphStorm can not automatically"
                                            "skip the last layer self-loop"
                                            "Please set use_self_loop to False",
                                            BUILTIN_TASK_RECONSTRUCT_NODE_FEAT)
                        else:
                            # skip the selfloop of the last layer to
                            # avoid information leakage.
                            self._model.gnn_encoder.skip_last_selfloop()
                            embs = gen_embs()
                            self._model.gnn_encoder.reset_last_selfloop()
                    else:
                        # we will use the computed embs directly
                        logging.warning("The gnn encoder %s does not support skip "
                                        "the last self-loop operation"
                                        "(skip_last_selfloop). There is a potential "
                                        "node feature leakage risk when doing %s training.",
                                        type(self._model.gnn_encoder),
                                        BUILTIN_TASK_RECONSTRUCT_NODE_FEAT)
                    decoder = self._model.task_decoders[task_info.task_id]
                    preds, labels = \
                        run_node_mini_batch_predict(decoder,
                                                    embs,
                                                    dataloader,
                                                    device=device,
                                                    return_proba=return_proba,
                                                    return_label=do_eval)
                    ntype = list(preds.keys())[0]
                    pre_results[task_info.task_id] = (preds[ntype], labels[ntype] \
                        if labels is not None else None)

        if do_eval:
            test_start = time.time()
            val_score, test_score = self.evaluator.evaluate(
                pre_results, pre_results, 0)
            sys_tracker.check('run evaluation')
            if get_rank() == 0:
                self.log_print_metrics(val_score=val_score,
                                       test_score=test_score,
                                       dur_eval=time.time() - test_start,
                                       total_steps=0)

        g = data.g
        if save_embed_path is not None:
            save_gsgnn_embeddings(g,
                                  save_embed_path,
                                  embs,
                                  node_id_mapping_file=node_id_mapping_file,
                                  save_embed_format=save_embed_format)
            barrier()
            sys_tracker.check('save embeddings')

        barrier()

        if save_prediction_path is not None:
            target_ntypes = set()
            task_infos = mt_loader.task_infos
            dataloaders = mt_loader.dataloaders
            for task_info in task_infos:
                if task_info.task_type in \
                    [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
                    target_ntypes.add(task_info.task_config.target_ntype)
                elif task_info.task_type in \
                    [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
                    target_ntypes.add(task_info.task_config.target_etype[0][0])
                    target_ntypes.add(task_info.task_config.target_etype[0][2])
                else:
                    # task_info.task_type is BUILTIN_TASK_LINK_PREDICTION
                    # There is no prediction results for link prediction
                    continue

            nid_shuffler = NodeIDShuffler(g, node_id_mapping_file, list(target_ntypes)) \
                    if node_id_mapping_file else None

            for task_info, dataloader in zip(task_infos, dataloaders):
                task_id = task_info.task_id
                if task_id in pre_results:
                    save_pred_path = os.path.join(save_prediction_path, task_id)
                    if task_info.task_type in \
                        [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
                        pred, _ = pre_results[task_id]
                        if pred is not None:
                            shuffled_preds = {}

                            target_ntype = task_info.task_config.target_ntype
                            pred_nids = dataloader.target_nidx[target_ntype]
                            if node_id_mapping_file is not None:
                                pred_nids = nid_shuffler.shuffle_nids(
                                    target_ntype, pred_nids)

                            shuffled_preds[target_ntype] = (pred, pred_nids)
                            save_node_prediction_results(shuffled_preds, save_pred_path)
                    elif task_info.task_type in \
                        [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
                        pred, _ = pre_results[task_id]
                        if pred is not None:
                            shuffled_preds = {}
                            target_etype = task_info.task_config.target_etype[0]
                            pred_eids = dataloader.target_eidx[target_etype]

                            pred_src_nids, pred_dst_nids = \
                                g.find_edges(pred_eids, etype=target_etype)

                            if node_id_mapping_file is not None:
                                pred_src_nids = nid_shuffler.shuffle_nids(
                                    target_etype[0], pred_src_nids)
                                pred_dst_nids = nid_shuffler.shuffle_nids(
                                    target_etype[2], pred_dst_nids)
                            shuffled_preds[target_etype] = \
                                (pred, pred_src_nids, pred_dst_nids)
                            save_edge_prediction_results(shuffled_preds, save_pred_path)

                    else:
                        # There is no prediction results for link prediction
                        # and feature reconstruction
                        continue

        # save relation embedding if any
        if get_rank() == 0:
            decoders = self._model.task_decoders
            for task_id, decoder in decoders.items():
                if isinstance(decoder, LinkPredictDistMultDecoder):
                    if save_embed_path is not None:
                        rel_emb_path = os.path.join(save_embed_path, task_id)
                        os.makedirs(rel_emb_path, exist_ok=True)
                        save_relation_embeddings(rel_emb_path, decoder)
