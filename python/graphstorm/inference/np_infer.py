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

    Infer wrapper for node classification and regression.
"""
import time
from dgl.distributed import DistTensor

from .graphstorm_infer import GSInfer
from ..model.utils import save_embeddings as save_gsgnn_embeddings
from ..model.utils import save_prediction_results
from ..model.utils import shuffle_predict
from ..model.gnn import do_full_graph_inference
from ..model.node_gnn import node_mini_batch_gnn_predict
from ..model.node_gnn import node_mini_batch_predict

from ..utils import sys_tracker, get_world_size, barrier

class GSgnnNodePredictionInfer(GSInfer):
    """ Node classification/regression infer.

    This is a highlevel infer wrapper that can be used directly
    to do node classification/regression model inference.

    Parameters
    ----------
    model : GSgnnNodeModel
        The GNN model for node prediction.
    rank : int
        The rank.
    """

    def infer(self, loader, save_embed_path, save_prediction_path=None,
              use_mini_batch_infer=False,
              node_id_mapping_file=None,
              return_proba=True,
              save_embed_format="pytorch"):
        """ Do inference

        The inference does three things:
        1. (Optional) Evaluate the model performance on a test set if given
        2. Generate node embeddings
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
        # TODO support multiple ntypes
        assert len(loader.data.eval_ntypes) == 1, \
            "GraphStorm only support single target node type for training and inference"
        ntype = loader.data.eval_ntypes[0]

        if use_mini_batch_infer:
            res = node_mini_batch_gnn_predict(self._model, loader, return_proba,
                                              return_label=do_eval)
            pred = res[0]
            embs = res[1]
            label = res[2] if do_eval else None

            if isinstance(embs, dict):
                embs = {ntype: embs[ntype]}
            else:
                embs = {ntype: embs}
        else:
            embs = do_full_graph_inference(self._model, loader.data, fanout=loader.fanout,
                                           task_tracker=self.task_tracker)
            res = node_mini_batch_predict(self._model, embs, loader, return_proba,
                                          return_label=do_eval)
            pred = res[0]
            label = res[1] if do_eval else None
        if isinstance(pred, dict):
            pred = pred[ntype]
        if isinstance(label, dict):
            label = label[ntype]
        sys_tracker.check('compute embeddings')

        device = self.device

        # do evaluation first
        # do evaluation if any
        if do_eval:
            test_start = time.time()
            val_score, test_score = self.evaluator.evaluate(pred, pred, label, label, 0)
            sys_tracker.check('run evaluation')
            if self.rank == 0:
                self.log_print_metrics(val_score=val_score,
                                       test_score=test_score,
                                       dur_eval=time.time() - test_start,
                                       total_steps=0)

        if save_embed_path is not None:
            if use_mini_batch_infer:
                g = loader.data.g
                ntype_emb = DistTensor((g.num_nodes(ntype), embs[ntype].shape[1]),
                    dtype=embs[ntype].dtype, name=f'gen-emb-{ntype}',
                    part_policy=g.get_node_partition_policy(ntype),
                    # TODO: this makes the tensor persistent in memory.
                    persistent=True)
                # nodes that do prediction in mini-batch may be just a subset of the
                # entire node set.
                ntype_emb[loader.target_nidx[ntype]] = embs[ntype]
            else:
                ntype_emb = embs[ntype]
            embeddings = {ntype: ntype_emb}

            save_gsgnn_embeddings(save_embed_path, embs, self.rank,
                get_world_size(),
                device=device,
                node_id_mapping_file=node_id_mapping_file,
                save_embed_format=save_embed_format)

        if save_prediction_path is not None:
            # shuffle pred results according to node_id_mapping_file
            if node_id_mapping_file is not None:
                g = loader.data.g

                pred_shape = list(pred.shape)
                pred_shape[0] = g.num_nodes(ntype)
                pred_data = DistTensor(pred_shape,
                    dtype=pred.dtype, name=f'predict-{ntype}',
                    part_policy=g.get_node_partition_policy(ntype),
                    # TODO: this makes the tensor persistent in memory.
                    persistent=True)
                # nodes that have predictions may be just a subset of the
                # entire node set.
                pred_data[loader.target_nidx[ntype]] = pred.cpu()
                pred = shuffle_predict(pred_data, node_id_mapping_file, ntype, self.rank,
                    get_world_size(), device=device)
            save_prediction_results(pred, save_prediction_path, self.rank)
        barrier()
        sys_tracker.check('save predictions')
