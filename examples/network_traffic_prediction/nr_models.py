
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

    Demonstration models for using GraphStorm APIs
"""

import time
import torch as th
import graphstorm as gs
from graphstorm.utils import barrier
from graphstorm.model import (GSgnnNodeModel,
                              GSNodeEncoderInputLayer,
                              GSEdgeEncoderInputLayer,
                              RelationalGCNEncoder,
                              EntityRegression,
                              RegressionLossFunc)
from graphstorm.trainer import GSgnnNodePredictionTrainer
from graphstorm.inference import GSgnnNodePredictionInferrer
from graphstorm.model.utils import (append_to_dict,
                                    save_shuffled_node_embeddings,
                                    save_node_prediction_results,
                                    NodeIDShuffler)


def get_input_feat_size(feat_size, window_size, ts_feat_names, ts_size):
    """ method sepcificly designed for getting the input feature size when having time series features
    """
    # reset feature size according to window size
    new_efeat_size = {}
    for a_type, f_size in feat_size.items():
        if a_type in ts_feat_names:
            # new feature size is static + window_size time seria
            new_efeat_size[a_type] = feat_size - len(ts_feat_names[a_type]) * ts_size  + window_size
        else:
            new_efeat_size[a_type] = feat_size

    return new_efeat_size

def get_st_feats(input_feats, ts_feat_names, ts_size):
    """ Extract static node/edge features only

    input_feats: dict of tensor, or list of dict of tensor
        If a dict of tensor, it contains node features. If a list of features, it contains
        edge features.
    ts_feat_name: dict of string and list
        A dict of node/edge type and time series feature names in format {type, [str, ...])
    ts_size: int
        The overall time series data size, e.g., 24.

    Tricks:
    1: ts features are always be concatinated at the end of edge features in outputs.
    2: all types of ts features have the same time seria size.

    Return:
    -------
    new_st_feats: dict of tensor, or list of dict of tensor

    """
    if isinstance(input_feats, dict):
        st_feats = {}
        for a_type, feats in input_feats.items():
            # has time seria features
            if a_type in ts_feat_names:
                ts_names = ts_feat_names[a_type]
                num_ts_names = len(ts_names)
                feats = feats[:, 0: -(num_ts_names * ts_size)]
            # only has static features, just copy the feature back.
            else:
                feats = feats
            st_feats[a_type] = feats
    elif isinstance(input_feats, list) and isinstance(input_feats[0], dict):
        st_feats = []
        for one_input_feats in input_feats:
            one_st_feats = {}
            for a_type, feats in one_input_feats.items():
                # has time seria features
                if a_type in ts_feat_names:
                    ts_names = ts_feat_names[a_type]
                    num_ts_names = len(ts_names)
                    feats = feats[:, 0: -(num_ts_names * ts_size)]
                # only has static features, just copy the feature back.
                else:
                    feats = feats            
                one_st_feats[a_type] = feats
            st_feats.append(one_st_feats)
    else:
        raise NotImplementedError(f'The format of {input_feats} is not a dict or a list dict')

    return st_feats

def get_one_step_ts_feats(input_feats, ts_feat_names, ts_size, window_size, step):
    """ Extract time seria features only

    edge_feats: list of dict of tuple and tensor
        A list of edge features for all blocks. Elements are dict in format [tuple, tensor]
    ts_feat_name: dict of tupe and string
        A dict of edge type and time serias feature names in format [tuple, string]
    ts_size: int
        The overall time seria size, e.g., 24.
    window_size: int
        The sliding window size, e.g., 7.
    step: int
        The current sliding step.
    event_name: string
        Specific argument for the Network Congestion. Default is `event`.
    
    Assumption:
    1: time series features are always be concatinated at the end of edge features; 
    2: labels only contain time seria values. So need to extract the window + 1 values as labels.

    Return:
    -------
    new_ts_edge_feats: list of dict of tuple and tensor

    """
    if isinstance(input_feats, dict):
        one_step_ts_feats = {}
        for a_type, feats in input_feats.items():
            # has ts features
            if a_type in ts_feat_names:
                ts_names = ts_feat_names[a_type]
                num_ts_names = len(ts_names)
                ts_feats = feats[:, -(num_ts_names * ts_size): ]
                # For generasity: if has mutliple ts_feats, mean them into the ts_size
                ts_feats = ts_feats.view(feats.shape[0], -1, ts_size).mean(dim=1)
                ts_feats_in_window = ts_feats[:, step: (step + window_size)]
            # only has static features, copy input feats
            else:
                ts_feats_in_window = feats
            one_step_ts_feats[a_type] = ts_feats_in_window
    elif isinstance(input_feats, list):
        one_step_ts_feats = []
        # process features layer by layer
        for input_feat in input_feats:
            new_feats = {}
            for a_type, feats in input_feat.items():
                # has ts features
                if a_type in ts_feat_names:
                    ts_names = ts_feat_names[a_type]
                    num_ts_names = len(ts_names)
                    ts_feats = feats[:, -(num_ts_names * ts_size): ]
                    # For generasity: if has mutliple ts_feats, mean them into the ts_size
                    ts_feats = ts_feats.view(feats.shape[0], -1, ts_size).mean(dim=1)
                    ts_feats_in_window = ts_feats[:, step: (step + window_size)]
                # only has static features, copy input feats
                else:
                    ts_feats_in_window = feats
                new_feats[a_type] = ts_feats_in_window
            one_step_ts_feats.append(new_feats)
    else:
        raise NotImplementedError(f'The format of {input_feats} is not a dict or a list dict')

    return one_step_ts_feats

def combine_st_ts_feats(types, st_feats, ts_feats):
    """ Combine static, and time-seria together as input features.

    Both features should be in a dict or a list of dict format, and have the same length.
    The feature dict is like: {type1: tensor, type2: tensor, ...}. An extreme case is that
    there is no features, which is like {}

    The major issue is that not all types have the two types of features. So, we
    need to check all types. 

    The order of concatination is static, time serias, and event.
    """
    if isinstance(st_feats, dict) and isinstance(ts_feats, dict):
        pass
        assert len(st_feats) == len(ts_feats), 'static and time serias features should have' + \
                                               'the same number of types.'
        all_feats = {}
        for a_type in types:
            feats = []
            if a_type in st_feats:
                feats.append(st_feats[a_type])
            if a_type in ts_feats:
                feats.append(ts_feats[a_type])

            if feats:
                all_feats[a_type] = th.concat(feats, dim=-1)
    elif isinstance(st_feats, list) and isinstance(ts_feats, list):
        pass
        assert len(st_feats) == len(ts_feats), 'static and time serias features should have' + \
                                               'the same length!'
        assert len(st_feats[0]) == len(ts_feats[0]), 'static and time serias features should ' + \
                                                     'have the same number of types.'
        all_feats = []

        for st_feat, ts_feat in zip(st_feats, ts_feats):
            all_feat = {}
            for a_type in types:
                feats = []
                if a_type in st_feat:
                    feats.append(st_feat[a_type])
                if a_type in ts_feat:
                    feats.append(ts_feat[a_type])

                if feats:
                    all_feat[a_type] = th.concat(feats, dim=-1)

            all_feats.append(all_feat)
    else:
        raise NotImplementedError(f'The format of {st_feats} and {ts_feats} are not a dict ' + \
                                  'or a list dict')

    return all_feats

def get_ts_labels(labels, ts_size, window_size, step):
    """
    labels: dict of tensor
        A dict of edge type and lables in format [tuple, tensor]

    Assumption:
    1/: labels only contain time seria values. So need to extract window + 1 values as labels.

    """
    # process labels for time seria prediction
    new_labels = {}
    for a_type, ts_labels in labels.items():
        assert ts_labels.shape[1] == ts_size, \
            f'The label dimension must be the same as time serie size!'

        # extract the next time values after window size
        ts_label = ts_labels[:, (step + window_size)]
        new_labels[a_type] = ts_label.unsqueeze(dim=-1)
    return new_labels

class RgcnNRModel4TS(GSgnnNodeModel):
    """ Customized RGCN model for Air Traffic Prediction

    This RGCN model inheritates the GraphStorm GSgnnNodeModel, but add a few components
    designed specificly for the Air Traffic Prediction task:

    1. It uses a window size to set the previous demands and traffic the model can see;
    2. Based on the data seen in the previous window size, predicts the next traffic value;
    3. Iterates the values in the given time points for one forward step, and
    4. Aggregates all loss in one step and return

    Arguments
    ----------
    g: DistGraph
        A DGL DistGraph
    num_hid_layers: int
        The number of gnn layers
    node_feat_field: dict of list of strings
        The list features for each node type to be used in the model
    hid_size: int
        The dimension of hidden layers.
    target_etype: tuple of str
        Target etype for prediction
    window_size: int
        The window size of history data that a model can. Default is 4.
    time_window: int
        The overall time points in the input demands, traffics, and events. Default is 24.
    edge_feat_field: dict of list of strings
        The list features for each edge type to be used in the model
    """
    def __init__(self,
                 g,
                 num_hid_layers,
                 node_feat_field,
                 edge_feat_field,
                 edge_feat_mp_op,
                 hid_size,
                 target_ntype,
                 ts_nfeat_names,
                 ts_efeat_names,
                 ts_size,
                 window_size=7):
        super(RgcnNRModel4TS, self).__init__(alpha_l2norm=0.)
        self.alpha_l2norm = 0.
        self.target_ntype = target_ntype
        self._ts_nfeat_names = ts_nfeat_names
        self._ts_efeat_names = ts_efeat_names
        self._ts_size = ts_size
        self._window_size = window_size

        # extract feature size
        total_nfeat_size = gs.get_node_feat_size(g, node_feat_field)
        total_efeat_size = gs.get_edge_feat_size(g, edge_feat_field)
        input_nfeat_size = get_input_feat_size(total_nfeat_size, window_size,
                                               self._ts_nfeat_names, ts_size)
        input_efeat_size = get_input_feat_size(total_efeat_size, window_size,
                                               self._ts_efeat_names, ts_size)
        # set an input layer encoder
        node_encoder = GSNodeEncoderInputLayer(g=g, feat_size=input_nfeat_size,
                                               embed_size=hid_size)
        self.set_node_input_encoder(node_encoder)
        edge_encoder = GSEdgeEncoderInputLayer(g=g, feat_size=input_efeat_size,
                                               embed_size=hid_size)
        self.set_edge_input_encoder(edge_encoder)

        # set a GNN encoder
        gnn_encoder = RelationalGCNEncoder(g=g, 
                                           h_dim=hid_size,
                                           out_dim=hid_size,
                                           num_hidden_layers=num_hid_layers-1,
                                           edge_feat_name=edge_feat_field,
                                           edge_feat_mp_op=edge_feat_mp_op)
        self.set_gnn_encoder(gnn_encoder)

        # set a decoder specific to edge regression task
        decoder = EntityRegression(hid_size, out_dim=1)
        self.set_decoder(decoder)

        # classification loss function
        self.set_loss_func(RegressionLossFunc())

        # initialize model's optimizer
        self.init_optimizer(lr=0.01, sparse_optimizer_lr=0.01, weight_decay=0)

    def forward(self, blocks, node_feats, edge_feats, labels, input_nodes=None):
        """
        This forward uses the sliding windows method for time series feature and prediction.
        That is, use static feature + time seria feature in a window to predict the (window + 1)
        time seria value. Then slide the window one step ahead to predict the next values.

        edge_feats: list of dict of tensors
            A list of edge features in the format: {str: tensor}. The length of the list should
            be equal to the length of blocks. Trainer will provide the `edge_feats`. And only
            edge types that have features will be included.
        """
        ts_losses = []
        # extract static features, only once because it does not change in one batch.
        # as users know the details of data, so customized model can use a trick to extract
        # time series data, which are always concated at the end of each node and edge features.
        # Based on this trick and the _ts_size, it is easy to separate static features from time
        # series ones.
        st_node_feats = get_st_feats(node_feats, self._ts_nfeat_names, self._ts_size)
        st_edge_feats = get_st_feats(edge_feats, self._ts_efeat_names, self._ts_size)

        # ------------- Process Time Serial Data ------------- #
        for step in range(0, (self._ts_size - self._window_size)):
            ts_node_feats = get_one_step_ts_feats(node_feats, self._ts_nfeat_names,
                                                  self._ts_size, self._window_size, step)
            ts_edge_feats = get_one_step_ts_feats(edge_feats, self._ts_efeat_names,
                                                  self._ts_size, self._window_size, step)
            ntypes = list(blocks[0].ntypes)
            all_node_feats = combine_st_ts_feats(ntypes, st_node_feats, ts_node_feats)
            can_etypes = list(blocks[0].canonical_etypes)
            all_edge_feats = combine_st_ts_feats(can_etypes, st_edge_feats, ts_edge_feats)

            new_labels = get_ts_labels(labels, self._ts_size, self._window_size, step)

            # run input encoder
            node_input_embs = self.node_input_encoder(all_node_feats, input_nodes)
            edge_input_embs = self.edge_input_encoder(all_edge_feats)

            # run gnn encoder
            encode_embs = self.gnn_encoder(blocks, node_input_embs, edge_input_embs)

            # run decoder
            logits = self.decoder(encode_embs[self.target_ntype])

            # compute loss per window
            target_ntype = list(new_labels.keys())[0]
            pred_loss = self.loss_func(logits, new_labels[target_ntype])
            
            ts_losses.append(pred_loss)

        ts_loss = th.tensor(0.).to(pred_loss.device)
        for loss in ts_losses:
            ts_loss += loss
        ts_loss = ts_loss / len(ts_losses)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(pred_loss.device)
        # L2 regularization of dense parameters
        for d_para in self.get_dense_params():
            reg_loss += d_para.square().sum()

        # weighted addition to the total loss
        return ts_loss + self.alpha_l2norm * reg_loss

    def predict(self, blocks, node_feats, edge_feats,
                input_nodes, return_prob=False, use_ar=False, predict_step=-1):
        """
        Add a new argument, ``use_ar``, to indicate if this ``predict()`` function
        will use auto-regressive method. When use auto-regressive method, it will
        not do for-loop, but compute one predict step.
        
        Callers of this function, e.g., Inferrers, need to handle the time series
        processing and provide a value of ``predict_step`` argument.

        Parameters
        ----------
        use_ar: bool
            Determine whether to use autoregressive method.
        predict_step: int
            The step to be predicted.
        Returns
        --------
            ts_predicts: dict of string and 2D tensor
        """
        st_node_feats = get_st_feats(node_feats, self._ts_nfeat_names, self._ts_size)
        st_edge_feats = get_st_feats(edge_feats, self._ts_efeat_names, self._ts_size)

        if use_ar:
            assert predict_step >= 0 and predict_step < (self._ts_size - self._window_size), \
                        'To use autoregressive, must provide a positive predict_step, and ' + \
                        'the predict_step must be smaller than (time seria size - window size).'
            ts_node_feats = get_one_step_ts_feats(node_feats, self._ts_nfeat_names,
                                                  self._ts_size, self._window_size, predict_step)
            ts_edge_feats = get_one_step_ts_feats(edge_feats, self._ts_efeat_names,
                                                  self._ts_size, self._window_size, predict_step)
            ntypes = list(blocks[0].ntypes)
            all_node_feats = combine_st_ts_feats(ntypes, st_node_feats, ts_node_feats)
            can_etypes = list(blocks[0].canonical_etypes)
            all_edge_feats = combine_st_ts_feats(can_etypes, st_edge_feats, ts_edge_feats)

            # run input encoder
            node_input_embs = self.node_input_encoder(all_node_feats, input_nodes)
            edge_input_embs = self.edge_input_encoder(all_edge_feats)

            # run gnn encoder
            encode_embs = self.gnn_encoder(blocks, node_input_embs, edge_input_embs)

            # run decoder
            target_logits = self.decoder(encode_embs[self.target_ntype])
            ts_predicts = {self.target_ntype: target_logits}
        else:
            # ------------- Use the Same TS Processing Method ------------- #
            predicts_tensor = None
            for step in range(0, (self._ts_size - self._window_size)):
                ts_node_feats = get_one_step_ts_feats(node_feats, self._ts_nfeat_names,
                                                    self._ts_size, self._window_size, step)
                ts_edge_feats = get_one_step_ts_feats(edge_feats, self._ts_efeat_names,
                                                    self._ts_size, self._window_size, step)
                ntypes = list(blocks[0].ntypes)
                all_node_feats = combine_st_ts_feats(ntypes, st_node_feats, ts_node_feats)
                can_etypes = list(blocks[0].canonical_etypes)
                all_edge_feats = combine_st_ts_feats(can_etypes, st_edge_feats, ts_edge_feats)

                # run input encoder
                node_input_embs = self.node_input_encoder(all_node_feats, input_nodes)
                edge_input_embs = self.edge_input_encoder(all_edge_feats)

                # run gnn encoder
                encode_embs = self.gnn_encoder(blocks, node_input_embs, edge_input_embs)

                # run decoder
                target_logits = self.decoder(encode_embs[self.target_ntype])

                if predicts_tensor is not None:
                    predicts_tensor = th.concat([predicts_tensor, target_logits], dim=-1)
                else:
                    predicts_tensor = target_logits

            ts_predicts = {self.target_ntype: predicts_tensor}

        # print(ts_predicts[self.target_ntype].shape)
        return ts_predicts, encode_embs

    @property
    def ts_size(self):
        return self._ts_size

    @property
    def window_size(self):
        return self._window_size

    @property
    def st_efeat_names(self):
        return self._st_efeat_names

    @property
    def ts_efeat_names(self):
        return self._ts_efeat_names

class NodePredictionTrainer4TS(GSgnnNodePredictionTrainer):
    """ Simply overwrite GSgnnNodePredictionTrainer's ``eval()`` function, using mini-batch
    inference only, and use a customized ``node_mini_batch_gnn_predict`` to implement the
    time series prediction without using autoregression method.
    """
    def eval(self, model, val_loader, test_loader, use_mini_batch_infer, total_steps,
             return_proba=True):
        teval = time.time()

        val_pred, _, val_label = node_mini_batch_gnn_predict(model, val_loader, return_proba,
                                                             return_label=True, use_ar=False)

        if test_loader is not None:
            test_pred, _, test_label = \
                node_mini_batch_gnn_predict(model, test_loader, return_proba,
                                            return_label=True)
        else: # there is no test set
            test_pred = None
            test_label = None
        ntype = list(val_label.keys())[0]

        val_pred = val_pred[ntype].to(self.device)
        val_label = val_label[ntype].to(self.device)
        if test_pred is not None:
            test_pred = test_pred[ntype].to(self.device)
            test_label = test_label[ntype].to(self.device)
        
        val_score, test_score = self.evaluator.evaluate(val_pred, test_pred,
                                                        val_label, test_label, total_steps)
        if gs.get_rank() == 0:
            self.log_print_metrics(val_score=val_score,
                                    test_score=test_score,
                                    dur_eval=time.time() - teval,
                                    total_steps=total_steps)
        return val_score

class NodePredictionInferrer4TS(GSgnnNodePredictionInferrer):
    """ Simply overwrite GSgnnNodePredictionInferrer's ``infer()`` method, using mini-batch
    inference only.
    
    For inference, need to use autoregresion method because we can not get real value as time
    window input, need to use predicted value to predict the next ones.
    """
    def infer(self, loader, save_embed_path, save_prediction_path=None,
              use_mini_batch_infer=False,
              node_id_mapping_file=None,
              return_proba=True,
              save_embed_format="pytorch",
              init_step=0,
              infer_steps=14):
        """
        New arguments
        --------------
        init_step: int
            The starting point for prediction. That is, the saved model will need to use historical
            data from the init_step to (init_step + model.window_size) to predict the value in the
            (init_step + model.window_size) + 1 step.
        infer_steps: int
            Steps to be predicted. That is, the save model will predict from init_step
            (init_step + model.window_size), and run infer_steps. When sliding one window ahead, use
            predicted value as input, and predict the next ones.
        """

        # ====== Not do evaluation in the auto-agressive inference ====== #

        # ====== Not save embedding in this customized inferrer ========= #
        assert save_embed_path is None, \
            "Unable to save the node embeddings when using mini batch inference." \
            "It is not guaranteed that mini-batch prediction will cover all the nodes."

        self._model.eval()

        # get the graph
        g = loader.data.g
        # get the model metadata
        window_size = self._model.window_size

        # This customized inferer only handle the first ntype for inference.
        infer_ntype = list(loader.target_nidx.keys())[0]

        # get all demands features
        inventory_nfeats = g.nodes[infer_ntype].data['inventory_amounts']
        th_inventory_nfeats = inventory_nfeats[th.arange(inventory_nfeats.shape[0])]

        predict_results = None
        for predict_step in range(init_step, (init_step + infer_steps)):
            res = node_mini_batch_gnn_predict(self._model, loader, return_proba,
                                                return_label=False, use_ar=True,
                                                predict_step=predict_step)
            preds = res[0]

            replace_step = predict_step + window_size
            # Replace the next step inventory_amounts with the predicitons
            th_inventory_nfeats[:, replace_step] = preds[infer_ntype].squeeze()
            inventory_nfeats[th.arange(th_inventory_nfeats.shape[0])] = th_inventory_nfeats
            g.nodes[infer_ntype].data['inventory_amounts'] = inventory_nfeats

            if predict_results is None:
                predict_results = preds[infer_ntype]
            else:
                predict_results = th.concat([predict_results, preds[infer_ntype]], dim=-1)

        total_predict_results = {infer_ntype: predict_results}

        nid_shuffler = None
        g = loader.data.g

        if save_prediction_path is not None:
            # save_embed_path may be None. In that case, we need to init nid_shuffler
            if nid_shuffler is None:
                nid_shuffler = NodeIDShuffler(g, node_id_mapping_file,
                                              list(total_predict_results.keys())) \
                    if node_id_mapping_file else None
            shuffled_preds = {}
            for ntype, pred in total_predict_results.items():
                pred_nids = loader.target_nidx[ntype]
                if node_id_mapping_file is not None:
                    pred_nids = nid_shuffler.shuffle_nids(ntype, pred_nids)
                shuffled_preds[ntype] = (pred, pred_nids)

            save_node_prediction_results(shuffled_preds, save_prediction_path)

        barrier()


def node_mini_batch_gnn_predict(model, loader, return_proba=True,
                                return_label=False, use_ar=False, predict_step=-1):
    device = model.device
    data = loader.data
    g = data.g
    preds = {}
    # For this customized model, we only use one target node type
    target_ntype = list(loader.target_nidx.keys())[0]

    if return_label:
        assert loader.label_field is not None, \
            "Return label is required, but the label field is not provided when" \
            "initlaizing the loader."

    embs = {}
    labels = {}
    model.eval()

    len_dataloader = max_num_batch = len(loader)
    global_num_batch = th.tensor([len_dataloader], device=device)
    if gs.utils.is_distributed():
        th.distributed.all_reduce(tensor, op=th.distributed.ReduceOp.MAX)
        max_num_batch = tensor[0]

    dataloader_iter = iter(loader)

    with th.no_grad():
        for iter_l in range(max_num_batch):
            tmp_keys = []
            blocks = None
            if iter_l < len_dataloader:
                input_nodes, seeds, blocks = next(dataloader_iter)
                if not isinstance(input_nodes, dict):
                    assert len(g.ntypes) == 1
                    input_nodes = {g.ntypes[0]: input_nodes}

            nfeat_fields = loader.node_feat_fields
            node_input_feats = data.get_node_feats(input_nodes, nfeat_fields, device)
            efeat_fields = loader.edge_feat_fields
            edge_input_feats = data.get_blocks_edge_feats(blocks, efeat_fields, device)

            if blocks is None:
                continue
            # Remove additional keys (ntypes) added for WholeGraph compatibility
            for ntype in tmp_keys:
                del input_nodes[ntype]
            blocks = [block.to(device) for block in blocks]
            pred, emb = model.predict(blocks, node_input_feats, edge_input_feats,
                                      input_nodes, return_proba, use_ar, predict_step)

            # process lables for time seria prediction
            label_field = loader.label_field
            lbl = data.get_node_feats(seeds, label_field)

            ts_labels = []
            for step in range(0, (model.ts_size - model.window_size)):
                ts_labels_in_window = get_ts_labels(lbl, model.ts_size,
                                                    model.window_size, step)
                ts_labels.append(ts_labels_in_window[target_ntype])
            new_labels = {target_ntype: th.concat(ts_labels, dim=1)}

            if return_label:
               append_to_dict(new_labels, labels)

            # pred can be a Tensor or a dict of Tensor
            # emb can be a Tensor or a dict of Tensor
            if isinstance(pred, dict):
                append_to_dict(pred, preds)
            else:
                assert len(seeds) == 1, \
                    f"Expect prediction results of multiple node types {label.keys()}" \
                    f"But only get results of one node type"
                ntype = list(seeds.keys())[0]
                append_to_dict({ntype: pred}, preds)

            if isinstance(emb, dict):
                append_to_dict(emb, embs)
            else:
                ntype = list(seeds.keys())[0]
                append_to_dict({ntype: emb}, embs)

    model.train()
    preds = {
        ntype: th.cat(preds[ntype])
        for ntype in preds if ntype in target_ntype
    }
    for ntype, ntype_emb in embs.items():
        embs[ntype] = th.cat(ntype_emb)
    if return_label:
        for ntype, ntype_label in labels.items():
            labels[ntype] = th.cat(ntype_label)
        return preds, embs, labels
    else:
        return preds, embs, None

