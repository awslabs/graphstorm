"""RGNN for node tasks
"""
import os
import time
import torch as th

import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import apex
import dgl
import abc
import psutil

from .rgnn import M5GNNBase
from .utils import rand_gen_trainmask
from .emb_cache import EmbedCache
from .extract_node_embeddings import prepare_batch_input
from .hbert import get_bert_flops_info
from .utils import save_embeddings as save_node_embeddings

class M5GNNNodeModel(M5GNNBase):
    """ RGNN model for node tasks

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    config: M5GNNConfig
        The M5 GNN configuration
    bert_model: dict
        A dict of BERT models in a format of ntype -> bert_model
    train_task: bool
        Whether it is a training task
    """
    def __init__(self, g, config, bert_model, train_task=True):
        super(M5GNNNodeModel, self).__init__(g, config, bert_model, train_task)
        self.predict_ntype = config.predict_ntype
        self.save_predict_path = config.save_predict_path
        self.bert_hidden_size = {ntype: bm.config.hidden_size for ntype, bm in bert_model.items()}

    def inference(self, target_nidx, bert_emb_cache=None):
        '''This performs inference on the target nodes.

        Parameters
        ----------
        tartet_ndix : tensor
            The node IDs of the predict node type where we perform prediction.
        bert_emb_cache : dict of embedding cache
            The embedding cache for the nodes in the input graph.

        Returns
        -------
        tensor
            The prediction results.
        '''
        g = self._g
        device = 'cuda:%d' % self.dev_id
        outputs = self.compute_embeddings(g, device, bert_emb_cache, {self.predict_ntype: target_nidx})
        outputs = outputs[self.predict_ntype]

        return self.predict(self.decoder(outputs[0:len(outputs)]))

    def fit(self, loader, train_data, bert_emb_cache=None):
        g = self._g
        device = 'cuda:%d' % self.dev_id
        gnn_encoder = self.gnn_encoder
        decoder = self.decoder
        embed_layer = self.embed_layer
        bert_model = self.bert_model
        combine_optimizer = self.combine_optimizer

        # The bert_emb_cache is used in following cases:
        # 1) We don't need to fine-tune Bert, i.e., args.train_nodes == 0.
        #    In this case, we only generate bert bert_emb_cache once before model training.
        # 2) We want to use bert cache to speedup model training, i.e. args.use_bert_cache == True
        #    We generate bert bert_emb_cache before model training.
        #    If args.refresh_cache is set to True, the bert_emb_cache is refreshed every epoch.
        #    Otherwise, it is not updated unless some text nodes are selected as trainable text nodes.
        # 3) GNN warnup when args.gnn_warmup_epochs > 0. We generate the bert emb_cache before model training.
        #    In the first gnn_warmup_epochs epochs, the number of trainable text nodes are set to 0 and
        #    the bert_emb_cache is not refreshed.
        #    After gnn_warmup_epochs, we follow the Case 2 and Case 4 to control the bert_emb_cache.
        # 4) if args.use_bert_cache is False and args.train_nodes > 0, no emb_cache is used unless Case 3.
        if (self.train_nodes == 0 or self.use_bert_cache or self.gnn_warmup_epochs > 0) \
            and (bert_emb_cache is None) and (len(self.bert_static) > 0): # it is not initialized elsewhere
            bert_emb_cache = self.generate_bert_cache(g)
            if self.train_nodes == 0 and g.rank() == 0:
                print('Use fixed BERT embeddings.')
            elif g.rank() == 0:
                print('Compute BERT cache.')

        # training loop
        print("start training...")
        dur = []
        total_steps = 0
        num_input_nodes = 0
        bert_forward_time = 0
        gnn_forward_time = 0
        back_time = 0
        early_stop = False # used when early stop is True
        for epoch in range(self.n_epochs):
            if gnn_encoder is not None:
                gnn_encoder.train()
            if embed_layer is not None:
                embed_layer.train()
            for ntype in bert_model.keys():
                bert_model[ntype].train()
            t0 = time.time()

            # GNN has been pre-trained, clean the cached bert embedding, if bert cache is not used.
            if epoch == self.gnn_warmup_epochs and self.train_nodes > 0 and self.use_bert_cache is False:
                bert_emb_cache = None

            for i, (input_nodes, seeds, blocks) in enumerate(loader):
                total_steps += 1

                # in the case of a graph with a single node type the returned seeds will not be
                # a dictionary but a tensor of integers this is a possible bug in the DGL code.
                # Otherwise we will select the seeds that correspond to the category node type
                if type(seeds) is dict:
                    seeds = seeds[self.predict_ntype]     # we only predict the nodes with type "category"
                if type(input_nodes) is not dict:
                    input_nodes = {self.predict_ntype: input_nodes}
                for _, nodes in input_nodes.items():
                    num_input_nodes += nodes.shape[0]
                batch_tic = time.time()

                gnn_embs, bert_forward_time, gnn_forward_time = \
                    self.encoder_forward(blocks, input_nodes, bert_emb_cache,
                                         bert_forward_time, gnn_forward_time, epoch)

                emb = gnn_embs[self.predict_ntype]

                if self.mixed_precision and self.mp_opt_level == 'O1':
                    # avoid model auto cast as dgl does not support it.
                    # apex opt level O0 does not use mix precision.
                    with apex.amp.disable_casts():
                        logits = decoder(emb)
                else:
                    logits = decoder(emb)

                lbl = train_data.labels[seeds].to(device)
                loss = self.loss_func(logits, lbl)

                t3 = time.time()
                gnn_loss = loss.item()
                combine_optimizer.zero_grad()
                loss.backward()
                combine_optimizer.step()
                back_time += (time.time() - t3)

                train_score = self.evaluator.compute_score(self.predict(logits), lbl)

                self.log_metric("Train loss", loss.item(), total_steps, report_step=total_steps)
                for metric in  self.evaluator.metric:
                    self.log_metric("Train {}".format(metric), train_score[metric], total_steps, report_step=total_steps)


                if i % 20 == 0 and g.rank() == 0:
                    if self.verbose:
                        self.print_info(epoch, i,  num_input_nodes / 20,
                                        (bert_forward_time / 20, gnn_forward_time / 20, back_time / 20))
                    print("Part {} | Epoch {:05d} | Batch {:03d} | Train Loss (ALL|GNN): {:.4f}|{:.4f} | Time: {:.4f}".
                            format(g.rank(), epoch, i,  loss.item(), gnn_loss, time.time() - batch_tic))
                    for metric in self.evaluator.metric:
                        print("Train {}: {:.4f}".format(metric, train_score[metric]))
                    num_input_nodes = bert_forward_time = gnn_forward_time = back_time = 0
                if self.evaluator is not None and \
                    self.evaluator.do_eval(total_steps, epoch_end=False):
                    val_score = self.eval(g.rank(), train_data, bert_emb_cache, total_steps)

                    if self.evaluator.do_early_stop(val_score):
                        early_stop = True

                # early_stop, exit current interation.
                if early_stop is True:
                    break

            # end of an epoch
            th.distributed.barrier()
            epoch_time = time.time() - t0
            if g.rank() == 0:
                print("Epoch {} take {}".format(epoch, epoch_time))
            dur.append(epoch_time)

            # re-generate cache
            if self.use_bert_cache and self.refresh_cache and epoch >= self.gnn_warmup_epochs:
                bert_emb_cache = self.generate_bert_cache(g)
            self.save_model_embed(epoch, None, g, bert_emb_cache)

            if self.evaluator is not None and self.evaluator.do_eval(total_steps, epoch_end=True):
                val_score = self.eval(g.rank(), train_data, bert_emb_cache, total_steps)
                if self.evaluator.do_early_stop(val_score):
                    early_stop = True

            # early_stop, exit training
            if early_stop is True:
                break

        if g.rank() == 0:
            if self.verbose:
                if self.evaluator is not None:
                    self.evaluator.print_history()

    def eval(self, rank, train_data, bert_emb_cache, total_steps):
        """ do the model evaluation using validiation and test sets

            Parameters
            ----------
            rank: int
                Distributed rank
            train_data: M5gnnNodeTrainData
                Training data
            bert_emb_cache: dict of tensor
                Bert embedding cahce
            total_steps: int
                Total number of iterations.

            Returns
            -------
            float: validation score
        """
        teval = time.time()
        target_nidx = th.cat([train_data.val_idx, train_data.test_idx])
        pred = self.inference(target_nidx, bert_emb_cache)

        val_pred, test_pred = th.split(pred,
                                       [len(train_data.val_idx),
                                       len(train_data.test_idx)])
        val_label = train_data.labels[train_data.val_idx]
        val_label = val_label.to(val_pred.device)
        test_label = train_data.labels[train_data.test_idx]
        test_label = test_label.to(test_pred.device)
        val_score, test_score = self.evaluator.evaluate(
            val_pred, test_pred,
            val_label, test_label,
            total_steps)
        if rank == 0:
            self.log_print_metrics(val_score=val_score,
                                    test_score=test_score,
                                    dur_eval=time.time() - teval,
                                    total_steps=total_steps)
        return val_score

    def infer(self, data, bert_emb_cache=None):
        g = self.g
        device = 'cuda:%d' % self.dev_id

        if (bert_emb_cache is None) and (len(self.bert_static) > 0):
            bert_emb_cache = self.generate_bert_cache(g)
            if g.rank() == 0:
                print('Compute BERT cache.')

        print("start inference ...")
        # TODO: Make it more efficient
        # We do not need to compute the embedding of all node types
        outputs = self.compute_embeddings(g, device, bert_emb_cache)
        embeddings = outputs[self.predict_ntype]

        # Save prediction result into disk
        if g.rank() == 0:
            predicts = []
            # TODO(xiangsx): Make it distributed (more efficient)
            # The current implementation is only memory efficient
            for start in range(0, len(embeddings), 10240):
                end = start + 10240 if start + 10240 < len(embeddings) else len(embeddings)
                predict = self.predict(self.decoder(embeddings[start:end]))
                predicts.append(predict)
            predicts = th.cat(predicts, dim=0)
            os.makedirs(self.save_predict_path, exist_ok=True)
            th.save(predicts, os.path.join(self.save_predict_path, "predict.pt"))

        th.distributed.barrier()

        # do evaluation if any
        if self.evaluator is not None and \
            self.evaluator.do_eval(0, epoch_end=True):
            test_start = time.time()
            pred = self.predict(self.decoder(embeddings[data.test_idx]))
            labels = data.labels[data.test_idx]
            pred = pred.to(device)
            labels = labels.to(device)

            val_score, test_score = self.evaluator.evaluate(
                pred, pred,
                labels, labels,
                0)
            if g.rank() == 0:
                self.log_print_metrics(val_score=val_score,
                                        test_score=test_score,
                                        dur_eval=time.time() - test_start,
                                        total_steps=0)

        save_embeds_path = self.save_embeds_path
        if save_embeds_path is not None:
            # User may not want to save the node embedding
            # save node embedding
            save_node_embeddings(save_embeds_path,
                embeddings, g.rank(), th.distributed.get_world_size())
            th.distributed.barrier()
