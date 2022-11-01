"""Edge classification based on RGNN
"""
import time
import torch as th

import torch.nn as nn
from torch.utils.data import DataLoader
import dgl

from .rgnn import GSgnnBase

from ..dataloading.utils import modify_fanout_for_target_etype
from .utils import save_embeddings as save_node_embeddings

class GSgnnEdgeModel(GSgnnBase):
    """ RGNN edge classification model

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    config: GSConfig
        The graphstorm GNN configuration
    bert_model: dict
        A dict of BERT models in a format of ntype -> bert_model
    task_tracker: GSTaskTrackerAbc
        Task tracker used to log task progress
    train_task: bool
        Whether it is a training task
    """
    def __init__(self, g, config, bert_model, task_tracker=None, train_task=True):
        super(GSgnnEdgeModel, self).__init__(
            g, config, bert_model, task_tracker, train_task)

        self.bert_hidden_size = {ntype: bm.config.hidden_size for ntype, bm in bert_model.items()}
        # TODO needs to be extended to multiple
        self.target_etype = [tuple(target_etype.split(',')) for target_etype in config.target_etype][0]

        if train_task:
            # adjusting the evaluation fanout if the removal of target edge is requested.
            if config.remove_target_edge:
                target_etypes = self.target_etype
                reverse_edge_types_map = config.reverse_edge_types_map
                for e in target_etypes:
                    if e in reverse_edge_types_map and reverse_edge_types_map[e] not in target_etypes:
                        target_etypes.append(reverse_edge_types_map[e])
                self._eval_fanout = modify_fanout_for_target_etype(
                    g=g, fanout=self.eval_fanout, target_etypes=target_etypes)

        self.log_params(config.__dict__)
        self.alpha_l2norm = config.alpha_l2norm

    def inference(self, val_test_nodes, val_src_dst_pairs, test_src_dst_pairs, val_labels, test_labels, bert_emb_cache=None):
        '''This performs inference on the target nodes.

        Parameters
        ----------
        val_test_nodes : tensor
            The node IDs of the predict node type where we perform prediction.
        val_src_dst_pairs: dict of node pairs
            The src and dst node IDs of the validation pairs for different etype.
        test_src_dst_pairs: dict of node pairs
            The src and dst node IDs of the test pairs for different etype.
        val_labels: tensor
            The labels for all validation edges.
        test_labels: tensor
            The labels for all test edges.
        bert_emb_cache : dict of embedding cache
            The embedding cache for the nodes in the input graph.

        Returns
        -------
        tensor
            The prediction results.
        '''
        g = self._g
        device = 'cuda:%d' % self.dev_id
        # TODO
        """
        1. Collect all the node ids from the edge id to create the target nidxs
        2. Call the compute embeddings function
        3. Call the decoder on the results
        """
        node_embeddings = self.compute_embeddings(g, device, bert_emb_cache, val_test_nodes)

        t0 = time.time()
        # find the target src and dst ntypes
        decoder_eval_batch_size = self.eval_batch_size
        target_src_ntype, target_etype, target_dst_ntype = g.to_canonical_etype(self.target_etype)
        decoder = self.decoder
        if decoder is not None:
            decoder.eval()
        with th.no_grad():
            for ntype in node_embeddings:
                node_embeddings[ntype] = node_embeddings[ntype].dist_tensor
            if len(val_src_dst_pairs[target_etype][0]) > 0:
                val_len = len(val_src_dst_pairs[target_etype][0])
                # TODO can use DGL dataloader here so that no need to
                # retreive labels for every step
                dataloader = DataLoader(th.arange(val_len),
                                        batch_size=decoder_eval_batch_size,
                                        shuffle=False)
                # save preds and labels together in order not to shuffle
                # the order when gather tensors from other trainers
                val_preds_list = []
                val_labels_list = []
                val_node_embeddings_src = \
                    node_embeddings[target_src_ntype][val_src_dst_pairs[target_etype][0]]
                val_node_embeddings_dst = \
                    node_embeddings[target_dst_ntype][val_src_dst_pairs[target_etype][1]]
                for i, (val_idx) in enumerate(dataloader):
                    self.keep_alive(report_step=i)
                    if i % 10 == 0:
                        print(f"Decoder val batch number {i}")
                    val_preds_list.append(
                        self.predict(decoder.module.predict(
                            val_node_embeddings_src[val_idx].to(device),
                            val_node_embeddings_dst[val_idx].to(device))))
                    val_labels_list.append(val_labels[val_idx].to(device))
                # can't use torch.stack here becasue the size of last tensor is different
                val_preds = th.cat(val_preds_list)
                val_labels = th.cat(val_labels_list)

            if len(test_src_dst_pairs[target_etype][0]) > 0:
                test_len = len(test_src_dst_pairs[target_etype][0])
                dataloader = DataLoader(th.arange(test_len), batch_size=decoder_eval_batch_size, shuffle=False)
                test_preds_list = []
                test_labels_list = []
                test_node_embeddings_src = node_embeddings[target_src_ntype][test_src_dst_pairs[target_etype][0]]
                test_node_embeddings_dst = node_embeddings[target_dst_ntype][test_src_dst_pairs[target_etype][1]]
                for i, (test_idx) in enumerate(dataloader):
                    self.keep_alive(report_step=i)
                    if i % 10 == 0:
                        print(("Decoder test batch number {}".format(i)))
                    test_preds_list.append(self.predict(decoder.module.predict(test_node_embeddings_src[test_idx].to(device),
                                                                                test_node_embeddings_dst[test_idx].to(device))))

                    test_labels_list.append(test_labels[test_idx].to(device))
                test_preds = th.cat(test_preds_list)
                test_labels = th.cat(test_labels_list)
        th.distributed.barrier()
        if decoder is not None:
            decoder.train()
        if g.rank() == 0:
            print(("Time for decoder prediction operation {}").format(time.time()-t0))

        return val_preds, val_labels, test_preds, test_labels

    def fit(self, loader, train_data, bert_emb_cache=None):
        '''The fit function to train the model.

        Parameters
        ----------
        g : DGLGraph
            The input graph
        loader : GSgnn dataloader
            The dataloader generates mini-batches to train the model.
        bert_emb_cache : dict of embedding cache
            The embedding cache for the nodes in the input graph.
        '''
        g = self.g
        device = 'cuda:%d' % self.dev_id
        gnn_encoder = self.gnn_encoder
        decoder = self.decoder
        embed_layer = self.embed_layer
        bert_model = self.bert_model
        combine_optimizer = self.combine_optimizer

        # The bert_emb_cache is used in following cases:
        # 1) We don't need to fine-tune Bert, i.e., train_nodes == 0.
        #    In this case, we only generate bert bert_emb_cache once before model training.
        # 2) We want to use bert cache to speedup model training, i.e. use_bert_cache == True
        #    We generate bert bert_emb_cache before model training.
        #    If refresh_cache is set to True, the bert_emb_cache is refreshed every epoch.
        #    Otherwise, it is not updated unless some text nodes are selected as trainable text nodes.
        # 3) GNN warnup when gnn_warmup_epochs > 0. We generate the bert emb_cache before model training.
        #    In the first gnn_warmup_epochs epochs, the number of trainable text nodes are set to 0 and
        #    the bert_emb_cache is not refreshed.
        #    After gnn_warmup_epochs, we follow the Case 2 and Case 4 to control the bert_emb_cache.
        # 4) if use_bert_cache is False and train_nodes > 0, no emb_cache is used unless Case 3.
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
        best_epoch = 0
        num_input_nodes = 0
        bert_forward_time = 0
        gnn_forward_time = 0
        back_time = 0
        total_steps = 0
        early_stop = False # used when early stop is True
        for epoch in range(self.n_epochs):
            if gnn_encoder is not None:
                gnn_encoder.train()
            decoder.train()
            if embed_layer is not None:
                embed_layer.train()
            for ntype in bert_model.keys():
                bert_model[ntype].train()

            t0 = time.time()
            # GNN has been pre-trained, clean the cached bert embedding, if bert cache is not used.
            if epoch == self.gnn_warmup_epochs and self.train_nodes > 0 and self.use_bert_cache is False:
                bert_emb_cache = None

            for i, (input_nodes, batch_graph, blocks) in enumerate(loader):
                total_steps += 1
                blocks = [blk.to(device) for blk in blocks]
                if type(input_nodes) is not dict:
                    # This happens on a homogeneous graph.
                    assert len(g.ntypes) == 1
                    input_nodes = {"node": input_nodes}

                for _, nodes in input_nodes.items():
                    num_input_nodes += nodes.shape[0]

                batch_graph = batch_graph.to(device)
                batch_tic = time.time()
                gnn_embs, bert_forward_time, gnn_forward_time = \
                    self.encoder_forward(blocks, input_nodes, bert_emb_cache,
                                         bert_forward_time, gnn_forward_time, epoch)

                # TODO expand code for multiple edge types
                # retrieving seed edge id from the graph to find labels
                seeds = batch_graph.edges[self.target_etype].data[dgl.EID]

                logits = decoder(batch_graph, gnn_embs)
                lbl = train_data.labels[seeds].to(device)

                # add regularization loss to all parameters to avoid the unused parameter errors
                pred_loss = self.loss_func(logits, lbl)

                reg_loss = th.tensor(0.).to(device)
                # L2 regularization of dense parameters
                for d_para in self.get_dense_params():
                    reg_loss += d_para.square().sum()

                # weighted addition to the total loss
                total_loss = pred_loss + self.alpha_l2norm * reg_loss

                t3 = time.time()
                gnn_loss = pred_loss.item()
                combine_optimizer.zero_grad()
                total_loss.backward()
                combine_optimizer.step()
                back_time += (time.time() - t3)

                train_score = self.evaluator.compute_score(self.predict(logits), lbl)

                self.log_metric("Train loss", total_loss.item(), total_steps, report_step=total_steps)
                for metric in self.evaluator.metric:
                    self.log_metric("Train {}".format(metric), train_score[metric], total_steps,
                                    report_step=total_steps)

                if i % 20 == 0 and g.rank() == 0:
                    if self.verbose:
                        self.print_info(epoch, i, num_input_nodes / 20,
                                        (bert_forward_time / 20, gnn_forward_time / 20, back_time / 20))
                    # Print task specific info.
                    print(
                        "Part {} | Epoch {:05d} | Batch {:03d} | Train Loss (ALL|GNN): {:.4f}|{:.4f} | Time: {:.4f}".
                        format(g.rank(), epoch, i, total_loss.item(), gnn_loss, time.time() - batch_tic))
                    for metric in self.evaluator.metric:
                        print("Train {}: {:.4f}".format(metric, train_score[metric]))
                    num_input_nodes = bert_forward_time = gnn_forward_time = back_time = 0

                # save model and embeddings every save_model_per_iters
                if self.save_model_per_iters > 0 and i % self.save_model_per_iters == 0 and i != 0:
                    self.save_model_embed(epoch, i, g, bert_emb_cache)

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
                if g.rank() == 0:
                    print('Refresh BERT cache.')
                bert_emb_cache = self.generate_bert_cache(g)
            # save model and node embeddings
            self.save_model_embed(epoch, None, g, bert_emb_cache)

            if self.evaluator is not None and self.evaluator.do_eval(total_steps, epoch_end=True):
                val_score = self.eval(g.rank(), train_data, bert_emb_cache, total_steps)

                if self.evaluator.do_early_stop(val_score):
                    early_stop = True

            th.distributed.barrier()

            # early_stop, exit training
            if early_stop is True:
                break

        print("Peak Mem alloc: {:.4f} MB".format(th.cuda.max_memory_allocated(device) / 1024 /1024))
        if g.rank() == 0:
            output = dict(best_test_score=self.evaluator.best_test_score,
                          best_val_score=self.evaluator.best_val_score,
                          peak_mem_alloc_MB=th.cuda.max_memory_allocated(device) / 1024 / 1024,
                          best_epoch=best_epoch)
            print(output)

    def eval(self, rank, train_data, bert_emb_cache, total_steps):
        """ do the model evaluation using validiation and test sets

            Parameters
            ----------
            rank: int
                Distributed rank
            train_data: GSgnnEdgePredictionTrainData
                Training data
            bert_emb_cache: dict of tensor
                Bert embedding cahce
            total_steps: int
                Total number of iterations.

            Returns
            -------
            float: validation score
        """
        test_start = time.time()
        val_labels = train_data.labels[train_data.val_idxs[self.target_etype[1]]]
        test_labels = train_data.labels[train_data.test_idxs[self.target_etype[1]]]
        pred = self.inference(train_data.val_test_nodes,
                              train_data.val_src_dst_pairs,
                              train_data.test_src_dst_pairs, val_labels, test_labels,
                              bert_emb_cache)

        val_pred, val_labels, test_pred, test_labels = pred

        val_score, test_score = self.evaluator.evaluate(
            val_pred, test_pred,
            val_labels, test_labels,
            total_steps)

        if rank == 0:
            self.log_print_metrics(val_score=val_score,
                                test_score=test_score,
                                dur_eval=time.time() - test_start,
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

        target_ntypes = data.target_ntypes
        # compute node embeddings
        outputs = self.compute_embeddings(g, device, bert_emb_cache)
        embeddings = {ntype: outputs[ntype] for ntype in target_ntypes}

        if self.evaluator is not None and \
            self.evaluator.do_eval(0, epoch_end=True):
            test_start = time.time()

            # Do evaluation
            target_etypes = data.target_etypes
            assert len(target_etypes) == 1, \
                "Only can do edge classification for one edge type"
            target_src_ntype, target_etype, target_dst_ntype = \
                g.to_canonical_etype(target_etypes[0])
            test_src_dst_pairs = data.test_src_dst_pairs[target_etype]
            decoder = self.decoder
            if decoder is not None:
                decoder.eval()

            test_labels = data.labels[data.test_idxs[target_etype]]
            with th.no_grad():
                if len(test_src_dst_pairs[0]) > 0:
                    test_len = len(test_src_dst_pairs[0])
                    dataloader = DataLoader(th.arange(test_len),
                        batch_size=self.eval_batch_size, shuffle=False)

                    test_preds_list = []
                    test_labels_list = []
                    test_node_embeddings_src = embeddings[target_src_ntype][test_src_dst_pairs[0]]
                    test_node_embeddings_dst = embeddings[target_dst_ntype][test_src_dst_pairs[1]]
                    for _, (test_idx) in enumerate(dataloader):
                        pred = self.predict(decoder.module.predict(
                            test_node_embeddings_src[test_idx].to(device),
                            test_node_embeddings_dst[test_idx].to(device)))
                        test_preds_list.append(pred)

                        test_labels_list.append(test_labels[test_idx].to(device))
                test_preds = th.cat(test_preds_list)
                test_labels = th.cat(test_labels_list)
            th.distributed.barrier()

            val_score, test_score = self.evaluator.evaluate(
                test_preds, test_preds,
                test_labels, test_labels,
                0)

            if g.rank() == 0:
                self.log_print_metrics(val_score=val_score,
                                       test_score=test_score,
                                       dur_eval=time.time() - test_start,
                                       total_steps=0)

        save_embeds_path = self.save_embeds_path

        # If save_embeds_path is set to None.
        # A user does not want to save the node embedding
        if save_embeds_path is not None:
            # Save node embedding
            save_node_embeddings(save_embeds_path,
                embeddings, g.rank(), th.distributed.get_world_size())
            th.distributed.barrier()
