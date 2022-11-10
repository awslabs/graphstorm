"""Link prediction based on RGNN
"""
import time
import torch as th

import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from .utils import save_embeddings as save_node_embeddings
from .utils import save_relation_embeddings

from .edge_decoder import LinkPredictDotDecoder, LinkPredictDistMultDecoder
from .rgnn import GSgnnBase

from ..eval import compute_acc_lp

class GSgnnLinkPredictionModel(GSgnnBase):
    """ RGNN link prediction model

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
        super(GSgnnLinkPredictionModel, self).__init__(
            g, config, bert_model, task_tracker, train_task)
        self._g = g

        # train specific configs
        if train_task:
            # sampling related
            self.negative_sampler = config.negative_sampler
            self.num_negative_edges = config.num_negative_edges
            self.exclude_training_targets = config.exclude_training_targets

        # train_etypes is used when loading decoder.
        # Inference script also needs train_etypes
        self.train_etypes = [tuple(train_etype.split(',')) \
            for train_etype in config.train_etype]
        self.use_dot_product = config.use_dot_product
        # decoder related
        if self.use_dot_product is False:
            self.gamma = config.gamma

        # evaluation
        self.num_negative_edges_eval = config.num_negative_edges_eval

        self.bert_hidden_size = {ntype: bm.config.hidden_size for ntype, bm in bert_model.items()}

        self.model_conf = {
            'task': 'link_predict',
            'train_etype': self.train_etypes,

            # GNN
            'gnn_model': self.gnn_model_type,
            'num_layers': self.n_layers,
            'hidden_size': self.n_hidden,
            'num_bases': self.n_bases,
            'dropout': self.dropout,
            'use_self_loop': self.use_self_loop,
        }
        # logging all the params of this experiment
        self.log_params(config.__dict__)
        self.alpha_l2norm = config.alpha_l2norm

    def init_dist_decoder(self, train):
        g = self.g
        dev_id = self.dev_id
        num_train_etype = len(self.train_etypes)

        # For backword compatibility, we add this check.
        # if train etype is 1, There is no need to use DistMult
        assert num_train_etype > 1 or self.use_dot_product, \
            "If number of train etype is 1, please use dot product"
        if self.use_dot_product:
            # if the training set only contains one edge type or it is specified in the arguments,
            # we use dot product as the score function.
            if g.rank() == 0:
                print('use dot product for single-etype task.')
                print("Using inner product objective for supervision")
            decoder = LinkPredictDotDecoder()
        else:
            if g.rank() == 0:
                print("Using distmult objective for supervision")
            decoder = LinkPredictDistMultDecoder(g,
                                                 self.n_hidden,
                                                 self.gamma)

        decoder = decoder.to(dev_id)
        if isinstance(decoder, LinkPredictDistMultDecoder):
            # decoder also need to be distributed
            decoder = DistributedDataParallel(decoder, device_ids=[dev_id], output_device=dev_id, find_unused_parameters=True)

        self.decoder = decoder

    def fit(self, loader):
        '''The fit function to train the model.

        Parameters
        ----------
        g : DGLGraph
            The input graph
        loader : GSgnn dataloader
            The dataloader generates mini-batches to train the model.
        '''
        g = self.g
        device = 'cuda:%d' % self.dev_id
        gnn_encoder = self.gnn_encoder
        decoder = self.decoder
        embed_layer = self.embed_layer
        bert_model = self.bert_model
        combine_optimizer = self.combine_optimizer

        # training loop
        print("start training...")
        dur = []
        best_epoch = 0
        num_input_nodes = 0
        bert_forward_time = 0
        gnn_forward_time = 0
        back_time = 0
        total_steps = 0
        val_mrr = None
        test_mrr = None
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
            for i, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(loader):
                total_steps += 1

                if type(input_nodes) is not dict:
                    # This happens on a homogeneous graph.
                    assert len(g.ntypes) == 1
                    input_nodes = {g.ntypes[0]: input_nodes}

                for _, nodes in input_nodes.items():
                    num_input_nodes += nodes.shape[0]

                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                batch_tic = time.time()

                gnn_embs, bert_forward_time, gnn_forward_time = \
                    self.encoder_forward(blocks, input_nodes,
                                         bert_forward_time, gnn_forward_time, epoch)

                # TODO add w_relation in calculating the score. The current is only valid for homogenous graph.
                pos_score = decoder(pos_graph, gnn_embs)
                neg_score = decoder(neg_graph, gnn_embs)

                score = th.cat([pos_score, neg_score])
                label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)])

                # add regularization loss to all parameters to avoid the unused parameter errors
                pred_loss = F.binary_cross_entropy_with_logits(score, label)

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

                train_acc = compute_acc_lp(pos_score, neg_score)

                self.log_metric("Train loss", total_loss.item(), total_steps, report_step=total_steps)
                for metric in train_acc.keys():
                    self.log_metric("Train {}".format(metric), train_acc[metric], total_steps,
                                    report_step=total_steps)

                if i % 20 == 0 and g.rank() == 0:
                    if self.verbose:
                        self.print_info(epoch, i, num_input_nodes / 20,
                                        (bert_forward_time / 20, gnn_forward_time / 20, back_time / 20))
                    print("Epoch {:05d} | Batch {:03d} | Total_Train Loss (ALL|GNN): {:.4f}|{:.4f} | Time: {:.4f}".
                            format(epoch, i, total_loss.item(), gnn_loss, time.time() - batch_tic))
                    for metric in train_acc.keys():
                        print("Train {}: {:.4f}".format(metric, train_acc[metric]))
                    num_input_nodes = bert_forward_time = gnn_forward_time = back_time = 0

                val_score = None
                if self.evaluator is not None and \
                    self.evaluator.do_eval(total_steps, epoch_end=False):
                    embeddings = self.compute_embeddings(g, device)
                    train_mrr = self.evaluator.evaluate_on_train_set(embeddings, decoder, device)
                    val_score = self.eval(g.rank(), embeddings, total_steps, train_mrr)

                    if self.evaluator.do_early_stop(val_score):
                        early_stop = True

                # Every n iterations, check to save the top k models. If has validation score, will save
                # the best top k. But if no validation, will either save the last k model or all models
                # depends on the setting of top k
                if self.save_model_per_iters > 0 and i % self.save_model_per_iters == 0 and i != 0:
                    self.save_topk_models(epoch, i, g, val_score)

                # early_stop, exit current interation.
                if early_stop is True:
                    break

            # ------- end of an epoch -------

            th.distributed.barrier()
            epoch_time = time.time() - t0
            if g.rank() == 0:
                print("Epoch {} take {}".format(epoch, epoch_time))
            dur.append(epoch_time)

            val_score = None
            if self.evaluator is not None and self.evaluator.do_eval(total_steps, epoch_end=True):
                # force to sync before doing full graph inference
                embeddings = self.compute_embeddings(g, device,
                                                     target_nidx=self.evaluator.target_nidx)
                val_score = self.eval(g.rank(), embeddings, total_steps)

                if self.evaluator.do_early_stop(val_score):
                    early_stop = True

            # After each epoch, check to save the top k models. If has validation score, will save
            # the best top k. But if no validation, will either save the last k model or all models
            # depends on the setting of top k. To show this is after epoch save, set the iteration
            # to be None, so that we can have a determistic model folder name for testing and debug.
            self.save_topk_models(epoch, None, g, val_score)

            th.distributed.barrier()

            # early_stop, exit training
            if early_stop is True:
                break

        print("Peak Mem alloc: {:.4f} MB".format(th.cuda.max_memory_allocated(device) / 1024 /1024))
        if g.rank() == 0:
            output = dict(best_test_mrr=self.evaluator.best_test_score, best_val_mrr=self.evaluator.best_val_score, final_test_mrr=test_mrr,
                          final_val_mrr=val_mrr, peak_mem_alloc_MB=th.cuda.max_memory_allocated(device) / 1024 / 1024,
                          best_epoch=best_epoch)
            print(output)

            if self.verbose:
                # print top k info only when required because sometime the top k is just the last k
                print(f'Top {len(self.topklist.toplist)} ranked models:')
                print([f'Rank {i+1}: epoch-{epoch}-iter-{iter}' \
                        for i, (epoch, iter) in enumerate(self.topklist.toplist)])

    def eval(self, rank, embeddings, total_steps, train_score=None):
        """ do the model evaluation using validiation and test sets

            Parameters
            ----------
            rank: int
                Distributed rank
            embeddings: dict of DistTensor
                node embeddings
            total_steps: int
                Total number of iterations.
            train_score: float
                Training mrr, used in print.

            Returns
            -------
            float: validation mrr
        """
        decoder = self.decoder
        device = 'cuda:%d' % self.dev_id
        val_mrr, test_mrr = self.evaluator.evaluate(embeddings, decoder, total_steps, device)

        if rank == 0:
            test_start = time.time()
            self.log_print_metrics(val_score=val_mrr,
                                    test_score=test_mrr,
                                    dur_eval=time.time() - test_start,
                                    total_steps=total_steps,
                                    train_score=train_score)
        return val_mrr

    def infer(self):
        g = self.g
        device = 'cuda:%d' % self.dev_id

        print("start inference ...")
        embeddings = self.compute_embeddings(g, device)

        if self.evaluator is not None and \
            self.evaluator.do_eval(0, epoch_end=True):
            self.eval(g.rank(), embeddings, 0)

        save_embeds_path = self.save_embeds_path
        assert save_embeds_path is not None
        # save node embedding
        save_node_embeddings(save_embeds_path,
            embeddings, g.rank(), th.distributed.get_world_size())
        th.distributed.barrier()
        # save relation embedding if any
        if g.rank() == 0:
            decoder = self.decoder.module \
            if isinstance(self.decoder, DistributedDataParallel) \
            else self.decoder

            if isinstance(decoder, LinkPredictDistMultDecoder):
                save_relation_embeddings(save_embeds_path, decoder)
