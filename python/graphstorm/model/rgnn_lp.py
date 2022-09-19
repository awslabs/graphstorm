"""
Link prediction GNN models

Support GNN models:
MLP-encoder + RGCN + Decoder
MLP-encoder + RGAT + Decoder
BERT-MLP-encoder + RGCN + Decoder
BERT-MLP-encoder + RGAT + Decoder
"""
import time

from .nn import LinkPredictDotDecoder, LinkPredictDistMultDecoder
from .rgnn_base import RelGNNBase
from ..config import BUILTIN_LP_LOSS_CROSS_ENTROPY
from ..config import BUILTIN_LP_LOSS_LOGSIGMOID_RANKING

import torch as th
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

class LinkPredictionRelGNN(RelGNNBase):
    """ GNN based link prediction model

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    config: GSConfig
        The graphstorm GNN configuration
    """
    def __init__(self, g, config):
        super(LinkPredictionRelGNN, self).__init__(g, config)

        # Link prediction specific
        self._train_etypes = [tuple(train_etype.split(',')) for train_etype in config.train_etype]

        # decoder related
        self._use_dot_product = config.use_dot_product
        if self.use_dot_product is False:
            self._gamma = config.gamma

        self.init_loss_func(config.lp_loss_func)

        # log all configs and params
        self.log_runtime_params(self.__dict__)

    def init_decoder(self, train=True):
        """ Init link prediction related decoder
        """
        g = self.g
        num_train_etype = len(self.train_etypes)
        if self.use_dot_product or num_train_etype == 1:
            # if the training set only contains one edge type or it is specified in the arguments,
            # we use dot product as the score function.
            if g.rank() == 0:
                print("use dot product for single-etype task.")
                print("Using inner product objective for supervision")
            decoder = LinkPredictDotDecoder()
        else:
            if g.rank() == 0:
                print("Using distmult objective for supervision")
            decoder = LinkPredictDistMultDecoder(
                len(g.etypes),
                {etype: i for i, etype in enumerate(g.etypes)},
                self.n_hidden,
                self.gamma)

        dev_id = self.dev_id
        decoder = decoder.to(dev_id)
        if isinstance(decoder, LinkPredictDistMultDecoder):
            # decoder also need to be distributed
            decoder = DistributedDataParallel(decoder, device_ids=[dev_id],
                output_device=dev_id, find_unused_parameters=True)
        self.decoder = decoder

    def cross_entropy(pos_score, neg_score, pos_graph=None, neg_graph=None):
        """ Cross entropy loss
            Positive links are treated as 1
            Negative links are treated as 0

        Parameters
        ----------
        pos_score: Tensor
            Tensor of positive scores
        neg_score: Tensor
            Tensor of negative scores
        pos_graph: Graph
            Graph containing positive edges
        neg_graph: Graph
            Graph: containing negative edges
        """
        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)])
        loss = F.binary_cross_entropy_with_logits(score, label)
        return loss

    def log_sigmoid(pos_score, neg_score, pos_graph=None, neg_graph=None):
        """ Log sigmoid loss

        Parameters
        ----------
        pos_score: Tensor
            Tensor of positive scores
        neg_score: Tensor
            Tensor of negative scores
        pos_graph: Graph
            Graph containing positive edges
        neg_graph: Graph
            Graph: containing negative edges
        """
        pos_loss = -F.logsigmoid(pos_score)
        neg_loss = -F.logsigmoid(-neg_score)

        # TODO(xiangsx): Add edge weight support
        # Use g.edges(etype=etype) to get eids

        # TODO(xiangsx): add adversarial sampling support

        neg_loss = th.mean(neg_loss)
        pos_loss = th.mean(pos_loss)
        loss = (neg_loss + pos_loss) / 2
        return loss

    def init_loss_func(self, loss_func_name):
        if loss_func_name == BUILTIN_LP_LOSS_CROSS_ENTROPY:
            self.loss_func = self.cross_entropy
        elif loss_func_name == BUILTIN_LP_LOSS_LOGSIGMOID_RANKING:
            self.loss_func = self.log_sigmoid
        else:
            assert False, f"Unknow loss func for link prediction {loss_func_name}" \
                "You should choose {} "

    def fit(self, loader):
        """
        The fit function to train the model.

        Parameters
        ----------
        loader: Dataloader
            Graph dataloader
        """
        g = self.g
        device = 'cuda:%d' % self.dev_id
        gnn_encoder = self.gnn_encoder
        decoder = self.decoder
        embed_layer = self.embed_layer
        combine_optimizer = self.combine_optimizer

        self.prepare_fit()

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
        for epoch in range(self.n_epochs):
            if embed_layer is not None:
                embed_layer.train()
            if gnn_encoder is not None:
                gnn_encoder.train()
            decoder.train()

            start_tic = time.time()
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

                input_embs, emb_forward_time = \
                    self.emb_forward()
                gnn_embs, gnn_forward_time = \
                    self.encoder_forward(blocks, input_embs, epoch)

                pos_score = decoder(pos_graph, gnn_embs)
                neg_score = decoder(neg_graph, gnn_embs)
                loss = self.loss_func(pos_score, neg_score, pos_graph, neg_graph)
                fw_tic = time.time()

                combine_optimizer.zero_grad()
                loss.backward()
                combine_optimizer.step()
                back_tic = time.time()

                train_matrics = self.compute_acc(pos_score, neg_score)

                self.log_metric()

                if self.debug and i % 20 == 0 and g.rank() == 0:
                    # print more training info when debugging
                    self.print_info(epoch, i, num_input_nodes / 20,
                                    (bert_forward_time / 20, gnn_forward_time / 20, back_time / 20))
                    # Print task specific info.
                    batch_time = time.time() - batch_tic
                    fw_time = fw_tic - batch_tic
                    bw_time = back_tic - fw_tic
                    print(f"Epoch {epoch:05d} | Batch {i:03d} | " \
                        f"Train Loss (ALL|GNN): {loss.item():.4f} | " \
                        f"Batch Time: {batch_time:.4f} | " \
                        f"Forward Time: {fw_time:.4f} | " \
                        f"Backward Time: {bw_time:.4f}")
                    for metric in train_matrics.keys():
                        print(f"Train {metric}: {train_matrics[metric]:.4f}")
                    num_input_nodes = bert_forward_time = gnn_forward_time = back_time = 0

                # Call save model and embeddings
                # save_model_emb will decide whether to save the model
                self.save_model_emb(epoch, total_steps, g)

                # Do evaluation when we have an evaluator
                if self._evaluator is not None and \
                    self._evaluator.do_eval(total_steps, epoch_end=False):
                    test_start_tic = time.time()
                    node_embeddings = self.compute_embeddings()
                    train_metrics = self._evaluator.evaluate_on_train( \
                        node_embeddings, decoder, total_steps, device)
                    val_metrics, test_metrics = \
                        self._evaluator.evaluate(node_embeddings, decoder, total_steps, device)
                    if g.rank() == 0:
                        self.log_metric(train_score=train_metrics,
                                        val_score=val_metrics,
                                        test_score=test_metrics,
                                        steps=total_steps)

            # end of an epoch
            th.distributed.barrier()
            epoch_time = time.time() - start_tic
            if g.rank() == 0:
                print(f"Epoch {epoch} take {epoch_time}")
            dur.append(epoch_time)

            # Call save model and embeddings
            # save_model_emb will decide whether to save the model
            self.save_model_embed(epoch, None, g)

            if self.evaluator is not None and \
                self.evaluator.do_eval(total_steps, epoch_end=True):
                # force to sync before doing full graph inference
                test_start_tic = time.time()
                node_embeddings = self.compute_embeddings()
                val_metrics, test_metrics = \
                    self._evaluator.evaluate(node_embeddings, decoder, total_steps, device)

                if g.rank() == 0:
                    self.log_metric(val_score=val_metrics,
                                    test_score=test_metrics,
                                    steps=total_steps)
                    print(f"Epoch {epoch} test take {time.time()-test_start_tic}")

            th.distributed.barrier()

        if self.debug:
            print("Peak Mem alloc: {:.4f} MB".format(th.cuda.max_memory_allocated(device) / 1024 /1024))
            if g.rank() == 0:
                output = dict(best_test_mrr=self.evaluator.best_test_score, best_val_mrr=self.evaluator.best_val_score, final_test_mrr=test_mrr,
                            final_val_mrr=val_mrr, peak_mem_alloc_MB=th.cuda.max_memory_allocated(device) / 1024 / 1024,
                            best_epoch=best_epoch)
                print(output)
