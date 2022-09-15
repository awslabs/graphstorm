"""Language model base implementation"""
import abc
import torch as th
import apex
import psutil
import time
import tqdm
import dgl

import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from .hbert import wrap_bert, get_bert_flops_info
from .utils import save_model as save_lm_model

from ..data.constants import TOKEN_IDX, ATT_MASK_IDX
from .emb_cache import EmbedCache
from .embed import DistGraphEmbed
from .extract_node_embeddings import extract_bert_embeddings_dist
from .extract_node_embeddings import prepare_batch_input
from .utils import save_sparse_embeds
from .utils import LazyDistTensor
from .utils import load_model as load_lm_model

class LanguageModelBase():
    """ Base language model

    Note: we assume each node type has a standalone BERT model.

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    config: M5GNNConfig
        Configurations
    bert_model: dict
        A dict of BERT models in a format of ntype -> bert_model
    verbose: bool
        If True, more information is printed
    """
    def __init__(self, g, config, bert_model, verbose=False):
        if verbose:
            print(config)

        self._g = g
        self._no_validation = config.no_validation
        self._feat_name = config.feat_name

        # lm fine tuning related
        self._train_nodes = config.train_nodes
        self._use_bert_cache = config.use_bert_cache
        self._refresh_cache = config.refresh_cache
        self._gnn_warmup_epochs = config.gnn_warmup_epochs

        # training related
        self._batch_size = config.batch_size
        assert self._batch_size > 0, "batch size must > 0."
        self._mini_batch_infer = config.mini_batch_infer
        self._dropout = config.dropout
        self._mixed_precision = config.mixed_precision
        self._mp_opt_level = config.mp_opt_level
        self._lr = config.lr
        self._bert_lr = config.bert_tune_lr
        self._weight_decay = config.wd_l2norm
        self._bert_infer_bs = config.bert_infer_bs
        self._n_epochs = config.n_epochs


        # distributed training config
        self._local_rank = config.local_rank
        self._ip_config = config.ip_config
        self._graph_name = config.graph_name
        self._part_config = config.part_config
        self._save_model_per_iters = config.save_model_per_iters
        self._save_model_path = config.save_model_path
        self._save_embeds_path = config.save_embeds_path
        self._restore_bert_model_path = config.restore_bert_model_path

        self._debug = config.debug
        self._verbose = verbose

        assert isinstance(bert_model, dict), \
            "The input bert_model must be a dict of BERT models " \
            "in the format of ntype -> bert_model"
        self._bert_model = bert_model

        self._evaluator = None
        # evaluation
        self._eval_batch_size = config.eval_batch_size

        self.model_conf = None
        self.model = None
        self.embed_layer = None
        self.emb_optimizer = None
        self.sparse_emb_optimizer = None
        self.decoder = None
        self.bert_train = {}
        self.bert_static = {}
        self.bert_hidden_size = {}

        # check pretrain_emb_layer config
        self.pretrain_emb_layer = config.pretrain_emb_layer \
            if hasattr(config, 'pretrain_emb_layer') else False
        if self.pretrain_emb_layer:
            self.sparse_lr = config.sparse_lr
        self._n_hidden = config.n_hidden if self.pretrain_emb_layer else 0

        # tracker info
        self.tracker = None
        if g.rank() == 0 and config.mlflow_tracker:
            self._mlflow_exp_name = config.mlflow_exp_name
            self._mlflow_run_name = config.mlflow_run_name
            self._mlflow_report_frequency = config.mlflow_report_frequency
            self.init_tracker() # Init self.tracker
        else:
            self._mlflow_report_frequency = 0 # dummy, as self.tracker is None, it is not used

    def init_tracker(self):
        """Initialize mlflow tracker"""
        # we need this identifier to allow logging for all processes and avoid
        # writting in the same location that is not allowed

        from m5_job_tracker import tracking_config
        tracking_id = ":proc_nbr:"+str(self.g.rank())
        tracking_config.set_tensorboard_config(
            {"log_dir": self._mlflow_exp_name+tracking_id}
        )
        mlflow_config = {"experiment_name": self._mlflow_exp_name+tracking_id,
                         "run_name": self._mlflow_run_name+tracking_id}

        tracking_config.set_mlflow_config(mlflow_config)
        self.tracker = tracking_config.get_or_create_client()
        print("Tracker run info : " + str(self.tracker.get_run_info()))
        # Log host level metadata
        self.tracker.log_current_host_attributes()

    def log_metric(self, metric_name, metric_value, step, report_step=None):
        if self.tracker is not None:
            if report_step is None or (report_step is not None and report_step % self.mlflow_report_frequency == 0):
                self.tracker.log_metric(metric_name, metric_value, step)

    def log_param(self, param_name, param_value, report_step=None):
        if self.tracker is not None:
            if report_step is None or (report_step is not None and report_step % self.mlflow_report_frequency == 0):
                self.tracker.log_param(param_name, param_value)

    def log_params(self, param_value):
        if self.tracker is not None:
            self.tracker.log_params(param_value)

    def setup_cuda(self, local_rank):
        # setup cuda env
        use_cuda = th.cuda.is_available()
        assert use_cuda, "Only support GPU training"
        dev_id = local_rank
        th.cuda.set_device(dev_id)
        self._dev_id = dev_id

    def get_feat_size(self):
        feat_size = {}
        g = self.g
        for ntype in g.ntypes:
            feat_field = self._feat_name
            # user can specify the name of the field
            feat_name = None if feat_field is None else \
                feat_field if isinstance(feat_field, str) \
                else feat_field[ntype] if ntype in feat_field else None

            if feat_name is None:
                feat_size[ntype] = 0
            else:
                # We force users to know which node type has node feature
                # This helps avoid unexpected training behavior.
                assert feat_name in g.nodes[ntype].data, \
                    f"Warning. The feat with name {feat_name} " \
                    f"does not exists for the node type {ntype}" \
                    "If not all your ntypes have node feature," \
                    "you can use --feat-name 'ntype0:feat0 ntype1:feat0`" \
                    "to specify feature names for each node type."
                feat_size[ntype] = g.nodes[ntype].data[feat_name].shape[1]
        return feat_size

    def init_lm_encoder(self):
        g = self.g
        bert_model = self.bert_model
        dev_id = self.dev_id

        self.feat_size = self.get_feat_size()

        text_feat_ntypes = []
        for ntype in g.ntypes:
            if TOKEN_IDX in g.nodes[ntype].data:
                text_feat_ntypes.append(ntype)

        print("Init distributed bert model ...")
        for ntype in bert_model.keys():
            if self.mixed_precision:
                bert_model[ntype] = apex.parallel.DistributedDataParallel(bert_model[ntype].to(dev_id))
            else:
                bert_model[ntype] = DistributedDataParallel(bert_model[ntype].to(dev_id),
                                                            device_ids=[dev_id], output_device=dev_id,
                                                            find_unused_parameters=False)

        if len(bert_model) > 0:
            bert_train, bert_static = wrap_bert(g, bert_model, bert_infer_bs=self.bert_infer_bs, debug=self.debug)
        else:
            bert_train = {}
            bert_static = {}

        # Restore the model to a checkpoint saved previously.
        if self.restore_bert_model_path is not None:
            print('load BERT model from ', self.restore_bert_model_path)
            load_lm_model(self.restore_bert_model_path, None, None, self.bert_model, None)

        self.bert_train = bert_train
        self.bert_static = bert_static


    def init_dist_decoder(self):
        pass

    def init_lm_decoder_optimizer(self):
        pass

    def init_loss_function(self):
        pass

    def generate_bert_cache(self, g):
        ''' Generate the cache of the BERT embeddings on the nodes of the input graph.

        The cache is specific to the input graph and the embeddings are generated with the model
        in this class.

        Parameters
        ----------
        g : DGLGraph
            The input graph

        Returns
        -------
        dict of EmbedCache
            The embedding caches for each node type.
        '''
        device = 'cuda:%d' % self.dev_id
        bert_emb_cache = {}
        assert isinstance(self.bert_train, dict)
        assert isinstance(self.bert_static, dict)
        assert len(self.bert_static) > 0 # Only bert_static is used in this case
        embs = extract_bert_embeddings_dist(g, self.bert_infer_bs, self.bert_train, self.bert_static,
                                            self.bert_hidden_size, dev=device,
                                            verbose=self.verbose, client=self.tracker,
                                            mlflow_report_frequency=self.mlflow_report_frequency)
        for ntype, emb in embs.items():
            bert_emb_cache[ntype] = EmbedCache(emb)
        return bert_emb_cache

    def init_lm_encoder_optimizer(self):
        bert_model = self.bert_model
        bert_params = list([])
        if len(bert_model) > 0:
            num_bert_params = {}
            for ntype, bm in bert_model.items():
                params = list(bm.parameters())
                bert_params = bert_params + params
                num_params = np.sum([np.prod(param.shape) for param in params])
                num_bert_params[ntype] = num_params
                print('Bert model of {} has {} parameters'.format(ntype, num_params))
            if self.mixed_precision:
                fine_tune_opt = apex.optimizers.FusedAdam(bert_params, lr=self.bert_lr)
                key_list = []
                bms = []
                for key, bm in bert_model.items():
                    key_list.append(key)
                    bms.append(bm)

                models, fine_tune_opt = apex.amp.initialize(bms, fine_tune_opt, opt_level=self.mp_opt_level)
                for key, bm in zip(key_list, models):
                    bert_model[key] = bm
            else:
                fine_tune_opt = th.optim.Adam(bert_params, lr=self.bert_lr)
        else:
            fine_tune_opt = None
        self.fine_tune_opt = fine_tune_opt

    def init_emb_layer(self):
        # prepare features
        g = self.g
        dev_id = self.dev_id

        text_feat_ntypes = []
        for ntype in g.ntypes:
            if TOKEN_IDX in g.nodes[ntype].data:
                text_feat_ntypes.append(ntype)

        # create embeddings
        embed_layer = DistGraphEmbed(g,
                                    self.get_feat_size(),
                                    text_feat_ntypes,
                                    self.n_hidden,
                                    bert_dim=self.bert_hidden_size,
                                    dropout=self.dropout,
                                    self_loop_init=False,
                                    use_node_embeddings=False)

        self.sparse_embeds = embed_layer.sparse_embeds
        # If there are dense parameters in the embedding layer
        # or we use Pytorch saprse embeddings.
        if len(embed_layer.input_projs) > 0:
            embed_layer = embed_layer.to(dev_id)
            embed_layer = DistributedDataParallel(embed_layer, device_ids=[dev_id], output_device=dev_id)
        self.embed_layer = embed_layer

    def init_emb_optimizer(self):
        g = self.g
        if len(self.sparse_embeds) > 0:
            sparse_emb_optimizer = dgl.distributed.optim.SparseAdam(
                self.sparse_embeds.values(), lr=self.sparse_lr)
            if g.rank() == 0:
                print('optimize DGL sparse embedding.')
        else:
            sparse_emb_optimizer = None
        emb_params = list(self.embed_layer.parameters())
        emb_optimizer = th.optim.Adam(emb_params, lr=self.lr, weight_decay=self.weight_decay)

        self.sparse_emb_optimizer = sparse_emb_optimizer
        self.emb_optimizer = emb_optimizer

    def init_m5gnn_model(self, train=True):
        ''' Initialize the M5GNN model.

        Argument
        --------
        train : bool
            Indicate whether the model is initialized for training.
        '''
        print("Init distributed Language model ...")
        self.init_lm_encoder()
        self.init_dist_decoder()
        self.init_lm_encoder_optimizer()
        self.init_lm_decoder_optimizer()
        if self.pretrain_emb_layer:
            self.init_emb_layer()
            self.init_emb_optimizer()
        self.init_loss_function()

    def save_model(self, epoch, i, g):
        '''Save the model for a certain iteration in an epoch.
        '''
        model_conf = self.model_conf
        decoder = self.decoder
        bert_model = self.bert_model
        assert model_conf is not None
        assert bert_model is not None

        # sync before model saving
        th.distributed.barrier()
        if self.save_model_path is not None and g.rank() == 0:
            start_save_t = time.time()
            save_model_path = self.save_model_path + '-' + str(epoch)
            if i is not None:
                save_model_path = save_model_path + '-' + str(i)
            save_lm_model(model_conf, save_model_path, gnn_model=None, embed_layer=self.embed_layer, bert_model=bert_model,
                          decoder=decoder)
            save_sparse_embeds(save_model_path, self.embed_layer)
            print('successfully save the model to ' + save_model_path)
            print('Time on save model {}'.format(time.time() - start_save_t))

        # wait for rank0 to save the model and/or embeddings
        th.distributed.barrier()



    def compute_embeddings(self, g, target_nidx_per_ntype, emb_cache=None):
        """
        compute node embeddings

        Parameters
        ----------
        g : DGLGraph
            The input graph
        target_nidx_per_ntype: dictionary of tensors
            The idices of nodes to generate embeddings.
        """
        th.distributed.barrier()
        bert_train = self.bert_train
        bert_static = self.bert_static
        bert_hidden_size = self.bert_hidden_size
        device = 'cuda:%d' % self.dev_id
        embeddings={}
        if target_nidx_per_ntype is None:
            target_nidx_per_ntype = {ntype: None for ntype in g.ntypes}
        for target_ntype, target_nidx in target_nidx_per_ntype.items():
            if target_nidx is None:
                target_mask_boolean = th.full((g.num_nodes(target_ntype),), True, dtype=th.bool)
                target_nidx = th.arange(g.num_nodes(target_ntype))
            else:
                target_mask_boolean = th.full((g.num_nodes(target_ntype),), False, dtype=th.bool)
                target_mask_boolean[target_nidx] = True

            dist_target_nidx = dgl.distributed.node_split(target_mask_boolean,
                                                          partition_book=g.get_partition_book(), ntype=target_ntype, force_even=False,
                                                          node_trainer_ids=g.nodes[target_ntype].data['trainer_id'])


            dataloader = DataLoader(dist_target_nidx, batch_size=self.eval_batch_size, shuffle=False)

            dist_embeddings = dgl.distributed.DistTensor((g.number_of_nodes(target_ntype),
                                                          self.n_hidden if self.pretrain_emb_layer else self.bert_hidden_size[target_ntype]),
                                                          dtype=th.float32, name='output_embeddings',
                                                          part_policy=g.get_node_partition_policy(target_ntype),
                                                          persistent=True)
            th.cuda.empty_cache()
            t = time.time()
            with th.no_grad():
                for i, (text_id) in enumerate(dataloader):
                    self.log_param("Dummy", "Keep alive", report_step=i)
                    input_nodes = {target_ntype: text_id}
                    train_mask = {target_ntype: th.full((text_id.shape[0],), False, dtype=th.bool)}
                    inputs, _ = prepare_batch_input(g,
                                                    bert_train,
                                                    bert_static,
                                                    bert_hidden_size,
                                                    input_nodes,
                                                    train_mask=train_mask,
                                                    emb_cache=emb_cache,
                                                    dev=device)

                    if self.pretrain_emb_layer:
                        input_nodes = {ntype: inodes.long().to(device) for ntype, inodes in input_nodes.items()}
                        # TODO support mix precision
                        embs = self.embed_layer(inputs, input_nodes=input_nodes, target_ntype=target_ntype)
                    else:
                        embs = inputs
                    out_embs = embs[target_ntype]
                    dist_embeddings[text_id] = out_embs.type(th.float32).cpu()
            th.distributed.barrier()
            embeddings[target_ntype] = LazyDistTensor(dist_embeddings, target_nidx)
            if g.rank() == 0:
                print("Compute embeddings with mini-batch inference takes : {:.4f}".format(time.time()-t))
        return embeddings

    def print_info(self, epoch, iter, num_input_nodes, compute_time):
        ''' Print basic information during training

        Parameters:
        epoch: int
            The epoch number
        iter: int
            The current iteration
        num_input_nodes: int
            number of input nodes
        compute_time: tuple of ints
            A tuple of (bert forward time, gnn forward time and backward time)
        '''
        if self.debug:
            flops_strs = get_bert_flops_info(self.bert_train, self.bert_static)
            print('Epoch {:05d} | Batch {:03d} | {}'.format(epoch, iter, flops_strs))
        bert_forward_time, gnn_forward_time, back_time = compute_time
        device = 'cuda:%d' % self.dev_id

        print("Epoch {:05d} | Batch {:03d} | GPU Mem reserved: {:.4f} MB | Peak Mem alloc: {:.4f} MB".
                format(epoch, iter,
                    th.cuda.memory_reserved(device) / 1024 / 1024,
                    th.cuda.max_memory_allocated(device) / 1024 /1024))
        print('Epoch {:05d} | Batch {:03d} | RAM memory {} used'.format(epoch, iter, psutil.virtual_memory()))
        print('Epoch {:05d} | Batch {:03d} | Avg input nodes per iter {} | Bert forward {:05f} | GNN forward {:05f} | Backward {:05f}'.format(
            epoch, iter,
            num_input_nodes,
            bert_forward_time,
            gnn_forward_time,
            back_time))

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def eval(self):
        pass

    @abc.abstractmethod
    def save(self):
        pass

    @abc.abstractmethod
    def load(self):
        pass

    @property
    def g(self):
        return self._g

    @property
    def restore_bert_model_path(self):
        return self._restore_bert_model_path

    @property
    def save_embeds_path(self):
        return self._save_embeds_path

    @property
    def g(self):
        return self._g

    @property
    def dev_id(self):
        return self._dev_id

    @property
    def verbose(self):
        return self._verbose

    @property
    def no_validation(self):
        return self._no_validation

    @property
    def local_rank(self):
        return self._local_rank

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def bert_model(self):
        return self._bert_model

    @property
    def n_hidden(self):
        return self._n_hidden

    @property
    def mini_batch_infer(self):
        return self._mini_batch_infer


    @property
    def dropout(self):
        return self._dropout

    @property
    def mixed_precision(self):
        return self._mixed_precision

    @property
    def mp_opt_level(self):
        return self._mp_opt_level


    @property
    def lr(self):
        return self._lr

    @property
    def weight_decay(self):
        return self._weight_decay

    @property
    def bert_lr(self):
        return self._bert_lr

    @property
    def debug(self):
        return self._debug

    @property
    def bert_infer_bs(self):
        return self._bert_infer_bs

    @property
    def train_nodes(self):
        return self._train_nodes

    @property
    def use_bert_cache(self):
        return self._use_bert_cache

    @property
    def refresh_cache(self):
        return self._refresh_cache

    @property
    def gnn_warmup_epochs(self):
        return self._gnn_warmup_epochs

    @property
    def n_epochs(self):
        return self._n_epochs

    @property
    def save_model_per_iters(self):
        return self._save_model_per_iters

    @property
    def save_model_path(self):
        return self._save_model_path

    def register_evaluator(self, evaluator):
        self._evaluator = evaluator

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def eval_batch_size(self):
        return self._eval_batch_size

    @property
    def mlflow_report_frequency(self):
        return self._mlflow_report_frequency

class LanguageModelMLM(LanguageModelBase):
    """Language model using Masked-Language Modeling for fine-tuning

    Note: Huggingface only

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    config: M5GNNConfig
        The M5 GNN configuration
    bert_model: dict
        A dict of BERT models in a format of ntype -> bert_model
        Here the bert_model only has value, i.e., the fine-tuning target.
    """
    def __init__(self, g, config, bert_model, tokenizer):
        super(LanguageModelMLM, self).__init__(g, config, bert_model)

        # setup cuda env
        self.setup_cuda(config.local_rank)
        assert len(bert_model) == 1, "We can only finetune one bert_model at once"
        self.tune_ntype = list(bert_model.keys())[0]
        self.tokenizer = tokenizer
        self.mlm_probability = config.mlm_probability
        self.vocab_size = bert_model[self.tune_ntype].config.vocab_size

    def init_lm_encoder(self):
        g = self.g
        bert_model = self.bert_model
        dev_id = self.dev_id

        text_feat_ntypes = []
        for ntype in g.ntypes:
            if TOKEN_IDX in g.nodes[ntype].data:
                text_feat_ntypes.append(ntype)
        model_conf = {
                'task': 'bert_finetune',
                'tune_ntype': self.tune_ntype
        }

        print("Init distributed bert model ...")
        for ntype in bert_model.keys():
            if self.mixed_precision:
                bert_model[ntype] = apex.parallel.DistributedDataParallel(bert_model[ntype].to(dev_id))
            else:
                bert_model[ntype] = DistributedDataParallel(bert_model[ntype].to(dev_id),
                                                                device_ids=[dev_id], output_device=dev_id)

        self.model_conf = model_conf

    # we can reuse init_lm_encoder_optimizer(self):

    def save_model(self, epoch, i, g):
        '''Save the model for a certain iteration in an epoch.
        '''
        model_conf = self.model_conf
        bert_model = self.bert_model
        assert model_conf is not None
        assert bert_model is not None
        th.distributed.barrier()
        if self.save_model_path is not None and g.rank() == 0:
            start_save_t = time.time()
            save_model_path = self.save_model_path + '-' + str(epoch)
            if i is not None:
                save_model_path = save_model_path + '-' + str(i)

            local_bert_model = {}
            # This tricky we use BertForPreTraining for fine-tuning but use BertModel for end2end training
            # BertModel is BertForPreTraining.bert
            for ntype in bert_model:
                local_bert_model[ntype] = bert_model[ntype].module.bert \
                    if isinstance(bert_model[ntype], DistributedDataParallel) else bert_model[ntype].bert
            save_lm_model(model_conf, save_model_path, gnn_model=None, embed_layer=None, bert_model=local_bert_model, decoder=None)

            print('successfully save the model to ' + save_model_path)
            print('Time on save model {}'.format(time.time() - start_save_t))

        # wait for rank0 to save the model and/or embeddings
        th.distributed.barrier()

    def _mask_tokens(self, inputs):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.

        Following: https://github.com/alontalmor/pytorch-transformers/blob/master/examples/run_lm_finetuning.py
        """
        tokenizer = self.tokenizer[self.tune_ntype]
        mlm_probability = self.mlm_probability

        labels = inputs.clone()
        probability_matrix = th.full(labels.shape, mlm_probability)
        special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        probability_matrix.masked_fill_(th.tensor(special_tokens_mask, dtype=th.bool), value=0.0)
        masked_indices = th.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = th.bernoulli(th.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = th.bernoulli(th.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = th.randint(len(tokenizer), labels.shape, dtype=th.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def evaluate(self, g, bert_model, val_idx, device):
        """ Evaluation func for MLM

        Following: https://github.com/alontalmor/pytorch-transformers/blob/master/examples/run_lm_finetuning.py
        """
        num_nodes = g.num_nodes(self.tune_ntype)
        dist_loss = dgl.distributed.DistTensor((num_nodes,),
                                               dtype=th.float32,
                                               name='lm_loss',
                                               part_policy=g.get_node_partition_policy(self.tune_ntype),
                                               persistent=True)
        val_mask = th.full((num_nodes,), False, dtype=th.bool)
        val_mask[val_idx] = True
        local_val_idx = dgl.distributed.node_split(val_mask,
                                                   g.get_partition_book(),
                                                   ntype=self.tune_ntype)
        print("{}:{}:{}".format(num_nodes, val_idx, local_val_idx))

        loader = DataLoader(local_val_idx, batch_size=self.eval_batch_size, shuffle=False)
        eval_loss = 0.0
        nb_eval_steps = 0

        for iter_l, input_nodes in enumerate(tqdm.tqdm(loader)):
            self.log_param("Dummy", "Keep alive", report_step=iter_l)

            inputs = g.nodes[self.tune_ntype].data[TOKEN_IDX][input_nodes]
            inputs, labels = self._mask_tokens(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)

            with th.no_grad():
                outputs = bert_model(inputs, labels=labels)
                prediction_scores = outputs.prediction_logits
                loss_fct = th.nn.CrossEntropyLoss()
                lm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), labels.view(-1))
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        # save loss into dist_loss so trainer 0 can access it
        # The implementation is tricky here. We save the loss of each trainer
        # into the dist_loss[trainer_rank_idx]
        dist_loss[th.distributed.get_rank()] = th.tensor(eval_loss).unsqueeze(0)

        # wait for all trainer to finish evaluation
        th.distributed.barrier()
        if g.rank() == 0:
            # load all losses in trainer 0
            num_trainers = th.distributed.get_world_size()
            eval_loss = dist_loss[th.arange(num_trainers)].mean()
            perplexity = th.exp(th.tensor(eval_loss))

            # return perplexity
            return perplexity.item()
        #dummy
        return 0

    def fit(self, batch_size, train_data):
        g = self.g
        device = 'cuda:%d' % self.dev_id

        assert self.tune_ntype == train_data.tune_ntype
        train_idx = train_data._train_idx
        val_idx = train_data._val_idx

        loader = DataLoader(train_idx, batch_size=batch_size, shuffle=True)

        bert_model = self.bert_model[self.tune_ntype]
        tr_loss, logging_loss = 0.0, 0.0
        eval_history = []
        print("start training...")
        iter_local = 0
        for epoch in range(self.n_epochs):
            bert_model.train()

            batch_tic = time.time()
            for i, (input_nodes) in enumerate(loader):
                iter_local += 1
                inputs = g.nodes[self.tune_ntype].data[TOKEN_IDX][input_nodes]
                attention_mask = g.nodes[self.tune_ntype].data[ATT_MASK_IDX][input_nodes]
                inputs, labels = self._mask_tokens(inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = bert_model(inputs, attention_mask=attention_mask, labels=labels)
                prediction_scores = outputs.prediction_logits
                loss_fct = th.nn.CrossEntropyLoss()
                mlm_loss = loss_fct(prediction_scores.view(-1, self.vocab_size), labels.view(-1))
                # ignore the seq_loss of BertForPreTraining
                # seq_loss is not usd.
                seq_loss = outputs.seq_relationship_logits.mean() * 0
                loss = mlm_loss + seq_loss

                tr_loss += loss.item()
                if self.mixed_precision:
                    with apex.amp.scale_loss(loss, self.fine_tune_opt) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if self.mixed_precision:
                    th.nn.utils.clip_grad_norm_(apex.amp.master_params(self.fine_tune_opt), 1.0)
                else:
                    th.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
                self.fine_tune_opt.step()
                bert_model.zero_grad()
                self.log_metric("Train loss", loss.item(), iter_local, report_step=iter_local)

                if g.rank() == 0 and i % 100 == 0:
                    # Log metrics
                    print("Part {} | Epoch {:05d} | Batch {:03d} | Train Loss {:.4f} | Time: {:.4f}".
                            format(g.rank(), epoch, i, (tr_loss - logging_loss) / 100, (time.time() - batch_tic) / 100))
                    logging_loss = tr_loss
                    batch_tic = time.time()

            th.cuda.empty_cache()
            th.distributed.barrier()
            # do evaluation in a distributed way.
            result = self.evaluate(g, bert_model, val_idx, device)
            if g.rank() == 0:
                eval_history.append((epoch, i, result))
                print("Epoch {:05d} MLM Eval perplexity {}".format(epoch, result))
                self.log_metric("MLM perplexity", result, iter_local)
            th.distributed.barrier()

            # save model
            self.save_model(epoch, i, g)

        if g.rank() == 0:
            if self.verbose:
                print("Eval history:".format(eval_history))
