"""model support for hugging face bert"""
import torch as th
import torch.nn as nn
import numpy as np
import time

from transformers import BertModel, BertForPreTraining, BertConfig
from torch.nn.parallel import DistributedDataParallel

from ..data.constants import TOKEN_IDX, VALID_LEN_IDX, ATT_MASK_IDX, TOKEN_TID_IDX

def is_distributed(bert_model):
    return isinstance(bert_model, DistributedDataParallel) \
            or isinstance(bert_model, nn.DataParallel)

def run_bert(bert_model, input_ids, attention_mask, token_type_ids, labels=None, use_bert_loss=False, dev='cpu'):
    """ Execute bert

    Parameters
    ----------
    bert_model: Bert model, it can be a HuggingFace Bert
    input_ids: Input token ids
    attention_mask: Attention mask, it can be a 2D bool mask tensor
        indicating attention masks for each sequence or an integer
        tensor indicating the valid sequence length.
    token_type_ids: Token type id.
    labels: Labels for computing the masked language modeling loss.
    use_bert_loss: Whether return bert loss.
    dev: Output device.
    """
    loss = None
    if isinstance(bert_model, BertForPreTraining) or \
        (is_distributed(bert_model) and isinstance(bert_model.module, BertForPreTraining)):
        # For huggingface BertForPreTraining, attention_mask should have
        # the same shape as input_ids.
        # The data processing process may compress attention mask into
        # a single int for each input sequence. We need to rebuild
        # the attention mask here.
        if len(attention_mask.shape) == 1:
            length = input_ids.shape[1]
            attention_mask = attention_mask.long().to(input_ids.device)
            att_mask = th.arange(0, length, device=input_ids.device)
            attention_mask = att_mask.reshape((1, -1)) < attention_mask.reshape((-1, 1))

        outputs = bert_model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             labels=labels,
                             next_sentence_label=th.full((labels.shape[0],), 1, dtype=th.long).to(dev),
                             output_hidden_states=True)
        if isinstance(outputs, dict):
            hidden_states = outputs['hidden_states']
        else:
            hidden_states = outputs.hidden_states
        out_embs = hidden_states[0]
        out_embs = th.mean(out_embs, dim=1)
    elif isinstance(bert_model, BertModel) or \
        (is_distributed(bert_model) and isinstance(bert_model.module, BertModel)):
        # For huggingface BertForPreTraining, attention_mask should have
        # the same shape as input_ids.
        # The data processing process may compress attention mask into
        # a single int for each input sequence. We need to rebuild
        # the attention mask here.
        if len(attention_mask.shape) == 1:
            length = input_ids.shape[1]
            attention_mask = attention_mask.long().to(input_ids.device)
            att_mask = th.arange(0, length, device=input_ids.device)
            attention_mask = att_mask.reshape((1, -1)) < attention_mask.reshape((-1, 1))

        outputs = bert_model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        if isinstance(outputs, dict):
            out_embs = outputs['pooler_output']
        else:
            out_embs = outputs.pooler_output
    else:
        assert False, 'wrong bert model'
    if use_bert_loss:
        loss = outputs.loss

    return out_embs, loss

def extract_bert_embed(nid, mask, bert_train, bert_static, bert_hidden_size, dev, verbose=False):
    """ Generate bert embeddings.

    The nid gives a list of nodes to generate bert embeddings. The mask is used to specify which nodes are
    trainable text nodes that back-propagation is needed. We use bert_train to generate embeddings for
    trainable text nodes and use bert_static to generate embeddings for non-trainable text nodes.

    Parameters
    ----------
    nid : th.Tensor
        Node IDs
    mask: th.Tensor
        A mask specifying which nodes' text is trainable.
    bert_train: Wrapper for trainable bert
        The bert model wrapper used to generate embeddings for trainable text nodes.
    bert_static: Wrapper for static bert
        The bert model wrapper used to generate embeddings for static text bides.
    bert_hidden_size: int
        The bert embedding hidden size.
    dev: th.device
        Device to put output in.
    verbose: bool
        Whether to print extra infor
    """
    train_nodes = nid[mask] if mask is not None else nid
    static_nodes = nid[~mask] if mask is not None else None
    text_embs = th.empty((nid.shape[0], bert_hidden_size), dtype=th.float32, device=dev)
    num_text_nodes = 0

    loss = None
    if verbose:
        print("{} text nodes".format(nid.shape[0]))

    if train_nodes.shape[0] > 0:
        num_text_nodes += len(train_nodes)
        train_nodes = train_nodes.to(dev)

        if verbose:
            print("{} trainable text nodes".format(train_nodes.shape[0]))

        if isinstance(bert_train, MGBertLossWrapper):
            out_embs, loss = bert_train(train_nodes)
        else:
            loss = None
            out_embs = bert_train(train_nodes)

        if mask is None:
            text_embs = out_embs.type(th.float32)
        else:
            text_embs[mask] = out_embs.type(th.float32)

    if static_nodes is not None and static_nodes.shape[0] > 0:
        static_nodes = static_nodes.to(dev)
        if verbose:
            print("{} static text nodes".format(static_nodes.shape[0]))
        out_embs = bert_static(static_nodes)
        text_embs[~mask] = out_embs.type(th.float32)

    return text_embs, loss

class MGBertWrapper(nn.Module):
    def __init__(self,
                 bert,
                 raw_feats,
                 debug=False):
        super(MGBertWrapper, self).__init__()
        self.bert = bert
        self.raw_feats = raw_feats
        self.debug = debug
        if self.debug:
            if isinstance(bert, nn.parallel.DistributedDataParallel):
                self.num_params = np.sum([param.numel() for param in bert.module.encoder.parameters()])
            else:
                self.num_params = np.sum([param.numel() for param in bert.encoder.parameters()])
        self.max_seq_lens = []
        self.flops = []

    def get_avg_seq_len(self):
        return np.mean(self.max_seq_lens)

    def get_avg_flops(self):
        return np.mean(self.flops)

    def forward(self, input_idx):
        dev = input_idx.device
        input_ids = labels = self.raw_feats[TOKEN_IDX][input_idx].to(dev)
        # If ATT_MASK_IDX does not exist, we expect the VALID_LEN_IDX
        # stores the valid token length (We can build attention mask based
        # on it.).
        attention_mask = self.raw_feats[ATT_MASK_IDX][input_idx].to(dev) \
            if ATT_MASK_IDX in self.raw_feats \
            else self.raw_feats[VALID_LEN_IDX][input_idx].to(dev)
        # token type ids are not used in most cases
        token_type_ids = None if (TOKEN_TID_IDX not in self.raw_feats) \
            else self.raw_feats[TOKEN_TID_IDX][input_idx].to(dev)

        if self.debug:
            t1 = time.time()

        out_embs, _ = run_bert(self.bert, input_ids, attention_mask, token_type_ids, labels=labels, use_bert_loss=False, dev=dev)
        if self.debug:
            th.cuda.synchronize(device=self.bert.device)
            t2 = time.time()
            seq_len = input_ids.shape[1]
            num_flops = 2 * self.num_params * seq_len * len(input_idx)
            self.max_seq_lens.append(seq_len)
            self.flops.append(num_flops / (t2 - t1))
        return out_embs

class StaticMGBertWrapper(nn.Module):
    def __init__(self,
                 bert,
                 raw_feats,
                 batch_size=128,
                 debug=False):
        super(StaticMGBertWrapper, self).__init__()
        self.bert = bert
        self.raw_feats = raw_feats
        self.batch_size = batch_size
        self.debug = debug
        if self.debug:
            if isinstance(bert, nn.parallel.DistributedDataParallel):
                self.num_params = np.sum([param.numel() for param in bert.module.encoder.parameters()])
            else:
                self.num_params = np.sum([param.numel() for param in bert.encoder.parameters()])
        self.max_seq_lens = []
        self.flops = []

    def get_avg_seq_len(self):
        return np.mean(self.max_seq_lens)

    def get_avg_flops(self):
        return np.mean(self.flops)

    def forward(self, input_idx):
        self.bert.eval()
        dev = input_idx.device
        input_ids = self.raw_feats[TOKEN_IDX][input_idx].to(dev)
        # If ATT_MASK_IDX does not exist, we expect the VALID_LEN_IDX
        # stores the valid token length (We can build attention mask based
        # on it.).
        attention_mask = self.raw_feats[ATT_MASK_IDX][input_idx].to(dev) \
            if ATT_MASK_IDX in self.raw_feats \
            else self.raw_feats[VALID_LEN_IDX][input_idx].to(dev)
        # token type ids are not used in most cases
        token_type_ids = None if TOKEN_TID_IDX not in self.raw_feats \
            else self.raw_feats[TOKEN_TID_IDX][input_idx].to(dev)

        with th.no_grad():
            input_ids_list = th.split(input_ids, self.batch_size, dim=0)
            attention_mask_list = th.split(attention_mask, self.batch_size, dim=0)
            token_type_ids_list = [None] * len(input_ids_list) if token_type_ids is None \
                else th.split(token_type_ids, self.batch_size, dim=0)
            out_embs_list = []

            if self.debug:
                t1 = time.time()

            num_flops = 0
            for input_ids, attention_mask, token_type_ids in zip(input_ids_list, attention_mask_list, token_type_ids_list):
                out_embs, _ = run_bert(self.bert, input_ids, attention_mask, token_type_ids, labels=input_ids, use_bert_loss=False, dev=dev)
                out_embs_list.append(out_embs)
                seq_len = input_ids.shape[1]
                self.max_seq_lens.append(seq_len)
                if self.debug:
                    num_flops += 2 * self.num_params * seq_len * len(input_ids)

            if self.debug:
                th.cuda.synchronize(device=self.bert.device)
                t2 = time.time()
                self.flops.append(num_flops / (t2 - t1))

            out_embs = th.cat(out_embs_list, dim=0)
        self.bert.train()
        return out_embs

class MGBertLossWrapper(nn.Module):
    def __init__(self,
                 bert,
                 raw_feats,
                 use_bert_loss=False):
        super(MGBertLossWrapper, self).__init__()
        self.bert = bert
        self.raw_feats = raw_feats
        self.use_bert_loss = use_bert_loss

    def forward(self, input_idx):
        dev = input_idx.device
        input_ids = labels = self.raw_feats[TOKEN_IDX][input_idx].to(dev)
        # If ATT_MASK_IDX does not exist, we expect the VALID_LEN_IDX
        # stores the valid token length (We can build attention mask based
        # on it.).
        attention_mask = self.raw_feats[ATT_MASK_IDX][input_idx].to(dev) \
            if ATT_MASK_IDX in self.raw_feats \
            else self.raw_feats[VALID_LEN_IDX][input_idx].to(dev)
        token_type_ids = None if (TOKEN_TID_IDX not in self.raw_feats) \
            else self.raw_feats[TOKEN_TID_IDX][input_idx].to(dev)
        out_embs, loss = run_bert(self.bert, input_ids, attention_mask, token_type_ids, labels=labels, use_bert_loss=self.use_bert_loss, dev=dev)

        return out_embs, loss

class NtypeMGBertWrapper(nn.Module):
    r""" Wripper of Bert model for distributed training.

    Multiple node types share the same Bert model. The input nodes are used
    for bert finetuning.

    Parameters
    ----------
    bert : Bert
        The bert model.
    raw_feats : dict of dict of tensors.
        The tokenized text features of different node types.
        For each node type, there are three features: input_ids, attention_mask and token_type_ids.
    """
    def __init__(self,
                 bert,
                 raw_feats):
        super(NtypeMGBertWrapper, self).__init__()
        self.bert = bert
        self.raw_feats = raw_feats

    def forward(self, input_idx, dev):
        """
        Parameters
        ----------
        input_idx : dict of tensor
            Input text nodes for bert finetuning.
        dev: torch device
            Device to hold the output.

        Returns
        -------
        Dict of tensor
            Bert embeddings correspond to input nodes.
        """
        input_ids = []
        attention_masks = []
        token_type_ids = []
        for ntype, idx in input_idx.items():
            input_id = self.raw_feats[ntype][TOKEN_IDX][idx].to(dev)
            # If ATT_MASK_IDX does not exist, we expect the VALID_LEN_IDX
            # stores the valid token length (We can build attention mask based
            # on it.).
            attention_mask = self.raw_feats[ntype][ATT_MASK_IDX][idx].to(dev) \
                if ATT_MASK_IDX in self.raw_feats[ntype] \
                else self.raw_feats[ntype][VALID_LEN_IDX][idx].to(dev)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            if TOKEN_TID_IDX in self.raw_feats[ntype]:
                token_type_id = self.raw_feats[ntype][TOKEN_TID_IDX][idx].to(dev)
                token_type_ids.append(token_type_id)
        input_ids = th.cat(input_ids, dim=0)
        attention_masks = th.cat(attention_masks, dim=0)
        token_type_ids = None if len(token_type_ids) == 0 else th.cat(token_type_ids, dim=0)
        out_emb, _ = run_bert(self.bert, input_ids, attention_masks, token_type_ids, labels=input_ids, use_bert_loss=False, dev=dev)

        offset = 0
        out_embs = {}
        for ntype, idx in input_idx.items():
            out_embs[ntype] = out_emb[offset:offset+idx.shape[0]]
            offset += offset+idx.shape[0]

        return out_embs

class NtypeStaticMGBertWrapper(nn.Module):
    r""" Wripper of Bert model for distributed training.

    Multiple node types share the same Bert model. The input nodes are used
    for bert inference.


    Parameters
    ----------
    bert : Bert
        The bert model.
    raw_feats : dict of dict of tensors
        The tokenized text features of different node types.
        For each node type, there are three features: input_ids, attention_mask and token_type_ids.
    batch_size : int
        Used for batched inference.
    """
    def __init__(self,
                 bert,
                 raw_feats,
                 batch_size=128):
        super(NtypeStaticMGBertWrapper, self).__init__()
        self.bert = bert
        self.raw_feats = raw_feats
        self.batch_size = batch_size

    def forward(self, input_idx, dev):
        """
        Parameters
        ----------
        input_idx : dict of tensor
            Input text nodes for bert inference.
        dev : torch device
            Device to hold the output.

        Returns
        -------
        Dict of tensor
            Bert embeddings correspond to input nodes.
        """
        input_ids = []
        attention_masks = []
        self.bert.eval()
        token_type_ids = []
        for ntype, idx in input_idx.items():
            if idx.shape[0] == 0:
                continue
            input_id = self.raw_feats[ntype][TOKEN_IDX][idx].to(dev)
            # If ATT_MASK_IDX does not exist, we expect the VALID_LEN_IDX
            # stores the valid token length (We can build attention mask based
            # on it.).
            attention_mask = self.raw_feats[ntype][ATT_MASK_IDX][idx].to(dev) \
                if ATT_MASK_IDX in self.raw_feats[ntype] \
                else self.raw_feats[ntype][VALID_LEN_IDX][idx].to(dev)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            if TOKEN_TID_IDX in self.raw_feats[ntype]:
                token_type_id = self.raw_feats[ntype][TOKEN_TID_IDX][idx].to(dev)
                token_type_ids.append(token_type_id)
        input_ids = th.cat(input_ids, dim=0)
        attention_masks = th.cat(attention_masks, dim=0)
        token_type_ids = None if len(token_type_ids) == 0 else th.cat(token_type_ids, dim=0)

        with th.no_grad():
            input_ids_list = th.split(input_ids, self.batch_size, dim=0)
            attention_mask_list = th.split(attention_masks, self.batch_size, dim=0)
            token_type_ids_list = th.split(token_type_ids, self.batch_size, dim=0)
            out_embs_list = []
            for input_ids, attention_masks, token_type_ids in zip(input_ids_list, attention_mask_list, token_type_ids_list):
                out_emb, _ = run_bert(self.bert, input_ids, attention_masks, token_type_ids, labels=input_ids, use_bert_loss=False, dev=dev)
                out_embs_list.append(out_emb)

            out_emb = th.cat(out_embs_list, dim=0)

        offset = 0
        out_embs = {}
        for ntype, idx in input_idx.items():
            if idx.shape[0] == 0:
                continue
            out_embs[ntype] = out_emb[offset:offset+idx.shape[0]]
            offset += offset+idx.shape[0]
        self.bert.train()
        return out_embs

class NtypeMGBertLossWrapper(nn.Module):
    r""" Wripper of Bert model for distributed training.

    Multiple node types share the same Bert model. The input nodes are used
    for bert finetuning. The wrapper also returns the bert loss.

    Parameters
    ----------
    bert : Bert
        The bert model.
    raw_feats : dict of dict of tensors.
        The tokenized text features of different node types.
        For each node type, there are three features: input_ids, attention_mask and token_type_ids.
    """
    def __init__(self,
                 bert,
                 raw_feats):
        super(NtypeMGBertLossWrapper, self).__init__()
        self.bert = bert
        self.raw_feats = raw_feats

    def forward(self, input_idx, dev):
        """
        Parameters
        ----------
        input_idx : Dict of tensor
            Input text nodes for bert inference.
        dev: Torch device
            Device to hold the output.

        Returns
        -------
        Dict of tensor
            Bert embeddings correspond to input nodes.
        Tensor
            Bert loss
        """
        input_ids = []
        attention_masks = []
        token_type_ids = []
        for ntype, idx in input_idx.items():
            input_id = self.raw_feats[ntype][TOKEN_IDX][idx].to(dev)
            # If ATT_MASK_IDX does not exist, we expect the VALID_LEN_IDX
            # stores the valid token length (We can build attention mask based
            # on it.).
            attention_mask = self.raw_feats[ntype][ATT_MASK_IDX][idx].to(dev) \
                if ATT_MASK_IDX in self.raw_feats[ntype] \
                else self.raw_feats[ntype][VALID_LEN_IDX][idx].to(dev)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            if TOKEN_TID_IDX in self.raw_feats[ntype]:
                token_type_id = self.raw_feats[ntype][TOKEN_TID_IDX][idx].to(dev)
                token_type_ids.append(token_type_id)
        input_ids = th.cat(input_ids, dim=0)
        attention_masks = th.cat(attention_masks, dim=0)
        token_type_ids = None if len(token_type_ids) == 0 else th.cat(token_type_ids, dim=0)
        out_emb, loss = run_bert(self.bert, input_ids, attention_masks, token_type_ids, labels=input_ids, use_bert_loss=True, dev=dev)

        offset = 0
        out_embs = {}
        for ntype, idx in input_idx.items():
            out_embs[ntype] = out_emb[offset:offset+idx.shape[0]]
            offset += offset+idx.shape[0]
        return out_embs, loss

def init_bert(gradient_checkpointing=False, use_bert_loss=False, verbose=False, bert_model_name='bert-base-uncased'):
    # the max_position_embeddings can not be changed at this point because we load pretrained models.
    # if the input sequence has length larger that the standart there is a size missmatch and the code fails
    # This problem should be taken care of by the tokenizer
    if use_bert_loss:
        print('BertForPreTraining')
        config = BertConfig.from_pretrained(bert_model_name, gradient_checkpointing=gradient_checkpointing)
        bert_model = BertForPreTraining.from_pretrained(bert_model_name, config=config)
    else:
        print('BertModel')
        config = BertConfig.from_pretrained(bert_model_name, gradient_checkpointing=gradient_checkpointing)
        bert_model = BertModel.from_pretrained(bert_model_name, config=config)

    return bert_model

def freeze_bert(bert_model, freeze_layers):
    if len(freeze_layers) > 0:
        for param in list(bert_model.embeddings.parameters()):
            param.requires_grad = False
            print("Freeze Embedding Layer")
        layer_indices = [int(l) for l in freeze_layers]
        for idx in layer_indices:
            for param in list(bert_model.encoder.layer[idx].parameters()):
                param.requires_grad = False
            print("Freeze Layer: {}".format(idx))

def wrap_bert(g, bert_model, use_bert_loss=False, bert_infer_bs=32, debug=False):
    # create Bert model wrapper.
    bert_train = {}
    bert_static = {}
    if isinstance(bert_model, dict):
        for ntype in bert_model.keys():
            if g.rank() == 0:
                print('node {} gets a separate BERT model'.format(ntype))
            if use_bert_loss:
                bert_train[ntype] = MGBertLossWrapper(
                        bert_model[ntype], g.nodes[ntype].data)
            else:
                bert_train[ntype] = MGBertWrapper(
                        bert_model[ntype], g.nodes[ntype].data, debug=debug)
            bert_static[ntype] = StaticMGBertWrapper(
                    bert_model[ntype], g.nodes[ntype].data, bert_infer_bs, debug=debug)
    else: # only have one bert model
        ntext_data = {}
        for ntype in g.ntypes:
            if g.rank() == 0:
                print('node {} gets the shared BERT model'.format(ntype))
            if TOKEN_IDX in g.nodes[ntype].data:
                ntext_data[ntype] = g.nodes[ntype].data
        assert len(ntext_data) > 0
        if use_bert_loss:
            bert_train = NtypeMGBertLossWrapper(
                bert_model, ntext_data)
        else:
            bert_train = NtypeMGBertWrapper(
                bert_model, ntext_data)
        bert_static = NtypeStaticMGBertWrapper(
            bert_model, ntext_data, bert_infer_bs)

    return bert_train, bert_static

def get_bert_flops_info(bert_train, bert_static):
    flops_strs = []
    for ntype in bert_train:
        flops_strs.append('train {}: max seq len: {:.1f}, {:.3f} TFLOPs'.format(ntype,
            bert_train[ntype].get_avg_seq_len(), bert_train[ntype].get_avg_flops()/1024/1024/1024/1024))
    for ntype in bert_static:
        flops_strs.append('static {}: max seq len: {:.1f}, {:.3f} TFLOPs'.format(ntype,
            bert_static[ntype].get_avg_seq_len(), bert_static[ntype].get_avg_flops()/1024/1024/1024/1024))

    return flops_strs
