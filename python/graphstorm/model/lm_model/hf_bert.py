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

    Huggingface bert support
"""
import time
import inspect

import numpy as np
import torch as th
from torch import nn

from transformers import BertModel, BertConfig

from .lm_model import TOKEN_IDX, ATT_MASK_IDX, VALID_LEN, TOKEN_TID_IDX
from .lm_model import GSFLanguageModelWrapper

def load_hfbert_model(bert_configs):
    """ Load huggingface bert model
    """
    model_name = bert_configs["model_name"]
    gradient_checkpointing = bert_configs["gradient_checkpoint"]
    signature = inspect.signature(BertConfig.__init__)
    hidden_dropout_prob = bert_configs["hidden_dropout_prob"] \
        if "hidden_dropout_prob" in bert_configs \
        else signature.parameters['hidden_dropout_prob'].default
    attention_probs_dropout_prob = bert_configs["attention_probs_dropout_prob"] \
        if "attention_probs_dropout_prob" in bert_configs \
        else signature.parameters['attention_probs_dropout_prob'].default
    config = BertConfig.from_pretrained(model_name,
                                        gradient_checkpointing=gradient_checkpointing,
                                        hidden_dropout_prob=hidden_dropout_prob,
                                        attention_probs_dropout_prob=attention_probs_dropout_prob)

    lm_model = BertModel.from_pretrained(model_name, config=config)
    return lm_model

class HFBertWrapper(GSFLanguageModelWrapper):
    """ Wrap huggingface BertModel.

    Parameters
    ----------
    bert_model: transformers.BertModel
        Huggingface Bert.
    num_train: int
        Number of trainable texts
    bert_infer_bs: int
        Batch size used for computing text embeddings for static bert
    profile: bool
        If True, compute flops statistics.
    """
    def __init__(self,
                 lm_model,
                 num_train,
                 lm_infer_batch_size=32,
                 profile=False):
        super(HFBertWrapper, self).__init__(
            lm_model, num_train,
            lm_model.config.hidden_size, [TOKEN_IDX, ATT_MASK_IDX, VALID_LEN, TOKEN_TID_IDX],
            lm_infer_batch_size, profile)

        self.origin_num_train = self.num_train
        assert isinstance(self.lm_model, BertModel), \
            "Language model must be transformers.BertModel"
        if self.profile:
            if isinstance(lm_model, nn.parallel.DistributedDataParallel):
                self.num_params = np.sum([param.numel() \
                    for param in lm_model.module.encoder.parameters()])
            else:
                self.num_params = np.sum([param.numel() for param in lm_model.encoder.parameters()])

    def _forward(self, input_ids, attention_masks, token_tids=None):
        outputs = self.lm_model(input_ids,
                                attention_mask=attention_masks,
                                token_type_ids=token_tids)
        if isinstance(outputs, dict):
            out_emb = outputs['pooler_output']
        else:
            out_emb = outputs.pooler_output
        return out_emb

    def _train_forward(self, input_ids, attention_masks, token_tids=None):
        if self.profile:
            t_train_start = time.time()

        out_emb = self._forward(input_ids, attention_masks, token_tids)

        if self.profile:
            th.cuda.synchronize(device=self.lm_model.device)
            seq_len = input_ids.shape[1]
            self.max_train_seq_lens.append(seq_len)
            train_num_flops = 2 * self.num_params * seq_len * len(input_ids)
            self.train_flops.append(train_num_flops / (time.time() - t_train_start))
        return out_emb

    def _static_forward(self, input_ids, attention_masks, token_tids=None):
        if self.profile:
            static_num_flops = 0
            t_static_start = time.time()

        if self.training:
            # only in traininig mode, we need to set lm_model to eval temporarily
            self.lm_model.eval()

        input_ids_list = th.split(input_ids, self.infer_bs, dim=0)
        attention_mask_list = th.split(attention_masks, self.infer_bs, dim=0)
        token_tids_list = th.split(token_tids, self.infer_bs, dim=0) \
            if token_tids is not None else [None] * len(input_ids_list)
        static_out_embs = []
        for static_iid, static_att_mask, static_token_tid \
            in zip(input_ids_list, attention_mask_list, token_tids_list):
            out_emb = self._forward(static_iid, static_att_mask, static_token_tid)
            static_out_embs.append(out_emb)

            if self.profile:
                seq_len = static_iid.shape[1]
                self.max_static_seq_lens.append(seq_len)
                static_num_flops += 2 * self.num_params * seq_len * len(static_iid)
        static_out_embs = th.cat(static_out_embs, dim=0)
        if self.training:
            self.lm_model.train()

        if self.profile:
            th.cuda.synchronize(device=self.lm_model.device)
            self.static_flops.append(static_num_flops / (time.time() - t_static_start))
        return static_out_embs

    def forward(self, input_ntypes, input_lm_feats):
        """ Forward

        Parameters
        ----------
        input_ntypes: list of str
            A list of input node types
        input_lm_feats: dict of dict of tensors
            Input language model related node features

        Return
        ------
        dict of tensor
            Node type -> text embedding (torch tensor)
        """
        if self.training:
            # do train()
            num_train = self.num_train
        else:
            # do eval()
            num_train = 0

        input_ids = []
        attention_masks = []
        token_tids = []
        dev = next(self.lm_model.parameters()).device
        input_id_lens = []

        for ntype in input_ntypes:
            input_id = input_lm_feats[ntype][TOKEN_IDX].to(dev)
            input_id_lens.append(input_id.shape[0])
            # If ATT_MASK_IDX does not exist, we expect the VALID_LEN
            # stores the valid token length
            if ATT_MASK_IDX in input_lm_feats[ntype]:
                # Get ATT_MASK. In some cases, ATT_MASK_IDX actually stores the Valid
                # mask lenght, i.e., len(attention_mask.shape) == 1, we need to convert it
                # into ATT_MASK_IDX.
                attention_mask = input_lm_feats[ntype][ATT_MASK_IDX].to(dev)
                if len(attention_mask.shape) == 1:
                    length = input_id.shape[1]
                    attention_mask = attention_mask.long()
                    att_mask = th.arange(0, length, device=input_id.device)
                    attention_mask = att_mask.reshape((1, -1)) < attention_mask.reshape((-1, 1))
            else:
                # Rebuild attention mask based on VALID_LEN.
                valid_len = input_lm_feats[ntype][VALID_LEN].to(dev)
                assert len(valid_len.shape) == 1
                length = input_id.shape[1]
                valid_len = valid_len.long()
                att_mask = th.arange(0, length, device=input_id.device)
                attention_mask = att_mask.reshape((1, -1)) < valid_len.reshape((-1, 1))

            input_ids.append(input_id)
            attention_masks.append(attention_mask.long())
            if TOKEN_TID_IDX in input_lm_feats[ntype]:
                token_tid = input_lm_feats[ntype][TOKEN_TID_IDX].to(dev)
                token_tids.append(token_tid.long())

        input_ids = th.cat(input_ids, dim=0)
        attention_masks = th.cat(attention_masks, dim=0)
        if len(token_tids) > 0:
            token_tids = th.cat(token_tids, dim=0)
            assert input_ids.shape[0] == token_tids.shape[0]
        else:
            token_tids = None

        if num_train == 0: # do static bert
            with th.no_grad():
                text_embs = self._static_forward(input_ids,
                                                attention_masks,
                                                token_tids)
        elif num_train == -1 or num_train > input_ids.shape[0]:
            # All nodes are used for training
            text_embs = self._train_forward(input_ids,
                                            attention_masks,
                                            token_tids)
        else: # randomly sample num_train nodes for training
            num_texts = input_ids.shape[0]
            text_embs = th.empty((num_texts, self.feat_size), dtype=th.float32, device=dev)

            train_idx = th.randint(num_texts, (num_train,), device=dev)
            train_input_ids = input_ids[train_idx]
            train_attention_masks = attention_masks[train_idx]
            train_token_tids = token_tids[train_idx] \
                if token_tids is not None else None
            train_out_embs = self._train_forward(train_input_ids,
                                                 train_attention_masks,
                                                 train_token_tids)
            text_embs[train_idx] = train_out_embs.type(th.float32)

            with th.no_grad():
                static_idx = th.full((num_texts,), True, dtype=th.bool, device=dev)
                static_idx[train_idx] = False
                static_input_ids = input_ids[static_idx]
                static_attention_masks = attention_masks[static_idx]
                static_token_tids = token_tids[static_idx] \
                    if token_tids is not None else None
                static_out_embs = self._static_forward(static_input_ids,
                                                       static_attention_masks,
                                                       static_token_tids)
                text_embs[static_idx] = static_out_embs.type(th.float32)

        offset = 0
        out_embs = {}
        for ntype, input_id_len in zip(input_ntypes, input_id_lens):
            out_embs[ntype] = text_embs[offset:offset+input_id_len]
            offset += input_id_len
        return out_embs

def wrap_hf_bert(bert_model, num_train=0, bert_infer_bs=32, profile=False):
    """ Wrap huggingface BertModel.

    Parameters
    ----------
    bert_model: transformers.BertModel
        Huggingface Bert.
    num_train: int
        Number of trainable texts
    bert_infer_bs: int
        Batch size used for computing text embeddings for static bert
    profile: bool
        If True, compute flops statistics.
    """
    assert isinstance(bert_model, BertModel), "Must be huggingface bert"
    # create Bert model wrapper.
    return HFBertWrapper(bert_model, num_train, bert_infer_bs, profile)
