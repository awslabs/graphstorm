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

    LLM-GNNs implementation.
"""
import dgl
import graphstorm as gs
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging

from graphstorm import model as gsmodel
from graphstorm.model.lm_model import TOKEN_IDX, ATT_MASK_IDX
from graphstorm.model.rgcn_encoder import RelationalGCNEncoder

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModel,
)

def get_lm_node_feats(g, lm_feat_names, lm_ntypes):
    """ Collect language model related node features

    Parameters
    ----------
    g: graph
        Graph data.
    lm_feat_names: list of str
        A list of llm features
    lm_ntypes: list of str
        A list of node types.

    Return
    ------
    A dict of dict of distributed tensor.
        {Node type: {Feature name: Feature stored as distributed tensor}}
    """
    lm_feats = {}
    for ntype in lm_ntypes:
        lm_feats[ntype] = {}
        for lm_fname in lm_feat_names:
            if lm_fname in g.nodes[ntype].data:
                lm_feats[ntype][lm_fname] = g.nodes[ntype].data[lm_fname]

    return lm_feats


class LLMGraphModel(gsmodel.GSgnnNodeModelBase):
    """ LLMGNN Model Class

    Parameters
    ----------
    g : dgl DistGraph
        Input DistDGL graph
    node_lm_configs : list
        language model config for each node type
    h_dim : int
        hidden dimension of GNN encoder
    out_dim : int
        Output dimension of the model, e.g. number of classes.
    num_layers : int
        Number of GNN encoder layers.
    target_ntype : str
        The node type for prediction.
    use_norm: boolean, optional
        If use layer normalization or not. Default: True
    alpha_l2norm: float, optional
        The alpha for L2 normalization. Default: 0
    lr : float, optional
        Normalization Method. Default: 0.001
    """
    def __init__(self, g, node_lm_configs, h_dim, out_dim, num_layers,
                 target_ntype,     # the node type to be predict
                 use_norm = True,   # use normalization or not, default is True
                 alpha_l2norm = 0,
                 lr = 0.001):
        super(LLMGraphModel, self).__init__()
        self.num_layers = num_layers
        self.target_ntype=target_ntype
        self.alpha_l2norm = alpha_l2norm
        self.lr = lr
        # assume only one LLM is used
        model_id = node_lm_configs[0]["model_name"]
        self.config = AutoConfig.from_pretrained(model_id)
        base_model = AutoModel.from_pretrained(
            model_id,config=self.config
        )
        peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
        self.llm = get_peft_model(base_model, peft_config)
        lm_feat_names = [TOKEN_IDX, ATT_MASK_IDX]
        self._lm_node_feats = {}
        for lm_config in node_lm_configs:
            # A list of node types sharing the same lm model
            lm_ntypes = lm_config["node_types"]
            if lm_config["model_name"] != model_id:
                logging.warning("Multiple LLM model names are found in the config. However, the example only supports one LLM.")
            lm_node_feats = get_lm_node_feats(g, lm_feat_names, lm_ntypes)
            for ntype, feats in lm_node_feats.items():
                assert ntype not in self._lm_node_feats, \
                        f"More than one BERT model runs on Node {ntype}."
                self._lm_node_feats[ntype] = feats
        self.out = nn.Linear(self.config.hidden_size, out_dim)
        self._loss_fn = gsmodel.ClassifyLossFunc(multilabel=False)
        self.gnn_encoder = RelationalGCNEncoder(g,
                                           h_dim, self.config.hidden_size,
                                           num_bases=-1,
                                           num_hidden_layers=num_layers -1)

    def forward(self, blocks, node_feats, edge_feats, labels, input_nodes):
        output_nodes = blocks[-1].dstdata[dgl.NID]
        input_ids = self._lm_node_feats[self.target_ntype][TOKEN_IDX][output_nodes].to(self.llm.device)
        attention_mask = self._lm_node_feats[self.target_ntype][ATT_MASK_IDX][output_nodes].to(self.llm.device)
        graph_tokens = self.gnn_encoder(blocks, node_feats)[self.target_ntype]

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        word_embeddings = self.llm.get_input_embeddings()
        # Tentatively leave the zero prompting line for debugging and testing
        #inputs_embeds = torch.cat([torch.zeros_like(graph_tokens.unsqueeze(1)), word_embeddings(input_ids)], dim=1)
        inputs_embeds = torch.cat([graph_tokens.unsqueeze(1), word_embeddings(input_ids)], dim=1)
        attention_mask = torch.cat([torch.ones((input_shape[0],1), device=self.llm.device), attention_mask], dim=1)
        model_output = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        # We use the last token in order to do the classification, as other causal models
        masked_hidden_states = model_output.last_hidden_state * attention_mask.unsqueeze(-1)
        last_token_indexes = (attention_mask.sum(dim=1, dtype=torch.int64) - 1)
        last_token_embeddings = masked_hidden_states[torch.arange(last_token_indexes.size(0)),last_token_indexes,:]
        h = self.out(last_token_embeddings)
        
        loss = self._loss_fn(h, labels[self.target_ntype])
        
        return loss

        
    def predict(self, blocks, node_feats, _, input_nodes, return_proba):
        output_nodes = blocks[-1].dstdata[dgl.NID]
        
        input_ids = self._lm_node_feats[self.target_ntype][TOKEN_IDX][output_nodes].to(self.llm.device)
        attention_mask = self._lm_node_feats[self.target_ntype][ATT_MASK_IDX][output_nodes].to(self.llm.device)
        graph_tokens = self.gnn_encoder(blocks, node_feats)[self.target_ntype]

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        word_embeddings = self.llm.get_input_embeddings()
        # Tentatively leave the zero prompting line for debugging and testing
        #inputs_embeds = torch.cat([torch.zeros_like(graph_tokens.unsqueeze(1)), word_embeddings(input_ids)], dim=1)
        inputs_embeds = torch.cat([graph_tokens.unsqueeze(1), word_embeddings(input_ids)], dim=1)
        attention_mask = torch.cat([torch.ones((input_shape[0],1), device=self.llm.device), attention_mask], dim=1)
        model_output = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        masked_hidden_states = model_output.last_hidden_state * attention_mask.unsqueeze(-1)
        last_token_indexes = (attention_mask.sum(dim=1, dtype=torch.int64) - 1)
        last_token_embeddings = masked_hidden_states[torch.arange(last_token_indexes.size(0)),last_token_indexes,:]
        logits = self.out(last_token_embeddings)

        if return_proba:
            return logits.argmax(dim=-1), torch.softmax(logits, 1)
        else:
            return logits.argmax(dim=-1), logits
        
    def create_optimizer(self):
        # Here we assume there are no sparse embeddings.
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.debug(f"Num of trainable params: {pytorch_total_params}")

        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def restore_model(self, restore_model_path):
        self.llm = AutoModel.from_pretrained(restore_model_path, config=self.config)

    def save_model(self, model_path):
        os.makedirs(model_path, exist_ok=True)
        os.chmod(model_path, 0o767)
        if gs.get_rank() == 0:
            self.llm.save_pretrained(model_path) 
