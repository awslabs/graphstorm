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
from graphstorm.model.edge_decoder import LinkPredictDotDecoder
from graphstorm.model.embed import GSNodeInputLayer
from dgl.nn import GraphConv, HeteroGraphConv

from peft import LoraConfig, get_peft_model, AutoPeftModel
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

class DummyNodeInputLayer(nn.Module):
    def __init__(self):
        super(DummyNodeInputLayer, self).__init__()

    def require_cache_embed(self):
        return False
    
    def forward(self, input_feats, input_nodes):
        return input_feats

class MarginContrastiveLoss(nn.Module):
    def __init__(self):
        super(MarginContrastiveLoss, self).__init__()
        self.margin = 0.5

    def forward(self, pos_score, neg_score):
        pos_distance = []
        neg_distance = []
        for k in pos_score:
            pos_distance.append(1 - pos_score[k])
            neg_distance.append(1 - neg_score[k])
        pos_distance = torch.cat(pos_distance, dim=0)
        neg_distance = torch.cat(neg_distance, dim=0)
        loss = F.relu(self.margin - neg_distance).pow(2) + pos_distance.pow(2)
        return loss.mean()

class GNNLLM_NC(gsmodel.GSgnnNodeModelBase):
    """ GNNLLM_NC Model Class

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
        super(GNNLLM_NC, self).__init__()
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
        self.gnn = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn.append(HeteroGraphConv({
                _etype: GraphConv(h_dim, h_dim) for _etype in g.etypes
            }))
        self.projection = nn.Linear(h_dim, self.config.hidden_size)

    def encode_graph(self, blocks, h):
        for layer, block in zip(self.gnn, blocks):
            h = layer(block, h)
            h = {k: F.relu(v) for k, v in h.items()}
        src_type, dst_type = blocks[0].ntypes
        graph_tokens = self.projection(h[dst_type])
        return graph_tokens

    def forward(self, blocks, node_feats, edge_feats, labels, input_nodes):
        output_nodes = blocks[-1].dstdata[dgl.NID]
        input_ids = self._lm_node_feats[self.target_ntype][TOKEN_IDX][output_nodes].to(self.llm.device)
        attention_mask = self._lm_node_feats[self.target_ntype][ATT_MASK_IDX][output_nodes].to(self.llm.device)
        graph_tokens = self.encode_graph(blocks, node_feats)
        input_shape = input_ids.size()
        # make sure input_ids are batch_size X seq_len
        input_ids = input_ids.view(-1, input_shape[-1])
        word_embeddings = self.llm.get_input_embeddings()
        # assuming graph_tokens has a shape of [batch_size, embedding_dim],
        # input_ids has a shape of [batch_size, seq_len, embedding_dim]
        # only one graph token is inserted ahead of input words 
        inputs_embeds = torch.cat([graph_tokens.unsqueeze(1), word_embeddings(input_ids)], dim=1)
        
        # enable attention computation on the inserted graph token
        attention_mask = torch.cat([torch.ones((input_shape[0],1), device=self.llm.device), attention_mask], dim=1)
        model_output = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        # We use the last token in order to do the classification, as other causal models
        masked_hidden_states = model_output.last_hidden_state * attention_mask.unsqueeze(-1)
        last_token_indexes = (attention_mask.sum(dim=1, dtype=torch.int64) - 1)
        last_token_embeddings = masked_hidden_states[torch.arange(last_token_indexes.size(0)),last_token_indexes,:]
        h = self.out(last_token_embeddings)
        
        loss = self._loss_fn(h, labels[self.target_ntype])
        # L2 regularization of dense parameters
        reg_loss = torch.tensor(0.).to(loss.device)
        for name, d_para in self.named_parameters():
            if 'gnn' in name:
                reg_loss += d_para.square().sum()
        reg_loss = self.alpha_l2norm * reg_loss
        return loss + reg_loss
        
    def predict(self, blocks, node_feats, _, input_nodes, return_proba):
        output_nodes = blocks[-1].dstdata[dgl.NID]
        
        input_ids = self._lm_node_feats[self.target_ntype][TOKEN_IDX][output_nodes].to(self.llm.device)
        attention_mask = self._lm_node_feats[self.target_ntype][ATT_MASK_IDX][output_nodes].to(self.llm.device)
        graph_tokens = self.encode_graph(blocks, node_feats)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        word_embeddings = self.llm.get_input_embeddings()
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
        # TODO: save self.out
        self.llm = AutoPeftModel.from_pretrained(restore_model_path)

    def save_model(self, model_path):
        os.makedirs(model_path, exist_ok=True)
        os.chmod(model_path, 0o767)
        if gs.get_rank() == 0:
            self.llm.save_pretrained(model_path) 

# It acts like a gnn_encoder in GSF built-in model
class GPEFT(nn.Module):
    def __init__(self,
                 g,
                 h_dim,            # hiddem dimension of LLM
                 num_layers,           # number of GNN layers
                 target_ntype,          # target node type, assuming same type for src and dst nodes
                 node_lm_configs,
                 ):
        super(GPEFT, self).__init__()
        self.num_layers = num_layers
        self.target_ntype = target_ntype
        # assume only one LLM is used
        model_id = node_lm_configs[0]["model_name"]
        self.config = AutoConfig.from_pretrained(model_id)
        self.out_dims = self.config.hidden_size
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
        self.gnn = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn.append(HeteroGraphConv({
                _etype: GraphConv(h_dim, h_dim) for _etype in g.etypes
            }))
        self.projection = nn.Linear(h_dim, self.out_dims)

    @property
    def device(self):
        return self.llm.device   

    def encode_graph(self, blocks, h):
        if blocks is None or len(blocks) == 0:
            return None
        else:
            for layer, block in zip(self.gnn, blocks):
                h = layer(block, h)
                h = {k: F.relu(v) for k, v in h.items()}
            src_type, dst_type = blocks[0].ntypes
            graph_tokens = self.projection(h[dst_type])
            return graph_tokens

    def forward(self, blocks, h):
        output_nodes = blocks[-1].dstdata[dgl.NID]
        input_ids = self._lm_node_feats[self.target_ntype][TOKEN_IDX][output_nodes].to(self.llm.device)
        attention_mask = self._lm_node_feats[self.target_ntype][ATT_MASK_IDX][output_nodes].to(self.llm.device)
        prompt = self.encode_graph(blocks, h)
        #if prompt is None:
        if True:
            model_output = self.llm(input_ids, attention_mask)
            masked_hidden_states = model_output.last_hidden_state * attention_mask.unsqueeze(-1)
            last_token_indexes = (attention_mask.sum(dim=1, dtype=torch.int64) - 1)
            last_token_embeddings = masked_hidden_states[torch.arange(last_token_indexes.size(0)),last_token_indexes,:]
        else:
            word_embeddings = self.llm.get_input_embeddings()
            # assuming graph_tokens has a shape of [batch_size, embedding_dim],
            # input_ids has a shape of [batch_size, seq_len, embedding_dim]
            # only one graph token is inserted ahead of input words 
            inputs_embeds = torch.cat([prompt.unsqueeze(1), word_embeddings(input_ids)], dim=1)
            attention_mask = torch.cat([torch.ones((input_ids.shape[0],1), device=self.llm.device), attention_mask], dim=1)
            model_output = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            
            masked_hidden_states = model_output.last_hidden_state * attention_mask.unsqueeze(-1)
            last_token_indexes = (attention_mask.sum(dim=1, dtype=torch.int64) - 1)
            last_token_embeddings = masked_hidden_states[torch.arange(last_token_indexes.size(0)),last_token_indexes,:]
        return {self.target_ntype: F.normalize(last_token_embeddings, p=2)}

class GNNLLM_LP(gsmodel.GSgnnLinkPredictionModelBase):
    """ GNNLLM_LP Model Class

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
    def __init__(self, g, node_lm_configs, h_dim, num_layers,
                 target_ntype,     # the node type to be predict
                 target_etype,     # the node type to be predict
                 use_norm = True,   # use normalization or not, default is True
                 alpha_l2norm = 0,
                 lr = 0.001):
        super(GNNLLM_LP, self).__init__()
        
        self.target_etype=target_etype
        self.alpha_l2norm = alpha_l2norm
        self.lr = lr
        self.gnn_encoder = GPEFT(g, h_dim, num_layers, target_ntype, node_lm_configs)
        # dummy node input encoder to enable LP Infer
        self.node_input_encoder = DummyNodeInputLayer()
        self.loss_fn = MarginContrastiveLoss()
        self.decoder = LinkPredictDotDecoder(self.gnn_encoder.out_dims)

    # Required by lp_infer 
    def inplace_normalize_node_embs(self, embs):
        return embs
    
    def forward(self, blocks, pos_graph, neg_graph, node_feats, edge_feats, pos_edge_feats, input_nodes):
        encode_embs = self.gnn_encoder(blocks, node_feats)

        pos_score = self.decoder(pos_graph, encode_embs)
        neg_score = self.decoder(neg_graph, encode_embs)

        assert pos_score.keys() == neg_score.keys(), \
            "Positive scores and Negative scores must have edges of same" \
            f"edge types, but get {pos_score.keys()} and {neg_score.keys()}"
        loss = self.loss_fn(pos_score, neg_score)
        # L2 regularization of dense parameters
        reg_loss = torch.tensor(0.).to(loss.device)
        for name, d_para in self.named_parameters():
            if 'gnn' in name:
                reg_loss += d_para.square().sum()
        reg_loss = self.alpha_l2norm * reg_loss
        return loss + reg_loss
        
    def create_optimizer(self):
        # Here we assume there are no sparse embeddings.
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.debug(f"Num of trainable params: {pytorch_total_params}")

        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def restore_model(self, restore_model_path):
        self.gnn_encoder.llm = AutoPeftModel.from_pretrained(restore_model_path)

    def save_model(self, model_path):
        os.makedirs(model_path, exist_ok=True)
        os.chmod(model_path, 0o767)
        if gs.get_rank() == 0:
            self.gnn_encoder.llm.save_pretrained(model_path) 