from graphstorm import model as gsmodel
import torch
import torch.nn as nn
import re
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)
from peft import LoraConfig, get_peft_model

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
        model_id = "facebook/opt-2.7b"
        config = AutoConfig.from_pretrained(model_id, num_labels=out_dim)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_id,config=config
        )
        peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
        self.llm = get_peft_model(base_model, peft_config)
        lm_feat_names = ['input_ids', 'attention_mask']
        self._lm_node_feats = {}
        for lm_config in node_lm_configs:
            # A list of node types sharing the same lm model
            lm_ntypes = lm_config["node_types"]
            lm_node_feats = get_lm_node_feats(g, lm_feat_names, lm_ntypes)
            for ntype, feats in lm_node_feats.items():
                assert ntype not in self._lm_node_feats, \
                        f"More than one BERT model runs on Node {ntype}."
                self._lm_node_feats[ntype] = feats
        for _ in range(num_layers):
            pass

    def forward(self, blocks, node_feats, edge_feats, labels, input_nodes):
        # input layer
        h = {}
        output_nodes = blocks[-1].dstdata["_ID"]

        input_ids = self._lm_node_feats[self.target_ntype]["input_ids"][output_nodes].to(self.llm.device)
        attention_mask = self._lm_node_feats[self.target_ntype]["attention_mask"][output_nodes].to(self.llm.device)
        model_output = self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels[self.target_ntype])
        return model_output.loss

        
    def predict(self, blocks, node_feats, _, input_nodes, return_proba):
        # input layer
        h = {}
        output_nodes = blocks[-1].dstdata["_ID"]
        input_ids = self._lm_node_feats[self.target_ntype]["input_ids"][output_nodes].to(self.llm.device)
        attention_mask = self._lm_node_feats[self.target_ntype]["attention_mask"][output_nodes].to(self.llm.device)
        model_output = self.llm(input_ids=input_ids, attention_mask=attention_mask)

        if return_proba:
            return model_output.logits.argmax(dim=-1), torch.softmax(model_output.logits, 1)
        else:
            return model_output.logits.argmax(dim=-1), model_output.logits
        
    def create_optimizer(self):
        # Here we don't set up an optimizer for sparse embeddings.
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Num of trainable params: {pytorch_total_params}")

        return torch.optim.Adam(self.parameters(), lr=self.lr)