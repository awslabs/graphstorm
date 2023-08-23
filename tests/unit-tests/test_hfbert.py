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
"""
from transformers import AutoTokenizer
from numpy.testing import assert_almost_equal
import pytest

from util import create_tokens
from graphstorm.model.lm_model import init_lm_model
from graphstorm.model.lm_model import BUILTIN_HF_BERT
from graphstorm.model.lm_model import TOKEN_IDX, ATT_MASK_IDX, TOKEN_TID_IDX, VALID_LEN
# pylint: disable=redefined-outer-name line-too-long missing-function-docstring

def comput_bert(lm_model, input_ids, attention_masks, token_type_ids=None):
    lm_model.lm_model.eval()
    outputs = lm_model.lm_model(input_ids,
                                attention_mask=attention_masks,
                                token_type_ids=token_type_ids)
    lm_model.lm_model.train()
    out_emb = outputs.pooler_output
    return out_emb.detach().cpu()

@pytest.mark.parametrize("num_train", [0, 10, -1])
@pytest.mark.parametrize("input_ntypes", [["n1", "n2", "n3"], ["n1"]])
@pytest.mark.parametrize("generate_tid", [True, False])
@pytest.mark.parametrize("lm_type, bert_model_name",[
    (BUILTIN_HF_BERT, "bert-base-uncased"),
    ("roberta", "roberta-base"),
    ("albert", "albert-base-v1"),
    ("camembert", "camembert-base"),
    ("ernie", "nghuyong/ernie-1.0-base-zh"),
    ("ibert", "kssteven/ibert-roberta-base"),
    ("mega", "mnaylor/mega-base-wikitext"),
    ("mpnet", "microsoft/mpnet-base"),
    ("nezha", "sijunhe/nezha-cn-base"),
    ("qdqbert", "bert-base-uncased"),
    ("roc_bert", "weiweishi/roc-bert-base-zh")])
def test_hfbert_wrapper(num_train, input_ntypes, generate_tid, lm_type, bert_model_name):
    device='cuda:0'
    max_seq_length = 32
    num_nodes = [100, 50, 1]
    lm_model = init_lm_model({"lm_type": lm_type,
                              "model_name": bert_model_name,
                              "gradient_checkpoint": True}, num_train=num_train)
    lm_model = lm_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    input_texts = [["A Graph neural network (GNN) is a class of artificial neural networks for processing data that can be represented as graphs."], ["Amazon Web Services, Inc. (AWS) is a subsidiary of Amazon that provides on-demand cloud computing platforms and APIs to individuals, companies, and governments, on a metered, pay-as-you-go basis."], ["Hello world!"]]
    inputs = {}
    for i, ntype in enumerate(input_ntypes):
        # test valid_len
        input_ids, valid_len, attention_mask, token_type_ids = \
            create_tokens(tokenizer=tokenizer,
                          input_text=input_texts[i],
                          max_seq_length=max_seq_length,
                          num_node=num_nodes[i],
                          return_token_type_ids=generate_tid)
        inputs[ntype] = (input_ids, valid_len, attention_mask, token_type_ids)

    input_lm_feats = {}
    for ntype in input_ntypes:
        input_lm_feats[ntype] = {
            TOKEN_IDX: inputs[ntype][0],
            VALID_LEN: inputs[ntype][1],
        }
        if generate_tid:
            input_lm_feats[ntype][TOKEN_TID_IDX] = inputs[ntype][3].detach()
    lm_model.eval()
    wrapper_emb = lm_model(input_ntypes, input_lm_feats)

    input_lm_feats2 = {}
    for ntype in input_ntypes:
        input_lm_feats2[ntype] = {
            TOKEN_IDX: inputs[ntype][0],
            ATT_MASK_IDX: inputs[ntype][2],
        }
        if generate_tid:
            input_lm_feats2[ntype][TOKEN_TID_IDX] = inputs[ntype][3].detach()
    lm_model.eval()
    wrapper_emb2 = lm_model(input_ntypes, input_lm_feats2)

    input_lm_feats3 = {}
    for ntype in input_ntypes:
        input_lm_feats3[ntype] = {
            TOKEN_IDX: inputs[ntype][0],
            # hfbert also allows ATT_MASK_IDX store the valid mask lenght
            ATT_MASK_IDX: inputs[ntype][2].sum(dim=1),
        }
        if generate_tid:
            input_lm_feats3[ntype][TOKEN_TID_IDX] = inputs[ntype][3].detach()
    lm_model.eval()
    wrapper_emb3 = lm_model(input_ntypes, input_lm_feats3)

    for ntype in input_ntypes:
        emb = comput_bert(lm_model,
                          inputs[ntype][0].to(device),
                          inputs[ntype][2].to(device),
                          inputs[ntype][3].to(device) \
                            if inputs[ntype][3] is not None else None)
        assert_almost_equal(wrapper_emb[ntype].detach().cpu().numpy(), emb.numpy(), decimal=5)
        assert_almost_equal(wrapper_emb2[ntype].detach().cpu().numpy(), emb.numpy(), decimal=5)
        assert_almost_equal(wrapper_emb3[ntype].detach().cpu().numpy(), emb.numpy(), decimal=5)

def test_hfbert_wrapper_profile():
    device='cuda:0'
    bert_model_name = "bert-base-uncased"
    max_seq_length = 32
    lm_model = init_lm_model({"lm_type": BUILTIN_HF_BERT,
                              "model_name": bert_model_name,
                              "gradient_checkpoint": True},
                              num_train=10,
                              profile=True)
    lm_model = lm_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    input_text = ["A Graph neural network (GNN) is a class of artificial neural networks for processing data that can be represented as graphs."]
    input_text_n1 = input_text * 100
    tokens_n1 = tokenizer(input_text_n1,  max_length=max_seq_length,
                           truncation=True, padding=True, return_tensors='pt')
    # we only use TOKEN_IDX and ATT_MASK_IDX
    input_ids_n1 = tokens_n1[TOKEN_IDX].share_memory_()
    att_mask_n1 = tokens_n1[ATT_MASK_IDX].share_memory_()

    input_lm_feats = {}
    input_lm_feats["n1"] = {
        TOKEN_IDX: input_ids_n1,
        ATT_MASK_IDX: att_mask_n1
    }
    lm_model.eval()
    lm_model(["n1"], input_lm_feats)
    lm_model.train()
    prof_train_len = lm_model.get_avg_train_seq_len()
    prof_train_flops = lm_model.get_avg_train_flops()
    prof_static_len = lm_model.get_avg_static_seq_len()
    prof_static_flops = lm_model.get_avg_static_flops()
    assert prof_train_len == -1
    assert prof_train_flops == -1
    assert prof_static_len > 0
    assert prof_static_flops > 0

    lm_model(["n1"], input_lm_feats)
    prof_train_len = lm_model.get_avg_train_seq_len()
    prof_train_flops = lm_model.get_avg_train_flops()
    prof_static_len = lm_model.get_avg_static_seq_len()
    prof_static_flops = lm_model.get_avg_static_flops()
    assert prof_train_len > 0
    assert prof_train_flops > 0
    assert prof_static_len > 0
    assert prof_static_flops > 0

if __name__ == '__main__':
    lm_type, bert_model_name = BUILTIN_HF_BERT, "bert-base-uncased"
    test_hfbert_wrapper(0, ["n1", "n2", "n3"], False, lm_type, bert_model_name)
    test_hfbert_wrapper(10, ["n1", "n2", "n3"], False, lm_type, bert_model_name)
    test_hfbert_wrapper(-1, ["n1", "n2", "n3"], False, lm_type, bert_model_name)

    test_hfbert_wrapper(0, ["n1", "n2", "n3"], True, lm_type, bert_model_name)
    test_hfbert_wrapper(10, ["n1", "n2", "n3"], True, lm_type, bert_model_name)
    test_hfbert_wrapper(-1, ["n1", "n2", "n3"], True, lm_type, bert_model_name)
    test_hfbert_wrapper(10, ["n1"], False, lm_type, bert_model_name)
    test_hfbert_wrapper_profile()
