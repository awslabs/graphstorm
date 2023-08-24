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
import torch as th

from graphstorm.model.lm_model import TOKEN_IDX, ATT_MASK_IDX, TOKEN_TID_IDX
# pylint: disable=missing-function-docstring

class Dummy:
    """dummy object used to create config objects
    constructor"""
    def __init__(self, arg_dict):
        self.__dict__.update(arg_dict)

def create_tokens(tokenizer, input_text, max_seq_length, num_node, return_token_type_ids=False):
    input_text = input_text * num_node
    tokens = tokenizer(input_text,  max_length=max_seq_length,
                        truncation=True, padding='max_length', return_tensors='pt')
    # we only use TOKEN_IDX and VALID_LEN_IDX
    input_ids = tokens[TOKEN_IDX]
    attention_mask = tokens[ATT_MASK_IDX]
    valid_len = tokens[ATT_MASK_IDX].sum(dim=1)
    token_type_ids = tokens.get(TOKEN_TID_IDX, th.zeros_like(input_ids))\
        if return_token_type_ids else None
    return input_ids, valid_len, attention_mask, token_type_ids
