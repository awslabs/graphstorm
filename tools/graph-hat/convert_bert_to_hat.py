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

    Convert Huggingface Bert Model into Hierarchical Transformer Model
"""
import os
import argparse

import torch
import copy
from transformers import AutoModelForMaskedLM, AutoTokenizer
from graphstorm.model.graph_transformer import (HAT_LAYOUTS,
                                                HATForMaskedLM,
                                                HATConfig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--layer-init-strategy', default='grouped', type=str,
                        choices=['linear', 'grouped', 'random', 'embeds-only', 'none'],
                        help='linear: S|D encoders are warm-started independently (one-by-one)'
                             'grouped: pairs of S|D are warm-started with weights from the very same level'
                             'random: D encoders are not warm-started'
                             'embeds-only: No warm-starting, except embeddings'
                             'none: No warm-starting')


    parser.add_argument('--layout', default='s1', type=str,
                        choices=HAT_LAYOUTS.keys(),
                        help='Sentences and Document (S|D) encoders layout.'
                             f'Choose a predefined layout from {HAT_LAYOUTS}'
                             'or define your own.')
    parser.add_argument('--bert-checkpoint', type=str, default=None,
                        help='Huggingface bert checkpoint used for initializing HAT. '
                        'If not provided, a google/bert_uncased_L-<Layers>_H-256_A-4 '
                        'checkpoint will be used.')
    parser.add_argument('--max-sentences', type=int, default=64)
    parser.add_argument('--max-sentence-length', type=int, default=128)
    parser.add_argument('--output', type=str,
                        help='Output dir to store HAT model checkpoint.')
    parser.add_argument('--hat-base-config-path', type=str,
                        help='Path to a base HAT config.')
    args = parser.parse_args()

    max_sentences = args.max_sentences
    max_sentence_length = args.max_sentence_length
    encoder_layout = {}
    hat_layout = HAT_LAYOUTS[args.layout].split('|') \
        if args.layout in HAT_LAYOUTS else args.layout.split('|')
    for idx, block_pattern in enumerate(hat_layout):
        encoder_layout[str(idx)] = {
            "sentence_encoder": True if 'S' in block_pattern else False,
            "document_encoder": True if 'D' in block_pattern else False}
    num_hidden_layers = len(encoder_layout.keys())

    bert_checkpoint = args.bert_checkpoint
    num_bert_layers = num_hidden_layers \
            if args.layer_init_strategy != 'linear' else num_hidden_layers * 2

    if bert_checkpoint is None:
        # bert_uncased_L-<X>_H-256_A-4 only supports X=2,4,6,8,10,12
        num_bert_layers = num_bert_layers + 1 if num_bert_layers % 2 else num_bert_layers
        bert_checkpoint = f'google/bert_uncased_L-{str(num_bert_layers)}_H-256_A-4'

    # load pre-trained bert model and tokenizer
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)

    # TODO: check number of layers of bert_model >= num_hidden_layers

    bert_config = bert_model.config
    print(bert_model.config)
    hat_config = HATConfig.from_pretrained(args.hat_base_config_path)
    # Text length parameters
    hat_config.max_sentence_length = max_sentence_length
    hat_config.max_sentences = max_sentences
    hat_config.max_position_embeddings = max_sentence_length
    hat_config.model_max_length = int(max_sentence_length * max_sentences)
    hat_config.num_hidden_layers = num_hidden_layers
    # Transformer parameters
    hat_config.hidden_size = bert_config.hidden_size
    hat_config.intermediate_size = bert_config.intermediate_size
    hat_config.num_attention_heads = bert_config.num_attention_heads
    hat_config.hidden_act = bert_config.hidden_act
    hat_config.encoder_layout = encoder_layout
    # Vocabulary parameters
    hat_config.vocab_size = bert_config.vocab_size
    hat_config.pad_token_id = bert_config.pad_token_id
    hat_config.bos_token_id = bert_config.bos_token_id
    hat_config.eos_token_id = bert_config.eos_token_id
    hat_config.type_vocab_size = bert_config.type_vocab_size

    # load dummy hi-transformer model
    hat_model = HATForMaskedLM.from_config(hat_config)

    if args.layer_init_strategy != 'none':
        # copy position embeddings
        hat_model.hi_transformer.embeddings.position_embeddings.weight.data[0] = \
            torch.zeros((bert_config.hidden_size,))
        hat_model.hi_transformer.embeddings.position_embeddings.weight.data[1:] = \
            bert_model.bert.embeddings.position_embeddings.weight[1:max_sentence_length + \
                                                                  hat_config.pad_token_id+1]
        # word_embeddings
        hat_model.hi_transformer.embeddings.word_embeddings.load_state_dict( \
            bert_model.bert.embeddings.word_embeddings.state_dict())
        hat_model.hi_transformer.embeddings.token_type_embeddings.load_state_dict( \
            bert_model.bert.embeddings.token_type_embeddings.state_dict())
        hat_model.hi_transformer.embeddings.LayerNorm.load_state_dict(
            bert_model.bert.embeddings.LayerNorm.state_dict())

        if args.layer_init_strategy != 'embeds-only':
            # copy transformer layers
            if args.layer_init_strategy != 'linear':
                for idx in range(num_hidden_layers):
                    if hat_model.config.encoder_layout[str(idx)]['sentence_encoder']:
                        hat_model.hi_transformer.encoder.layer[idx]\
                            .sentence_encoder.load_state_dict( \
                                bert_model.bert.encoder.layer[idx].state_dict())
                    if hat_model.config.encoder_layout[str(idx)]['document_encoder']:
                        if args.layer_init_strategy == 'grouped':
                            hat_model.hi_transformer.encoder.layer[idx]\
                                .document_encoder.load_state_dict(
                                    bert_model.bert.encoder.layer[idx].state_dict())
                        hat_model.hi_transformer.encoder.layer[idx]\
                            .position_embeddings.weight.data = \
                                bert_model.bert.embeddings.position_embeddings\
                                    .weight[1:max_sentences+2]
            else: # linear init
                for idx, layer_idx in enumerate(range(0, num_hidden_layers*2, 2)):
                    assert hat_model.config.encoder_layout[str(idx)]['sentence_encoder'] and \
                        hat_model.config.encoder_layout[str(idx)]['document_encoder'], \
                        "The HAT layout must be SD|SD|SD..|SD"
                    hat_model.hi_transformer.encoder.layer[idx]\
                        .sentence_encoder.load_state_dict(
                            bert_model.bert.encoder.layer[layer_idx].state_dict())
                    hat_model.hi_transformer.encoder.layer[idx]\
                        .document_encoder.load_state_dict(
                            bert_model.bert.encoder.layer[layer_idx+1].state_dict())
                    hat_model.hi_transformer.encoder.layer[idx]\
                        .position_embeddings.weight.data = \
                            bert_model.bert.embeddings.position_embeddings\
                                .weight[1:max_sentences+2]

        # copy lm_head
        hat_model.lm_head.dense.load_state_dict(\
            bert_model.cls.predictions.transform.dense.state_dict())
        hat_model.lm_head.layer_norm.load_state_dict(\
            bert_model.cls.predictions.transform.LayerNorm.state_dict())
        hat_model.lm_head.decoder.load_state_dict(\
            bert_model.cls.predictions.decoder.state_dict())
        hat_model.lm_head.bias = copy.deepcopy(bert_model.cls.predictions.bias)

    # save model
    hat_model.save_pretrained(os.path.join(args.output,
        f"hat-{args.layout}-{args.layer_init_strategy}-{bert_checkpoint}"))

    # save tokenizer
    tokenizer.save_pretrained(os.path.join(args.output,
        f"hat-{args.layout}-{args.layer_init_strategy}-{bert_checkpoint}"))
