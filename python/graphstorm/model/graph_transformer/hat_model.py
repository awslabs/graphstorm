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

    Graph Hierarchical Transformer
"""

import torch
import torch.utils.checkpoint
from packaging import version
from dataclasses import dataclass
from typing import Optional, Tuple
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, CosineEmbeddingLoss
from torch.nn.functional import normalize

from transformers import RobertaTokenizer, BertTokenizer, PretrainedConfig

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import (
    ModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaAttention, RobertaIntermediate, RobertaOutput
from transformers.activations import gelu
from transformers import PretrainedConfig


from .hat_config import HATConfig

try:
    import nltk
    from nltk import sent_tokenize
    nltk.download('punkt')
except:
    raise Exception('NLTK is not installed! Install it with `pip install nltk`...')


HAT_LAYOUTS = {
    's1': 'SD|SD|SD|SD|SD|SD',
    's2': 'S|SD|D|S|SD|D|S|SD|D',
    'p1': 'S|SD|S|SD|S|SD|S|SD',
    'p2': 'S|S|SD|S|S|SD|S|S|SD',
    'e1': 'SD|SD|SD|S|S|S|S|S|S',
    'e2': 'S|SD|D|S|SD|D|S|S|S|S',
    'l1': 'S|S|S|S|S|S|SD|SD|SD',
    'l2': 'S|S|S|S|S|SD|D|S|SD|D',
    'b1': 'S|S|SD|D|S|SD|D|S|S|S',
    'b2': 'S|S|SD|SD|SD|S|S|S|S',
    'f12': 'S|S|S|S|S|S|S|S|S|S|S|S',
    'f8': 'S|S|S|S|S|S|S|S',
    'f6': 'S|S|S|S|S|S',
}

@dataclass
class BaseModelOutputWithSentenceAttentions(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape
        `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*,
        returned when `output_hidden_states=True` is passed or
        when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of
            the embeddings + one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each
            layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*,
        returned when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of
            shape `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used
            to compute the weighted average in the self-attention
            heads.
        sentence_attentions (`tuple(torch.FloatTensor)`,
        *optional*, returned when `output_attentions=True` is
        passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of
            shape `(batch_size, num_heads, sequence_length, sequence_length)`.

            Sentence attentions weights after the attention
            softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    sentence_attentions: Optional[Tuple[torch.FloatTensor]] = None


def transform_tokens2sentences(hidden_states, num_sentences, max_sentence_length):
    """ Split a long sequence of tokens, i.e., hidden_states into num_sentences.

        s0_t0 s0_t1 ... s0_tn pad pad | s1_t0 s1_t1 ... pad | ... | smax_t0, ... pad |
            -->
        s0_t0 s0_t1 ... s0_tn pad pad
        s1_t0 s1_t1 ... s1_tm pad pad
        ...
        smax_t0, ...  pad pad pad pad

    Parameters
    ----------
    hidden_states:
        Tokens
    num_sentences: int
        Number of sentences in hidden_states
    max_sentence_length: int
        Maximum sentences length

    Return
        Reshared token tensor
    """
    # transform sequence into segments
    seg_hidden_states = torch.reshape(hidden_states, (hidden_states.size(0), num_sentences, max_sentence_length, hidden_states.size(-1)))
    # squash segments into sequence into a single axis (samples * segments, max_segment_length, hidden_size)
    hidden_states_reshape = seg_hidden_states.contiguous().view(hidden_states.size(0) * num_sentences,
                                                                max_sentence_length, seg_hidden_states.size(-1))

    return hidden_states_reshape


def transform_masks2sentences(hidden_states, num_sentences, max_sentence_length):
    """ Split a sequence of masks, i.e., hidden_states into num_sentences.
        Shall be invoked with transform_tokens2sentences

    Parameters
    ----------
    hidden_states:
        Masks
    num_sentences: int
        Number of sentences in hidden_states
    max_sentence_length: int
        Maximum sentences length

    Return
        Reshared mask tensor
    """
    # transform sequence into segments
    seg_hidden_states = torch.reshape(hidden_states, (hidden_states.size(0), 1, 1, num_sentences, max_sentence_length))
    # squash segments into sequence into a single axis (samples * segments, 1, 1, max_segment_length)
    hidden_states_reshape = seg_hidden_states.contiguous().view(hidden_states.size(0) * num_sentences,
                                                                1, 1, seg_hidden_states.size(-1))

    return hidden_states_reshape


def transform_sentences2tokens(seg_hidden_states, num_sentences, max_sentence_length):
    """ Transform sentences to tokens

    Parameters
    ----------
    hidden_states:
        Sentences
    num_sentences: int
        Number of sentences in hidden_states
    max_sentence_length: int
        Maximum sentences length

    Return
        Reshared token tensor
    """
    # transform squashed sequence into segments
    hidden_states = seg_hidden_states.contiguous().view(seg_hidden_states.size(0) // num_sentences, num_sentences,
                                                        max_sentence_length, seg_hidden_states.size(-1))
    # transform segments into sequence
    hidden_states = hidden_states.contiguous().view(hidden_states.size(0), num_sentences * max_sentence_length,
                                                    hidden_states.size(-1))
    return hidden_states

def create_position_ids_from_input_ids(input_ids, padding_idx, position_ids):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin
    at padding_idx+1. Padding symbols are ignored. This is modified from
    fairseq's `utils.make_positions`.

    Parameters
    ----------
    input_ids: torch.Tensor
        Input ids
    padding_idx: int
        Pad token id
    position_ids: position id

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    return position_ids[:, :input_ids.size(1)].repeat(input_ids.size(0), 1) * mask

class TransformerLayer(nn.Module):
    """ TransformerLayer

        Parameters
        ----------
        config: HATConfig
            Config
    """
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False):
        """ Forward function
        """
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        return outputs


class HATEmbeddings(nn.Module):
    """
    Similar as BertEmbeddings with a tiny tweak for positional embeddings indexing.

    Parameters
    ----------
        config: HATConfig
    """
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        # The entries at padding_idx do not contribute to the gradient
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size,
                                            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_sentence_length + self.padding_idx + 1,
                                                config.hidden_size,
                                                padding_idx=self.padding_idx)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(
            self.padding_idx + 1,
            config.max_sentence_length + self.padding_idx + 1).repeat(config.max_sentences).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        """ forward function
        """
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids,
                                                                  self.padding_idx,
                                                                  self.position_ids)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where
        # it is all zeros, which usually occurs when its auto-generated, registered
        # buffer helps users when tracing the model without passing token_type_ids.
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Parameters
        ----------
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

class HATLayer(nn.Module):
    """ Graph Hierarchical Transformer layer


        Parameters
        ----------
        config: HATConfig
            Model config
        use_sentence_encoder: bool
            Whether this layer has sentence encoder
        use_document_encoder: bool
            Whether this layer has document encoder
    """
    def __init__(self, config, use_sentence_encoder=True, use_document_encoder=True):
        super().__init__()
        self.max_sentence_length = config.max_sentence_length
        self.max_sentences = config.max_sentences
        self.hidden_size = config.hidden_size
        self.use_document_encoder = use_document_encoder
        self.use_sentence_encoder = use_sentence_encoder
        if self.use_sentence_encoder:
            self.sentence_encoder = TransformerLayer(config)
        if self.use_document_encoder:
            self.document_encoder = TransformerLayer(config)
            self.position_embeddings = nn.Embedding(config.max_sentences+1, config.hidden_size,
                                                    padding_idx=config.pad_token_id)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        num_sentences=None,
        output_attentions=False):
        """ Forward function
        """
        sentence_outputs = (None, None)
        if self.use_sentence_encoder:
            # transform sequences to sentences
            # Step 1, split a sequence of tokens into sentences
            sentence_inputs = transform_tokens2sentences(hidden_states,
                                                         num_sentences=num_sentences,
                                                         max_sentence_length=self.max_sentence_length)

            sentence_masks = transform_masks2sentences(attention_mask,
                                                       num_sentences=num_sentences,
                                                       max_sentence_length=self.max_sentence_length)

            sentence_outputs = self.sentence_encoder(sentence_inputs,
                                                     sentence_masks,
                                                     output_attentions=output_attentions)

            # transform sentences to tokens
            outputs = transform_sentences2tokens(sentence_outputs[0],
                                                 num_sentences=num_sentences,
                                                 max_sentence_length=self.max_sentence_length)
        else:
            outputs = hidden_states

        document_outputs = (None, None)

        if self.use_document_encoder:
            # gather sentence representative tokens
            # |  Sentence 1       | Sentence 2           | ...
            # [CLS] token .. token| [CLS] token .. token | ...
            #   V                     V                     V
            # [CLS]_1              [CLS]_2    ...        [CLS]_n
            # => CLS]_1 [CLS]_2 ... [CLS]_n
            sentence_global_tokens = outputs[:, ::self.max_sentence_length].clone()
            sentence_attention_mask = attention_mask[:, :, :, ::self.max_sentence_length].clone()

            sentence_positions = \
                torch.arange(1, num_sentences+1, device=outputs.device).repeat( \
                    outputs.size(0), 1) * (sentence_attention_mask.reshape( \
                        -1, num_sentences) >= -100).int().to(outputs.device)
            outputs[:, ::self.max_sentence_length] += \
                self.position_embeddings(sentence_positions)

            document_outputs = self.document_encoder(sentence_global_tokens,
                                                     sentence_attention_mask,
                                                     output_attentions=output_attentions)

            # replace sentence representative tokens
            outputs[:, ::self.max_sentence_length] = document_outputs[0]

        if output_attentions:
            return outputs, sentence_outputs[1], document_outputs[1]

        return outputs, None, None

class HATEncoder(nn.Module):
    """ Graph Hierarchical Transformer Encoder

        Parameters
        ----------
        config: HATConfig
            Model config
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [HATLayer(config,
                      use_sentence_encoder= \
                        self.config.encoder_layout[str(idx)]['sentence_encoder'],
                      use_document_encoder= \
                        self.config.encoder_layout[str(idx)]['document_encoder']) \
                for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        num_sentences=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True):
        """ Forward function
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_sentence_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    print("`use_cache=True` is incompatible with gradient checkpointing." \
                          "Setting `use_cache=False`...")
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask)
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    num_sentences,
                    output_attentions)

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_sentence_attentions = all_sentence_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_sentence_attentions
                ]
                if v is not None
            )

        return BaseModelOutputWithSentenceAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            sentence_attentions=all_sentence_attentions,
        )
    def _tie_weights(self):
        """
        Tie the weights between sentence positional embeddings across all layers.
        If the `torchscript` flag is set in the configuration,
        can't handle parameter sharing so we are cloning the
        weights instead.
        """
        original_position_embeddings = None
        for module in self.layer:
            if hasattr(module, "position_embeddings"):
                assert hasattr(module.position_embeddings, "weight")
                if original_position_embeddings is None:
                    original_position_embeddings = module.position_embeddings
                if self.config.torchscript:
                    module.position_embeddings.weight = nn.Parameter(original_position_embeddings.weight.clone())
                else:
                    module.position_embeddings.weight = original_position_embeddings.weight
        return

class HATPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HATConfig
    base_model_prefix = "hat"
    supports_gradient_checkpointing = True

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HATEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]

    @classmethod
    def from_config(cls, config):
        return cls._from_config(config)

class HATLMHead(nn.Module):
    """HAT Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class HATForMaskedLM(HATPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.hi_transformer = HATModel(config)
        self.lm_head = HATLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.hi_transformer.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.hi_transformer.embeddings.word_embeddings = value

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.hi_transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class HATTokenizer:
    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer
        self.config = HATConfig.from_pretrained(self._tokenizer.name_or_path)
        self._tokenizer.model_max_length = self.model_max_length
        self.type2id = {'input_ids': (self._tokenizer.cls_token_id, self._tokenizer.pad_token_id),
                        'token_type_ids': (0, 0),
                        'attention_mask': (1, 0),
                        'special_tokens_mask': (1, -100)}

    @property
    def model_max_length(self):
        return self.config.model_max_length

    @property
    def mask_token(self):
        return self._tokenizer.mask_token

    @property
    def mask_token_id(self):
        return self._tokenizer.mask_token_id

    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id

    @property
    def cls_token_id(self):
        return self._tokenizer.cls_token_id

    @property
    def sep_token_id(self):
        return self._tokenizer.sep_token_id

    @property
    def vocab(self):
        return self._tokenizer.vocab

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        return len(self._tokenizer)

    def pad(self, *args, **kwargs):
        return self._tokenizer.pad(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self._tokenizer.convert_tokens_to_ids(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self._tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._tokenizer.decode(*args, **kwargs)

    def tokenize(self, text, **kwargs):
        return self._tokenizer.tokenize(text, **kwargs)

    def encode(self, text, **kwargs):
        input_ids = self._tokenizer.encode_plus(text, add_special_tokens=False, **kwargs)
        input_ids = self.chunks(input_ids[: self.model_max_length - self.config.max_sentences],
                                chunk_size=self.config.max_sentence_length,
                                special_id=self.type2id['input_ids'])
        return input_ids

    def get_special_tokens_mask(self, *args, **kwargs):
        return self._tokenizer.get_special_tokens_mask(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        try:
            tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        except:
            tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(tokenizer=tokenizer)

    def save_pretrained(self, *args, **kwargs):
        return self._tokenizer.save_pretrained( *args, **kwargs)


def init_hat_for_mlm(lm_config):
    """ Init HAT model for masked LM pretraining

        Parameters
        ----------
        lm_config: dict
        Language model config.
    """
    model_name = lm_config["model_name"]
    config_kwargs = {
        "cache_dir": lm_config["cache_dir"] if "cache_dir" in lm_config else None,
        "revision": lm_config["model_revision"] if "model_revision" in lm_config else "main",
        "use_auth_token": lm_config["use_auth_token"] \
            if "use_auth_token" in lm_config else False
    }
    config = HATConfig.from_pretrained(model_name, **config_kwargs)
    model = HATForMaskedLM.from_pretrained(
        model_name,
        config=config)

    return model
