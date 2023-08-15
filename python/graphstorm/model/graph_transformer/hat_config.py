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

    Graph Hierarchical Transformer Config
"""
from transformers import PretrainedConfig

class HATConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.HAT`.
    It is used to instantiate a HAT model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults
    will yield a similar configuration
    to that of the HAT `kiddothe2b/hierarchical-transformer-base-4096
    <https://huggingface.co/kiddothe2b/hierarchical-transformer-base-4096>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig`
    and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig`
    for more information.

    Args:
        vocab_size: int
            Vocabulary size of the BERT model. Defines the number of different
            tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
            defaults: 30522
        max_sentences: int
            The maximum number of sentences that this model might ever be used with.
            default:64
        max_sentence_size: int
            The maximum sentence length that this model might ever be used with.
            defaults:128
        model_max_length: int
            The maximum  sequence length (max_sentences * max_sentence_size) that this model might ever be used with
            defaults:8192
        encoder_layout: Dict
            The sentence/document encoder layout.
        hidden_size: int
            Dimensionality of the encoder layers and the pooler layer.
            defaults:768
        num_hidden_layers: int
            Number of hidden layers in the Transformer encoder.
            defaults:12
        num_attention_heads: int
            Number of attention heads for each attention layer in the Transformer encoder.
            defaults:12
        intermediate_size: int
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            default:3072
        hidden_act: str or :obj:`Callable`
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
            default: gelu
        hidden_dropout_prob: float
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            defaults: 0.1
        attention_probs_dropout_prob: float
            The dropout ratio for the attention probabilities.
            defaults: 0.1
        max_position_embeddings: int
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
            defaults: 512
        type_vocab_size int
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
            defaults:2
        initializer_range: float
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            defaults: 0.02
        layer_norm_eps: float
            The epsilon used by the layer normalization layers.
            defaults: 1e-12
        position_embedding_type: str
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
            defaults: "absolute"
        use_cache: bool
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
            default: True
        classifier_dropout: float
            The dropout ratio for the classification head.
    """
    model_type = "hi-transformer"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        max_sentences=64,
        max_sentence_size=128,
        model_max_length=8192,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        encoder_layout=None,
        use_cache=True,
        classifier_dropout=None,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_sentences = max_sentences
        self.max_sentence_size = max_sentence_size
        self.model_max_length = model_max_length
        self.encoder_layout = encoder_layout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
