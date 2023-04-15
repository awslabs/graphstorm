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

    Generate example graph data using built-in datasets for node classifcation,
    node regression, edge classification and edge regression.
"""

class Tokenizer:
    """ A wrapper to a tokenizer.

    It is defined to process multiple strings.

    Parameters
    ----------
    tokenizer : a tokenizer
    max_seq_length : int
        The maximal length of the tokenization results.
    """
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, strs):
        """ Tokenization function.

        Parameters
        ----------
        strs : list of strings.
            The text data to be tokenized.

        Returns
        -------
        a dict of tokenization results.
        """
        tokens = []
        att_masks = []
        type_ids = []
        for s in strs:
            t = self.tokenizer(s, max_length=self.max_seq_length,
                               truncation=True, padding='max_length', return_tensors='pt')
            tokens.append(t['input_ids'])
            # The masks are small integers. We can use int4 or int8 to store them.
            # This can signficantly reduce memory consumption.
            att_masks.append(t['attention_mask'].to(th.int8))
            type_ids.append(t['token_type_ids'].to(th.int8))
        return {'token_ids': th.cat(tokens, dim=0).numpy(),
                'attention_mask': th.cat(att_masks, dim=0).numpy(),
                'token_type_ids': th.cat(type_ids, dim=0).numpy()}

def parse_tokenize(op):
    """ Parse the tokenization configuration

    The parser returns a function that tokenizes text with HuggingFace tokenizer.
    The tokenization function returns a dict of three Pytorch tensors.

    Parameters
    ----------
    op : dict
        The configuration for the operation.

    Returns
    -------
    callable : a function to process the data.
    """
    tokenizer = BertTokenizer.from_pretrained(op['bert_model'])
    max_seq_length = int(op['max_seq_length'])
    return Tokenizer(tokenizer, max_seq_length)

def parse_feat_ops(confs):
    """ Parse the configurations for processing the features

    The feature transformation:
    {
        "feature_col":  ["<column name>", ...],
        "feature_name": "<feature name>",
        "data_type":    "<feature data type>",
        "transform":    {"name": "<operator name>", ...}
    }

    Parameters
    ----------
    confs : list
        A list of feature transformations.

    Returns
    -------
    list of tuple : The operations
    """
    ops = []
    assert isinstance(confs, list), \
            "The feature configurations need to be in a list."
    for feat in confs:
        # TODO(zhengda) we will support data type in the future.
        dtype = None
        if 'transform' not in feat:
            transform = None
        else:
            transform = feat['transform']
            assert 'name' in transform, \
                    "'name' must be defined in the transformation field."
            if transform['name'] == 'tokenize_hf':
                transform = parse_tokenize(transform)
            else:
                raise ValueError('Unknown operation: {}'.format(transform['name']))
        feat_name = feat['feature_name'] if 'feature_name' in feat else None
        assert 'feature_col' in feat, \
                "'feature_col' must be defined in a feature field."
        ops.append((feat['feature_col'], feat_name, dtype, transform))
    return ops

