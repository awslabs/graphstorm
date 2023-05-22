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

    Base dataset for GSF dataset
"""
import multiprocessing
import torch as th
from transformers import AutoTokenizer
from .dataset import GSgnnDataset
from .constants import TOKEN_IDX, ATT_MASK_IDX, VALID_LEN

def local_tokenize(inputs):
    """
    This function applies the tokenizer on a batch of the input data.
    Returns
     -------
    Dictionary of tokens
    """
    bert_model_name, max_seq_length, val = inputs
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    sub_tokens = tokenizer(val,  max_length=max_seq_length,
                           truncation=True, padding=True, return_tensors='pt')

    # we only use TOKEN_IDX and ATT_MASK_IDX
    input_ids = sub_tokens[TOKEN_IDX].share_memory_()
    valid_len = sub_tokens[ATT_MASK_IDX].sum(dim=1).share_memory_()

    # We store valid attention mask length in VALID_LEN to save space.
    # We will convert it into ATT_MASK_IDX during forward pass.
    return {
        TOKEN_IDX: input_ids,
        VALID_LEN: valid_len
    }

class GSgnnTextDataset(GSgnnDataset):
    """r Basic class of GSgnn dataset
    """
    def __init__(self, name, url, raw_dir, force_reload=False, verbose=True, reverse_edge=True):
        self._encoding = 'utf-8'
        self._raw_text_feat = None

        super(GSgnnTextDataset, self).__init__(name,
                                               url=url,
                                               raw_dir=raw_dir,
                                               force_reload=force_reload,
                                               verbose=verbose,
                                               reverse_edge=reverse_edge)

    def _download_bert_embeddings(self):
        """
        This function downloads the bert embedding
        that are uploaded in the s3 if these exists otherwise None.
        Returns
        -------
        The embeddings dictionary
        """
        return None

    def load(self):
        # not implemented
        self._g = None
        raise NotImplementedError

    def tokenize_text(self, max_seq_length=128, bert_model_name='bert-base-uncased',
                      num_workers=32):
        """ Tokenize the raw text feature.

        We use python multiprocessing to tokenize the text in parallel.
        Please call this function at the very begnning of data processing
        logic, otherwise the cost of creating workers are huge (when the
        intended graph is very large).

        Parameters
        ----------
        max_seq_length: int
            max sequency length
        bert_model_name: str
            Huggingface bert model name
        num_workers: int
            max parallelism
        Returns
        ---------
        A dictionary of tokenized text data from differnt node types.
        """
        print("Using for tokenizer "+bert_model_name)
        limit = num_workers
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()
            if multiprocessing.cpu_count() < limit else limit)
        new_feats = {}
        with pool:
            for key, values in self._raw_text_feat.items():
                chunksize = 102400

                tasks = [values[x:x+chunksize] for x in range(0, len(values), chunksize)]
                print("Tokenizing: {}:{}/{}".format(key, len(values), len(tasks)))
                outputs = pool.map(local_tokenize, [(bert_model_name, max_seq_length, task)
                                                    for task in tasks])

                input_ids = []
                valid_len = []

                for output in outputs:
                    input_ids.append(output[TOKEN_IDX])
                    valid_len.append(output[VALID_LEN])

                input_ids = th.cat(input_ids)
                valid_len = th.cat(valid_len)

                print("Done tokenizing {}".format(key))
                # We store valid attention mask length in VALID_LEN to save space.
                # We will convert it into ATT_MASK_IDX during forward pass.
                new_feats[key] = {
                        TOKEN_IDX: input_ids,
                        VALID_LEN: valid_len,
                    }
        return new_feats

    @property
    def raw_text_feat(self):
        """
        The raw text per node.
        """
        return self._raw_text_feat
