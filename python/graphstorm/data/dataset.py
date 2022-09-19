from dgl.data.dgl_dataset import DGLDataset
from transformers import AutoTokenizer
import torch as th
import dgl
import os
import multiprocessing
from .constants import TOKEN_IDX, ATT_MASK_IDX, VALID_LEN_IDX

def local_tokenize(inputs):
    bert_model_name, max_seq_length, val = inputs
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    sub_tokens = tokenizer(val,  max_length=max_seq_length,truncation=True, padding=True, return_tensors='pt')

    # For either m5 bert or huggingface bert,
    # we only use TOKEN_IDX and VALID_LEN_IDX
    input_ids = sub_tokens[TOKEN_IDX].share_memory_()
    valid_len = sub_tokens[ATT_MASK_IDX].sum(dim=1).share_memory_()

    return {
        TOKEN_IDX: input_ids,
        VALID_LEN_IDX: valid_len
    }

class GSgnnDataset(DGLDataset):
    """r Basic class of GSgnn dataset
    """
    def __init__(self, name, url, raw_dir, force_reload=False, verbose=True, reverse_edge=True):
        self._encoding = 'utf-8'
        self._raw_text_feat = None

        super(GSgnnDataset, self).__init__(name,
                                           url=url,
                                           raw_dir=raw_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)

    def load(self):
        pass

    def has_cache(self):
        return False

    def save(self, path=None):
        pass

    def download(self):
        if self.raw_dir is not None:
            return
        assert False, "Will add auto download in the future"

    def process(self):
        pass

    def __getitem__(self, idx):
        r"""Gets the data object at index.
        """
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1

    def _load_bert_embeddings(self, cache_path, dataset):
        """
        This function loads the bert embedding locally if these exists otherwise None.
        Returns
        -------
        The embeddings dictionary
        """
        try:
            embs = th.load(os.path.join(cache_path, dataset+"_BERT_embeddings_abstract.th"))
        except:
            embs = None

        return embs

    def _download_bert_embeddings(self):
        """
        This function downloads the bert embedding that are uploaded in the s3 if these exists otherwise None.
        Returns
        -------
        The embeddings dictionary
        """
        return None

    @property
    def raw_text_feat(self):
        return self._raw_text_feat

    def tokenzie_text(self, max_seq_length=128, bert_model_name='bert-base-uncased', num_workers=32):
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
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() if multiprocessing.cpu_count() < limit else limit)

        new_feats = {}
        for key, values in self._raw_text_feat.items():
            chunksize = 102400

            tasks = [values[x:x+chunksize] for x in range(0, len(values), chunksize)]
            print("Tokenizing: {}:{}/{}".format(key, len(values), len(tasks)))
            outputs = pool.map(local_tokenize, [(bert_model_name, max_seq_length, task) for task in tasks])

            input_ids = []
            valid_len = []

            for output in outputs:
                input_ids.append(output[TOKEN_IDX])
                valid_len.append(output[VALID_LEN_IDX])

            input_ids = th.cat(input_ids)
            valid_len = th.cat(valid_len)

            print("Done tokenizing {}".format(key))
            new_feats[key] = {
                    TOKEN_IDX: input_ids,
                    VALID_LEN_IDX: valid_len,
                }
        return new_feats

    def preprocess_text_feat(self, max_seq_length=128, sep_token_extra=False, action='tokenize', cache_path=None,
                             bert_model_name='bert-base-uncased'):
        if action == 'tokenize':
            if cache_path is not None:
                print("Loading tokenized input from cache ...")
                for ntype in self._g.ntypes:
                    if 'text_idx' in self._g.nodes[ntype].data:
                        print("loading: %s" % ntype)
                        for dtype in [TOKEN_IDX, VALID_LEN_IDX]:
                            self._g.nodes[ntype].data[dtype] = th.load(os.path.join(cache_path,
                                         'token_%s_%s.pt' % (ntype, dtype)))
                return

            print("Using for tokenizer "+bert_model_name)
            tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            new_feats = {}
            for key, values in self._raw_text_feat.items():
                print("Tokenizing: %s" % key)
                tokens = tokenizer(values,  max_length=max_seq_length,truncation=True, padding=True, return_tensors='pt')
                new_feats[key] = {
                    TOKEN_IDX: tokens[TOKEN_IDX],
                    VALID_LEN_IDX: tokens[ATT_MASK_IDX].sum(dim=1)
                }
            for ntype in self._g.ntypes:
                if 'text_idx' not in self._g.nodes[ntype].data:
                    continue
                idx = self._g.nodes[ntype].data['text_idx']
                for name in new_feats[ntype]:
                    self._g.nodes[ntype].data[name] = new_feats[ntype][name][idx]
        else:
            assert False

class ConstructedGraphDataset(GSgnnDataset):
    """ A general dgl graph dataset wrapper used for loading the
        output DGL Graph of construct_graph.py.

        By default, construct_graph.py creates two output: 1) constructed graph; 2) partioned graph.
        ConstructedGraphDataset is used to load the constructed graph.

        Note: The graph filename should be path/<name>.dgl

        Examples:
        python3 construct_graph.py --name example --filepath input --output data --dist_output dist_data ...

        >>> dataset = ConstructedGraphDataset(example, data)
        >>> g = dataset[0]

    """
    def __init__(self, name, path, verbose=True):
        super(ConstructedGraphDataset, self).__init__(name,
                                                      url=None,
                                                      raw_dir=path,
                                                      force_reload=False,
                                                      verbose=verbose)

    def download(self):
        pass

    def process(self):
        root_path = self._raw_dir
        gname = self._name + ".dgl"
        assert os.path.exists(os.path.join(root_path, gname)), \
            "A saved DGLGraph file named {} should exist under {}".format(gname, root_path)
        g, _ = dgl.load_graphs(os.path.join(root_path, gname))
        self._g = g[0]
