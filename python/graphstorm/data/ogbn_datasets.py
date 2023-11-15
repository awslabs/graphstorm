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

    Base OGB Dataset with text features
"""
import os
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import torch as th
import psutil

from .text_dataset import GSgnnTextDataset

class OGBTextFeatDataset(GSgnnTextDataset):
    """ This class can be used for ogbn-arxiv, ogbn-papers100M and ogbn-products datasets.

    The text features are collected from the original titles and abstractrs of the papers
    for the first two graphs and the ASIN titles for the last graphs. The text features
    should be stored in the raw_dir location.

    Parameters
    ----------
    raw_dir : str
        The file locations
    dataset : str
        The name of the dataset. It has to be either "ogbn-products", "ogbn-arxiv"
        and "ogbn-papers100M".
    edge_pct : float
        Percentage of edges in the test set
    force_reload : bool
    verbose : bool
        Whether to print additional messages.
    reverse_edge : bool
        Whether we include reverse edges
    self_loop : bool
        Whether we include self edges
    max_sequence_length : int
        The maximum supported sequence length
    retain_original_features : Boolean
        Whether to use the original features or the lm generated ones
    lm_model_name : String
        The lm model used for tokenization
    is_homo: Boolean
        If we want to generate a homogeneous graph.
    """
    def __init__(self, raw_dir, dataset, edge_pct=1,
                 force_reload=False, verbose=True,
                 reverse_edge=True, self_loop=False,
                 max_sequence_length=512,
                 retain_original_features=True,
                 lm_model_name='bert-base-uncased',
                 is_homo=False):
        """

        """
        self._name = 'ogbn'
        self._dataset = dataset
        self._url = None
        self._raw_dir = raw_dir
        self.self_loop = self_loop
        self.max_sequence_length = max_sequence_length
        self.is_homo = is_homo

        if self.is_homo:
            self.node_type = dgl.distributed.constants.DEFAULT_NTYPE
            self.edge_type = dgl.distributed.constants.DEFAULT_ETYPE[1]
        else:
            self.node_type = 'node'
            self.edge_type, self.rev_edge_type = 'interacts', 'rev-interacts'
        self.target_etype = (self.node_type, self.edge_type, self.node_type)

        self.edge_pct = edge_pct
        if dataset == "ogbn-products":
            self._num_classes = 47
        elif dataset == "ogbn-arxiv":
            self._num_classes = 40
        elif dataset == "ogbn-papers100M":
            self._num_classes = 172
        self.retain_original_features = retain_original_features
        self.lm_model_name=lm_model_name
        super(OGBTextFeatDataset, self).__init__(self._name,
                                                 url=self._url,
                                                 raw_dir=raw_dir,
                                                 force_reload=force_reload,
                                                 verbose=verbose,
                                                 reverse_edge=reverse_edge)

    def load(self):
        # load from local storage
        root_path = self._raw_dir
        gname = self._name+'.bin'
        g, _ = dgl.load_graphs(os.path.join(root_path, gname))
        print(g[0])
        self._g = g[0]

    def has_cache(self):
        root_path = self._raw_dir
        gname = self._name+'.bin'
        return os.path.exists(os.path.join(root_path, gname))

    def save_graph(self, path):
        """ Save processed graph and the raw text data into disk.

        The raw texts for each node type are stored separately into different files.

        Parameters
        ----------
        path : str
            Where to save the output
        """
        # save the processed data
        gname = self._name + '.bin'
        print("Save graph {} into {}".format(self._g, os.path.join(path, gname)))
        print('before save graph {}'.format(psutil.virtual_memory()))
        dgl.save_graphs(os.path.join(path, gname), [self._g])
        print('Done save graph {}'.format(psutil.virtual_memory()))


    def process(self):
        """ Process ogbn dataset
        """
        data = DglNodePropPredDataset(name=self._dataset)
        print("Graph nodes ={}".format(data.graph[0].num_nodes()))

        # If we don't retain the original node features, we use text tokens as node features.
        if not self.retain_original_features:
            # this file contains the text data each line corresponds to a node id
            with open(os.path.join(self._raw_dir, "X.all.txt"),
                      "r", encoding='utf-8') as fin:
                text_feats_list = fin.readlines()
            assert len(text_feats_list) == data.graph[0].num_nodes()
            print("|node_text_list={}".format(len(text_feats_list)))

            # We tokenize the text before loading the ogbn graph into memory.
            # This helps reduce the overhead of creating multiple worker processes
            # during text tokenization. When a graph is large (e.g., papers100m),
            # the overhead is not nigligiable.
            self._raw_text_feat = {self.node_type:text_feats_list}
            text_feat = self.tokenize_text(self.max_sequence_length,
                                           lm_model_name=self.lm_model_name)

        splitted_idx = data.get_idx_split()
        train_idx = splitted_idx["train"]
        val_idx = splitted_idx["valid"]
        test_idx = splitted_idx["test"]
        graph, labels = data[0]
        labels = labels.long()
        self._num_classes = data.num_classes

        src, dst = graph.edges()
        # adding dummy node type since the returned seeds from the dataloader
        # will not have node type if only one is present
        if self.reverse_edge:
            # add reverse edges
            if not self.is_homo:
                g = dgl.heterograph({
                    (self.node_type, self.edge_type, self.node_type): (src, dst),
                    (self.node_type, self.rev_edge_type, self.node_type): (dst, src)
                })
            else:
                # create a homogeneous graph first and add reverse edges
                g = dgl.add_reverse_edges(dgl.heterograph({
                    (self.node_type, self.edge_type, self.node_type): (src, dst),
                }))
        else:
            g = dgl.heterograph({
                (self.node_type, self.edge_type, self.node_type): (src, dst)
            })

        # add self-loop
        if self.self_loop:
            print(f"Total edges before adding self-loop {graph.number_of_edges()}")
            g = g.remove_self_loop(self.edge_type).add_self_loop(self.edge_type)
            if self.reverse_edge:
                g = g.remove_self_loop(self.rev_edge_type).add_self_loop(self.rev_edge_type)
            print(f"Total edges after adding self-loop {graph.number_of_edges()}")

        # node masks
        train_mask = th.full(labels.shape, False, dtype=th.bool)
        train_mask[train_idx] = True
        test_mask = th.full(labels.shape, False, dtype=th.bool)
        test_mask[test_idx] = True
        val_mask = th.full(labels.shape, False, dtype=th.bool)
        val_mask[val_idx] = True
        g.nodes[self.node_type].data['train_mask'] = train_mask.squeeze()
        g.nodes[self.node_type].data['test_mask'] = test_mask.squeeze()
        g.nodes[self.node_type].data['val_mask'] = val_mask.squeeze()
        g.nodes[self.node_type].data['labels'] = labels.squeeze()

        # edge masks
        # edge_pct has to be between 0.2 and 1 since we will use by default 0.1 for validation
        # and 0.1 for testing as the smallest possible.
        assert self.edge_pct <= 1 and  self.edge_pct >= 0.2
        int_edges = g.number_of_edges(self.edge_type)
        if self.edge_pct == 1:
            g.edges[self.edge_type].data['train_mask'] = th.full((int_edges,), True, dtype=th.bool)
            if self.reverse_edge:
                g.edges[self.edge_type].data['train_mask'] = th.full((int_edges,), True,
                                                                      dtype=th.bool)
        else:
            # the validation pct is 0.1
            val_pct = 0.1
            train_pct = self.edge_pct - val_pct
            # the test is 1 - the rest
            g.edges[self.edge_type].data['train_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges[self.edge_type].data['val_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges[self.edge_type].data['test_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges[self.edge_type].data['train_mask'][: int(int_edges*train_pct)] = True
            g.edges[self.edge_type].data['val_mask'][int(int_edges*train_pct):
                                                  int(int_edges*self.edge_pct)] = True
            g.edges[self.edge_type].data['test_mask'][int(int_edges*self.edge_pct):] = True

            if self.reverse_edge and not self.is_homo:
                g.edges[self.rev_edge_type].data['train_mask'] = th.full((int_edges,), False,
                                                                      dtype=th.bool)
                g.edges[self.rev_edge_type].data['val_mask'] = th.full((int_edges,), False,
                                                                    dtype=th.bool)
                g.edges[self.rev_edge_type].data['test_mask'] = th.full((int_edges,), False,
                                                                     dtype=th.bool)
                g.edges[self.rev_edge_type].data['train_mask'][: int(int_edges * train_pct)] = True
                g.edges[self.rev_edge_type].data['val_mask'][int(int_edges*train_pct):
                                                          int(int_edges*self.edge_pct)] = True
                g.edges[self.rev_edge_type].data['test_mask'][int(int_edges*self.edge_pct):] = True

        print(g)
        self._g=g
        self._num_classes = data.num_classes

        print("Retaining original node features and discarding the text data. \
              This is the original input of ogbn.")
        self._g.nodes[self.node_type].data['feat'] = graph.ndata["feat"]

        if self.retain_original_features:
            print("Retaining original node features and "
                  "discarding the text data. This is the original input of ogbn.")
            self._g.nodes[self.node_type].data['feat'] = graph.ndata["feat"]
            self._raw_text_feat = {}
        else:
            # tokenize the original text
            for ntype in self._g.ntypes:
                for name in text_feat[ntype]:
                    self._g.nodes[ntype].data[name] = text_feat[ntype][name]

    def __getitem__(self, idx):
        r"""Gets the data object at index.
        """
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1

    @property
    def predict_category(self):
        """The node type to be predicted, which is node in this base dataset
        """
        return self.node_type

    @property
    def num_classes(self):
        """The number of classess of labels
        """
        return self._num_classes

    def _download_bert_embeddings(self):
        """
        This function downloads the bert embedding
        that are uploaded in the s3 if these exists otherwise None.
        Returns
        -------
        The embeddings dictionary
        """
        raise NotImplementedError
