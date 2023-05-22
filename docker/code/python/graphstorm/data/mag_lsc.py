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

    Ogb MAG-LSC dataset (https://ogb.stanford.edu/kddcup2021/mag240m/).
"""
import os
from ogb.lsc import MAG240MDataset
import dgl
import torch as th

from .dataset import GSgnnDataset
from ..utils import sys_tracker

class MAGLSCDataset(GSgnnDataset):
    """ The MAG-LSC dataset.
    """
    def __init__(self, raw_dir, edge_pct=1,
                 force_reload=False, verbose=True, reverse_edge=True):
        self._raw_dir = raw_dir
        self._target_etype = ["writes"]
        self.edge_pct = edge_pct
        self._num_classes = None
        super(MAGLSCDataset, self).__init__("mag-lsc",
                                            url=None,
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
        sys_tracker.check('before saving the graph')
        dgl.save_graphs(os.path.join(path, gname), [self._g])
        sys_tracker.check('Done saving the graph')


    def process(self):
        """ Process ogbn dataset
        """
        dataset = MAG240MDataset(root=self._raw_dir)
        print(f"The MAG-LSC graph has {dataset.num_papers} paper nodes, " +
              f"{dataset.num_authors} author nodes, {dataset.num_institutions} institution nodes.")
        print(f"The paper nodes has {dataset.num_paper_features} features and " +
              f"{dataset.num_classes} classes.")
        self._num_classes = dataset.num_classes

        years = th.tensor(dataset.paper_year)
        labels = th.tensor(dataset.paper_label).long()
        print('years:', years.shape, ', labels:', labels.shape)

        edge_index_writes = dataset.edge_index('author', 'paper')
        edge_index_cites = dataset.edge_index('paper', 'paper')
        edge_index_affiliated_with = dataset.edge_index('author', 'institution')

        if self.reverse_edge:
            # add reverse edges
            g = dgl.heterograph({
                ("author", "writes", "paper"): (edge_index_writes[0], edge_index_writes[1]),
                ("paper", "rev-writes", "author"): (edge_index_writes[1], edge_index_writes[0]),
                ("paper", "cites", "paper"): (edge_index_cites[0], edge_index_cites[1]),
                ("paper", "rev-cites", "paper"): (edge_index_cites[1], edge_index_cites[0]),
                ("author", "affliated_with", "institution"): (edge_index_affiliated_with[0],
                                                              edge_index_affiliated_with[1]),
                ("institution", "rev-affliated_with", "author"): (edge_index_affiliated_with[1],
                                                                  edge_index_affiliated_with[0]),
            })
        else:
            g = dgl.heterograph({
                ("author", "writes", "paper"): (edge_index_writes[0], edge_index_writes[1]),
                ("paper", "cites", "paper"): (edge_index_cites[0], edge_index_cites[1]),
                ("author", "affliated_with", "institution"): (edge_index_affiliated_with[0],
                                                              edge_index_affiliated_with[1]),
            })

        # node masks
        sys_tracker.check('Create node masks')
        split_dict = dataset.get_idx_split()
        train_idx = split_dict['train']
        val_idx = split_dict['valid']
        test_idx = split_dict['test-dev']
        train_mask = th.full(labels.shape, False, dtype=th.bool)
        train_mask[train_idx] = True
        test_mask = th.full(labels.shape, False, dtype=th.bool)
        test_mask[test_idx] = True
        val_mask = th.full(labels.shape, False, dtype=th.bool)
        val_mask[val_idx] = True
        g.nodes['paper'].data['train_mask'] = train_mask.squeeze()
        g.nodes['paper'].data['test_mask'] = test_mask.squeeze()
        g.nodes['paper'].data['val_mask'] = val_mask.squeeze()
        g.nodes['paper'].data['labels'] = labels.squeeze()

        # edge masks
        # edge_pct has to be between 0.2 and 1 since we will use by default 0.1 for validation
        # and 0.1 for testing as the smallest possible.
        sys_tracker.check('Create edge masks')
        assert self.edge_pct <= 1 and  self.edge_pct >= 0.2
        int_edges = g.number_of_edges("writes")
        if self.edge_pct < 1:
            # the validation pct is 0.1
            val_pct = 0.1
            train_pct = self.edge_pct - val_pct
            # the test is 1 - the rest
            g.edges["writes"].data['train_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges["writes"].data['val_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges["writes"].data['test_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges["writes"].data['train_mask'][: int(int_edges*train_pct)] = True
            g.edges["writes"].data['val_mask'][int(int_edges*train_pct):
                                               int(int_edges*self.edge_pct)] = True
            g.edges["writes"].data['test_mask'][int(int_edges*self.edge_pct):] = True

            if self.reverse_edge:
                g.edges["rev-writes"].data['train_mask'] = th.full((int_edges,), False,
                                                                   dtype=th.bool)
                g.edges["rev-writes"].data['val_mask'] = th.full((int_edges,), False,
                                                                 dtype=th.bool)
                g.edges["rev-writes"].data['test_mask'] = th.full((int_edges,), False,
                                                                  dtype=th.bool)
                g.edges["rev-writes"].data['train_mask'][: int(int_edges * train_pct)] = True
                g.edges["rev-writes"].data['val_mask'][int(int_edges*train_pct):
                                                       int(int_edges*self.edge_pct)] = True
                g.edges["rev-writes"].data['test_mask'][int(int_edges*self.edge_pct):] = True

        sys_tracker.check('Get paper features.')
        g.nodes['paper'].data['feat'] = th.as_tensor(dataset.all_paper_feat)

        print(g)
        self._g=g

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
        return 'paper'

    @property
    def num_classes(self):
        """The number of classess of labels
        """
        return self._num_classes

    @property
    def target_etype(self):
        """ The target edge type for prediction.
        """
        return self._target_etype
