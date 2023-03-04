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

    GraphStorm dataset that wrap the OGB MAG dataset
"""
import os
import dgl
import torch as th
from ogb.nodeproppred import DglNodePropPredDataset
from graphstorm.data.dataset import GSgnnDataset


class OGBMAGTextFeatDataset(GSgnnDataset):
    r""" OGB MAG graph dataset wrapped for GSF

        For Link prediction task, add an edge_pct argument to split the 'writes' edge for train/val/
    """
    def __init__(self, raw_dir, text_feat_type='paper', edge_pct=1, force_reload=False,
                 verbose=True, reverse_edge=True):
        self._name = 'ogbn-mag'
        self._raw_dir = raw_dir
        self._url = None
        self._text_feat_type = text_feat_type       # So far, not used
        # Directly download data will not retrain text embedding for the model, so set to True
        self.retain_original_features = True
        self.edge_pct = edge_pct
        self._force_reload = force_reload
        self._reverse_edge = reverse_edge
        # specific to the MAG data
        self._target_etype = 'writes'
        self._target_ntype = 'paper'

        super(OGBMAGTextFeatDataset, self).__init__(name=self._name,
                                                    url=self._url,
                                                    raw_dir=raw_dir,
                                                    force_reload=force_reload,
                                                    verbose=verbose,
                                                    reverse_edge=reverse_edge)

    def load(self):
        """ Load data from the local storage
        """
        root_path = self._raw_dir
        gname = self._name + '.bin'
        g, _ = dgl.load_graphs(os.path.join(root_path, gname))
        print(g[0])
        self._g = g[0]

    def has_cache(self):
        """ Check if the processed data already exist in the local storage
        """
        root_path = self._raw_dir
        gname = self._name+'.dgl'
        return os.path.exists(os.path.join(root_path, gname))

    def save_graphs(self, path):
        """ Save processed graph  into the local storage

        Parameters
        ----------
        path : str
            Where to save the output
        """
        # save the processed data
        gname = self._name + '.dgl'
        print("Save graph {} into {}".format(self._g, os.path.join(path, gname)))
        print('before saving the graph')
        dgl.save_graphs(os.path.join(path, gname), [self._g])
        print('Done saving the graph')

    def process(self):
        """ Process the OGBN MAG data
        1. reconstruct the MAG heterogeneous graph;
        2. split train/val/test for the paper nodes and assign mask;
        3. assign masks and label to the target;
        4. split train/val/test for the writes edges and assign mask;
        5. extract the orignal embedding features and assign
        """
        data = DglNodePropPredDataset(name=self._name)
        graph, labels = data[0]
        splitted_idx = data.get_idx_split()
        self._num_classes = data.num_classes

        num_paper_nodes = graph.num_nodes('paper')
        num_author_nodes = graph.num_nodes('author')
        num_institution_nodes = graph.num_nodes('institution')
        num_topic_nodes = graph.num_nodes('field_of_study')
        print(f'The OGB MAG graph has {num_paper_nodes} paper nodes, {num_author_nodes} author \
                nodes, {num_institution_nodes} institution nodes, and {num_topic_nodes} \
                topic nodes.')

        years = graph.nodes['paper'].data['year']
        labels = labels['paper'].long()
        print('years:', years.shape, ', labels:', labels.shape)

        edge_dict = {}
        for src_type, e_type, dst_type in graph.canonical_etypes:
            src, dst = graph.edges(etype=(src_type, e_type, dst_type))

            if self._reverse_edge:
                edge_dict[(src_type, e_type, dst_type)] = (src, dst)
                edge_dict[(dst_type, 'rev-' + e_type, src_type)] = (dst, src)
            else:
                edge_dict[(src_type, e_type, dst_type)] = (src, dst)

        g = dgl.heterograph(edge_dict)

        train_idx = splitted_idx["train"]['paper']
        val_idx = splitted_idx["valid"]['paper']
        test_idx = splitted_idx["test"]['paper']

        # node masks
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
        assert self.edge_pct <= 1 and  self.edge_pct >= 0.2
        int_edges = g.number_of_edges("writes")
        if self.edge_pct < 1:
            # the validation pct is 0.1
            val_pct = 0.1
            train_pct = self.edge_pct - val_pct
            # the test is 1 - the rest

            if self._reverse_edge:
                g.edges["writes"].data['train_mask'] = th.full((int_edges,),
                                                               False,
                                                               dtype=th.bool)
                g.edges["rev-writes"].data['train_mask'] = th.full((int_edges,),
                                                                   False,
                                                                   dtype=th.bool)
                g.edges["writes"].data['val_mask'] = th.full((int_edges,),
                                                             False,
                                                             dtype=th.bool)
                g.edges["rev-writes"].data['val_mask'] = th.full((int_edges,),
                                                                 False,
                                                                 dtype=th.bool)
                g.edges["writes"].data['test_mask'] = th.full((int_edges,),
                                                              False,
                                                              dtype=th.bool)
                g.edges["rev-writes"].data['test_mask'] = th.full((int_edges,),
                                                                  False,
                                                                  dtype=th.bool)

                g.edges["writes"].data['train_mask'][: int(int_edges*train_pct)] = True
                g.edges["rev-writes"].data['train_mask'][: int(int_edges * train_pct)] = True
                g.edges["writes"].data['val_mask'][int(int_edges*train_pct):
                                                int(int_edges*self.edge_pct)] = True
                g.edges["rev-writes"].data['val_mask'][int(int_edges*train_pct):
                                                    int(int_edges*self.edge_pct)] = True
                g.edges["writes"].data['test_mask'][int(int_edges*self.edge_pct):] = True
                g.edges["rev-writes"].data['test_mask'][int(int_edges*self.edge_pct):] = True
            else:
                g.edges["writes"].data['train_mask'] = th.full((int_edges,), False, dtype=th.bool)
                g.edges["writes"].data['val_mask'] = th.full((int_edges,), False, dtype=th.bool)
                g.edges["writes"].data['test_mask'] = th.full((int_edges,), False, dtype=th.bool)

                g.edges["writes"].data['train_mask'][: int(int_edges*train_pct)] = True
                g.edges["writes"].data['val_mask'][int(int_edges*train_pct):
                                                int(int_edges*self.edge_pct)] = True
                g.edges["writes"].data['test_mask'][int(int_edges*self.edge_pct):] = True

        if self.retain_original_features:
            print("Retaining original node features and discarding the text data. \
                  This is the original input of ogbn.")
            g.nodes['paper'].data['feat'] = graph.nodes['paper'].data["feat"]

        self._g = g

    def __getitem__(self, idx):
        r"""Gets the data object at index.
        """
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        r""" This dataset has only one graph so the length is 1
        """
        return 1

    @property
    def predict_category(self):
        """The node type to be predicted, which is 'paper' in this MAG dataset
        """
        return self._target_ntype

    @property
    def num_classes(self):
        """The number of classess of labels
        """
        return self._num_classes

    @property
    def target_etype(self):
        """The edge type to be predicted, which is 'writes' in the MAG dataset
        """
        return self._target_etype
