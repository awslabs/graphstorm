from ogb.nodeproppred import DglNodePropPredDataset
import csv
import dgl
import torch as th
import boto3
import psutil
import os
import argparse

from .dataset import M5gnnDataset

class OGBTextFeatDataset(M5gnnDataset):
    """
    This class can be used for ogbn-arxiv, ogbn-papers100M and ogbn-products datasets. The text features are collected
    from the original titles and abstractrs of the papers for the first two graphs and the ASIN titles for the last
    graphs. The text features should be stored in the raw_dir location.
    """
    def __init__(self, raw_dir, dataset, edge_pct=1,
                 retain_original_features=False,
                 force_reload=False, verbose=True,
                 reverse_edge=True, self_loop=False,
                 max_sequence_length=512,
                 bert_model_name='bert-base-uncased'):
        """

        Parameters
        ----------
        raw_dir  str The file locations
        retain_original_features bool whether we retain the original features.
        force_reload
        verbose
        reverse_edge bool whether we include reverse edges
        self_loop bool whether we include self edges
        max_sequence_length int what is the maximum supported sequence length
        edge_pct float percentage of edges in the test set
        """
        self._name = 'ogbn'
        self._dataset=dataset
        self._url = None
        self._raw_dir=raw_dir
        self.reverse_edge=reverse_edge
        self.self_loop=self_loop
        self.max_sequence_length=max_sequence_length
        self.retain_original_features = retain_original_features
        self.target_etype = "interacts"
        self.edge_pct = edge_pct
        self.bert_model_name=bert_model_name
        if dataset == "ogbn-products":
            self._num_classes=47
        elif dataset=="ogbn-arxiv":
            self._num_classes=40
        elif dataset=="ogbn-papers100M":
            self._num_classes = 172
        super(OGBTextFeatDataset, self).__init__(self._name,
                                                      url=self._url,
                                                      raw_dir=raw_dir,
                                                      force_reload=force_reload,
                                                      verbose=verbose)

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
        A query_asin_match.pkl file is created to store the meta-information of raw text data.

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
        # this file contains the text data each line corresponds to a node id
        with open(os.path.join(self._raw_dir, "X.all.txt"), "r") as fin:
            text_feats_list = fin.readlines()
        print("|node_text_list={}".format(len(text_feats_list)))

        # We tokenize the text before loading the ogbn graph into memory.
        # This helps reduce the overhead of creating multiple worker processes
        # during text tokenization. When a graph is large (e.g., papers100m),
        # the overhead is not nigligiable.
        self._raw_text_feat = {'node':text_feats_list}
        text_feat = self.tokenzie_text(self.max_sequence_length,
                                       bert_model_name=self.bert_model_name)

        data = DglNodePropPredDataset(name=self._dataset)
        print("Graph nodes ={}".format(data.graph[0].num_nodes()))
        assert len(text_feats_list) == data.graph[0].num_nodes()

        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
        graph, labels = data[0]
        labels = labels.long()
        self._num_classes = data.num_classes

        src, dst = graph.edges()
        # adding dummy node type since the returned seeds from the dataloader
        # will not have node type if only one is present
        if self.reverse_edge:
            # add reverse edges
            g = dgl.heterograph({
                ("node", "interacts", "node"): (src, dst),
                ("node", "rev-interacts", "node"): (dst, src)
            })
        else:
            g = dgl.heterograph({
                ("node", "interacts", "node"): (src, dst)
            })

        # add self-loop
        if self.self_loop:
            print(f"Total edges before adding self-loop {graph.number_of_edges()}")
            g = g.remove_self_loop('interacts').add_self_loop('interacts')
            if self.reverse_edge:
                g = g.remove_self_loop('rev-interacts').add_self_loop('rev-interacts')
            print(f"Total edges after adding self-loop {graph.number_of_edges()}")

        # node masks
        train_mask = th.full(labels.shape, False, dtype=th.bool)
        train_mask[train_idx] = True
        test_mask = th.full(labels.shape, False, dtype=th.bool)
        test_mask[test_idx] = True
        val_mask = th.full(labels.shape, False, dtype=th.bool)
        val_mask[val_idx] = True
        g.nodes['node'].data['train_mask'] = train_mask.squeeze()
        g.nodes['node'].data['test_mask'] = test_mask.squeeze()
        g.nodes['node'].data['val_mask'] = val_mask.squeeze()
        g.nodes['node'].data['labels'] = labels.squeeze()

        # edge masks
        # edge_pct has to be between 0.2 and 1 since we will use by default 0.1 for validation and 0.1 for testing as
        # the smallest possible.
        assert self.edge_pct <= 1 and  self.edge_pct >= 0.2
        int_edges = g.number_of_edges("interacts")
        if self.edge_pct == 1:
            g.edges["interacts"].data['train_mask'] = th.full((int_edges,), True, dtype=th.bool)
            g.edges["rev-interacts"].data['train_mask'] = th.full((int_edges,), True, dtype=th.bool)
        else:
            # the validation pct is 0.1
            val_pct = 0.1
            train_pct = self.edge_pct - val_pct
            # the test is 1 - the rest
            g.edges["interacts"].data['train_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges["rev-interacts"].data['train_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges["interacts"].data['val_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges["rev-interacts"].data['val_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges["interacts"].data['test_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges["rev-interacts"].data['test_mask'] = th.full((int_edges,), False, dtype=th.bool)

            g.edges["interacts"].data['train_mask'][: int(int_edges*train_pct)] = True
            g.edges["rev-interacts"].data['train_mask'][: int(int_edges * train_pct)] = True
            g.edges["interacts"].data['val_mask'][int(int_edges*train_pct):int(int_edges*self.edge_pct)] = True
            g.edges["rev-interacts"].data['val_mask'][int(int_edges*train_pct):int(int_edges*self.edge_pct)] = True
            g.edges["interacts"].data['test_mask'][int(int_edges*self.edge_pct):] = True
            g.edges["rev-interacts"].data['test_mask'][int(int_edges*self.edge_pct):] = True

        print(g)
        self._g=g
        self._num_classes = data.num_classes

        if self.retain_original_features:
            print("Retaining original node features and discarding the text data. This is the original input of ogbn.")
            self._g.nodes['node'].data['feat'] = graph.ndata["feat"]
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
        return 'node'

    @property
    def num_classes(self):
        return self._num_classes

    def _download_bert_embeddings(self):
        """
        This function downloads the bert embedding that are uploaded in the s3 if these exists otherwise None.
        Returns
        -------
        The embeddings dictionary
        """
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='query-asin')
    parser.add_argument("--filepath", type=str, default=None)
    parser.add_argument("--savepath", type=str, default=None)
    parser.add_argument("--edge_pct", type=float, default=1)
    parser.add_argument("--dataset",type=str,default="ogbn-arxiv")
    parser.add_argument('--bert_model_name',type=str,default="bert-base-uncased")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--retain_original_features", type=lambda x: (str(x).lower() in ['true', '1']), default=False)
    args = parser.parse_args()
    # only for test
    dataset = OGBTextFeatDataset(args.filepath,
                                 edge_pct=args.edge_pct,
                                 dataset=args.dataset,
                                 bert_model_name=args.bert_model_name,
                                 max_sequence_length=args.max_sequence_length,
                                 retain_original_features=args.retain_original_features)
    dataset.save_graph(args.savepath)
