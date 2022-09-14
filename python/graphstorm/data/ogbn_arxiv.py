from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import csv
import dgl
import torch as th
import boto3
import os

from .dataset import M5gnnDataset

class OGBArxivTextFeatDataset(M5gnnDataset):
    """r ogbn-arxiv dataset for node classification
    """
    def __init__(self, raw_dir, text_feat_type='title', retain_original_features=False,
                 force_reload=False, verbose=True, reverse_edge=True, self_loop=True, max_text_length=-1):
        self._name = 'ogbn-arxiv'
        self._url = None
        self._raw_dir=raw_dir
        self.reverse_edge=reverse_edge
        self.self_loop=self_loop
        self.max_text_length=max_text_length
        self.text_feat_type=text_feat_type
        self.retain_original_features = retain_original_features

        super(OGBArxivTextFeatDataset, self).__init__(self._name,
                                                      url=self._url,
                                                      raw_dir=raw_dir,
                                                      force_reload=force_reload,
                                                      verbose=verbose)

    def download(self):
        if not os.path.exists(self._raw_dir + self._name + "/titleabs.tsv") or not os.path.exists(self._raw_dir + self._name + "/nodeidx2paperid.csv"):
            print("Downloading raw features")
            s3 = boto3.client('s3')
            if not os.path.exists((self._raw_dir + self._name + "/")):
                os.makedirs(self._raw_dir + self._name + "/")
            bucket_name="graphlytics-dataset"
            object_name="ogbn-arxiv-original-features/titleabs.tsv"
            file_name="titleabs.tsv"
            s3.download_file(bucket_name, object_name,file_name)
            os.rename(file_name, self._raw_dir + self._name +"/"+ file_name)
            s3 = boto3.client('s3')
            bucket_name="graphlytics-dataset"
            object_name="ogbn-arxiv-original-features/nodeidx2paperid.csv"
            file_name="nodeidx2paperid.csv"
            s3.download_file(bucket_name, object_name,file_name)
            os.rename(file_name, self._raw_dir + self._name + "/"+file_name)
        else:
            print("Raw features available locally constructing dgl graph")
        return self.process()

    def process(self):
        """ The ogbn-arxiv has 2 files the features in titleabs and the mapping in nodeidx2paperid the rest data are
         downloadable
        """
        if not os.path.exists(self._raw_dir + self._name + "/titleabs.tsv") or not os.path.exists(
                self._raw_dir + self._name + "/nodeidx2paperid.csv"):
            return self.download()
        data = DglNodePropPredDataset(name="ogbn-arxiv")
        evaluator = Evaluator(name="ogbn-arxiv")
        tsv_file = open(self._raw_dir + self._name + "/titleabs.tsv")
        datafeat = csv.reader((line.replace('\0', '') for line in tsv_file), delimiter="\t")
        next(datafeat)
        tsv_map_file = open(self._raw_dir + self._name + "/nodeidx2paperid.csv")
        map = csv.reader(tsv_map_file, delimiter=",")
        map_feats = {}
        next(map)
        # map feats dictionary contais the map from the text features to the node ids
        for row in map:
            map_feats[int(row[1])] = int(row[0])
        text_feats_dic = {}

        for row in datafeat:
            if len(row) == 3:
                if int(row[0]) in map_feats:
                    text_feats_dic[map_feats[int(row[0])]] = (row[1], row[2])

        if self.text_feat_type == "title":
            text_feats_list = [text_feats_dic[k][0] for k in range(len(text_feats_dic))]
        elif self.text_feat_type == "abstract":
            text_feats_list = [text_feats_dic[k][1] for k in range(len(text_feats_dic))]
        else:
            raise ValueError()

        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
        graph, labels = data[0]
        src, dst = graph.edges()
        # adding dummy node type since the returned seeds from the dataloader
        # will not have node type if only one is present
        # TODO need to fix in the feature to support single edge-type homo graphs
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

        # the text features have the same order as the original dgl.NID
        g.nodes['node'].data['text_idx'] = th.tensor(th.range(0,g.number_of_nodes("node")-1),dtype=int)
        text_feat = {'node':text_feats_list}

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
        print(g)
        self._g=g
        self._raw_text_feat = text_feat
        self._num_classes = data.num_classes

        if self.retain_original_features:
            print("Retaining original node features and discarding the text data. This is the original input of ogbn.")
            g.nodes['node'].data['feat'] = graph.ndata["feat"]
            del g.nodes['node'].data['text_idx']
            self._raw_text_feat = {}

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
        if self.text_feat_type=="title":
            if not os.path.exists(self._raw_dir + self._name + "/ogbn-arxiv_BERT_embeddings_title.th"):
                print("Downloading bert embeddings")
                s3 = boto3.client('s3')
                if not os.path.exists((self._raw_dir + self._name +"/")):
                    os.makedirs(self._raw_dir + self._name +"/")
                bucket_name="graphlytics-dataset"
                object_name="ogbn-arxiv-bert-embeddings/ogbn-arxiv_BERT_embeddings_title.th"
                file_name="ogbn-arxiv_BERT_embeddings_title.th"
                try:
                    s3.download_file(bucket_name, object_name,file_name)
                    embs=th.load("ogbn-arxiv_BERT_embeddings_title.th")
                    os.rename(file_name, self._raw_dir + self._name + "/"+file_name)
                except:
                    embs = None
            else:
                print("Bert embeddings available locally")
                file_name="ogbn-arxiv_BERT_embeddings_title.th"
                embs = th.load(self._raw_dir + self._name + "/"+file_name)
            return embs
        else:
            if not os.path.exists(self._raw_dir + self._name + "/ogbn-arxiv_BERT_embeddings_abstract.th"):
                print("Downloading bert embeddings")
                s3 = boto3.client('s3')
                if not os.path.exists((self._raw_dir + self._name + "/")):
                    os.makedirs(self._raw_dir + self._name + "/")
                bucket_name = "graphlytics-dataset"
                object_name = "ogbn-arxiv-bert-embeddings/ogbn-arxiv_BERT_embeddings_abstract.th"
                file_name = "ogbn-arxiv_BERT_embeddings_abstract.th"
                try:
                    s3.download_file(bucket_name, object_name, file_name)
                    embs = th.load("ogbn-arxiv_BERT_embeddings_abstract.th")
                    os.rename(file_name, self._raw_dir + self._name + "/" + file_name)
                except:
                    embs = None
            else:
                print("Bert embeddings available locally")
                file_name = "ogbn-arxiv_BERT_embeddings_abstract.th"
                embs = th.load(self._raw_dir + self._name + "/" + file_name)
            return embs
