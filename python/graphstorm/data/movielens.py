"""Builtin MovieLens100k dataset
"""
import os
import csv
import json

import dgl
import torch as th

from .dataset import GSgnnDataset
from .utils import parse_category_single_feat
from .utils import get_id

class MovieLens100kNCDataset(GSgnnDataset):
    """r Movielens dataset for node classification
    """
    def __init__(self, raw_dir, edge_pct=1,
                 max_sequence_length=512, retain_original_features=True, user_text=False,
                 user_age_as_label=False, force_reload=False, verbose=True):
        """
        Parameters
        ----------
        raw_dir: str
            The file locations
        edge_pct: float
            percentage of edges in the test set
        max_sequence_length: int
            what is the maximum supported sequence length
        retain_original_features: bool
            whether we retain the original features.
        user_text: bool
            whether to use occupation as user text
        force_reload: bool
            whether to reload the raw dataset
        verbose: bool
            whether to print debug info
        """
        name = 'ml-100k'
        url = None
        self.max_sequence_length = max_sequence_length
        self.retain_original_features = retain_original_features
        self.user_text = user_text
        self.user_age_as_label = user_age_as_label
        self.target_etype = ('user', 'rating', 'movie')
        self.edge_pct = edge_pct
        if self.user_text:
            assert self.retain_original_features
        super(MovieLens100kNCDataset, self).__init__(name,
                                                     url=url,
                                                     raw_dir=raw_dir,
                                                     force_reload=force_reload,
                                                     verbose=verbose)

    def save_graph(self, path):
        """ Save processed graph into disk.

        Parameters
        ----------
        path : str
            Where to save the output
        """
        # save the processed data
        gname = self._name + '.bin'
        print("Save graph {} into {}".format(self._g, os.path.join(path, gname)))
        dgl.save_graphs(os.path.join(path, gname), [self._g])
        ginfo = self._name + '.json'
        with open(ginfo, 'w', encoding="utf-8") as f:
            json.dump({"num_class": self._num_classes},f)

    def load(self):
        # load from local storage
        root_path = self._raw_dir
        gname = self._name+'.bin'
        g, _ = dgl.load_graphs(os.path.join(root_path, gname))
        print(g[0])
        self._g = g[0]
        ginfo = self._name + '.json'
        with open(ginfo, 'r', encoding="utf-8") as f:
            info = json.load(f)
            self._num_classes = info["num_class"]

    def has_cache(self):
        root_path = self._raw_dir
        gname = self._name+'.bin'
        return os.path.exists(os.path.join(root_path, gname))

    def process(self):
        """ The movielens data has has 4 files
            u.user: user feature, id|age|gender|occupation|zipcode
            u.item: item feature, id|title|year|_url|unknown|Action|Adventure|Animation|Children|
                                  Comedy|Crime|Documentary|Drama|Fantasy|Film-Noir|Horror|Musical|
                                  Mystery|Romance|Sci-Fi|Thriller|War|Western
            u.base: user rate item edges
            u.test: user rate item edges
        """
        root_path = os.path.join(self._raw_dir, self._name)
        user_file = os.path.join(root_path, 'u.user')
        item_file = os.path.join(root_path, 'u.item')
        text_feat = {}
        separator = '|'
        with open(user_file, newline='', encoding="ISO-8859-1") as csvfile:
            reader = csv.reader(csvfile, delimiter=separator)

            user_ids = []
            age = []
            gender = []
            occupation = []
            for line in reader:
                user_ids.append(line[0])
                age.append(int(line[1]))
                gender.append(line[2])
                occupation.append(line[3])

            # encode user id
            unids = []
            unid_map = {}
            for node in user_ids:
                unid, _ = get_id(unid_map, node)
                unids.append(unid)
            unids = th.tensor(unids, dtype=th.int64)

            age = th.tensor(age, dtype=th.float32)
            if self.retain_original_features:
                # encode age
                if not self.user_age_as_label:
                    min_age = th.min(age)
                    max_age = th.max(age)
                    age = (age - min_age) / (max_age - min_age)
                    age = age.unsqueeze(dim=1)

                # encode gender
                gender, _ = parse_category_single_feat(gender)
                gender = th.tensor(gender, dtype=th.float32)

                # encode occupation
                if self.user_text is False:
                    occupation, _ = parse_category_single_feat(occupation)
                    occupation = th.tensor(occupation, dtype=th.float32)
                    user_feat = th.cat((age, gender, occupation), dim=1) \
                            if not self.user_age_as_label else th.cat((gender, occupation), dim=1)
                else:
                    if not self.user_age_as_label:
                        user_feat = th.cat((age, gender), dim=1)
                    else:
                        user_feat = gender
                    text_feat['user'] = occupation

        with open(item_file, newline='', encoding="ISO-8859-1") as csvfile:
            reader = csv.reader(csvfile, delimiter=separator)

            movie_ids = []
            title = []
            year = []
            movie_labels = []
            for line in reader:
                movie_ids.append(line[0])
                if line[1] == 'unknown':
                    title.append('unknown')
                    year.append(0)
                else:
                    title.append(line[1][:-7])
                    year.append(int(line[2][-4:]))
                movie_labels.append([int(l) for l in line[5:]])

            # encode user id
            inids = []
            inid_map = {}
            for node in movie_ids:
                inid, _ = get_id(inid_map, node)
                inids.append(inid)
            inids = th.tensor(inids, dtype=th.int64)

            # title feature
            text_feat['movie'] = title

            if self.retain_original_features:
                # encode year
                year = th.tensor(year, dtype=th.float32)
                min_year = th.min(year)
                max_year = th.max(year)
                year = (year - min_year) / (max_year - min_year)
                year = year.unsqueeze(dim=1)

            self._num_classes = len(movie_labels[0])
            # In movielens, a movie can have multiple genre tags.
            # To simplify the unitest, we only use the first existing tag as the
            # training lable.
            # TODO(xiangsx): Change it to multi-label multi-class classification
            # dataset when needed.
            new_labels = []
            for label in movie_labels:
                first_label_idx = 0
                for i, lbl in enumerate(label):
                    if lbl == 1:
                        first_label_idx = i
                        break
                new_labels.append(first_label_idx)
            movie_labels = new_labels
            # encode movie labels
            movie_labels = th.tensor(movie_labels, dtype=th.long)

        # load graph structure
        base_file = os.path.join(root_path, 'u1.base')
        test_file = os.path.join(root_path, 'u1.test')
        heads = {}
        tails = {}
        edge_data = {}
        with open(base_file, newline='', encoding="ISO-8859-1") as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')

            for line in reader:
                rel = ('user', 'rating', 'movie')
                if rel not in heads:
                    heads[rel] = []
                    tails[rel] = []
                    edge_data[rel] = []
                heads[rel].append(unid_map[line[0]])
                tails[rel].append(inid_map[line[1]])
                edge_data[rel].append(int(line[2]))

        with open(test_file, newline='', encoding="ISO-8859-1") as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')

            for line in reader:
                rel = ('user', 'rating', 'movie')
                heads[rel].append(unid_map[line[0]])
                tails[rel].append(inid_map[line[1]])
                edge_data[rel].append(int(line[2]))

        graph_edges = {}
        node_dicts = {
            'user': unids.shape[0],
            'movie': inids.shape[0]
        }
        for key, _ in heads.items():
            graph_edges[key] = (heads[key], tails[key])
            graph_edges[(key[2], key[1] + '-rev', key[0])] = (tails[key], heads[key])

        g = dgl.heterograph(graph_edges, num_nodes_dict=node_dicts)
        if self.retain_original_features:
            g.nodes['user'].data['feat'] = user_feat
            g.nodes['movie'].data['feat'] = year
        if self.user_text:
            g.nodes['user'].data['text_idx'] = unids
        g.nodes['movie'].data['text_idx'] = inids
        g.nodes['movie'].data['genre'] = movie_labels
        g.edges['rating'].data['rate'] = th.tensor(edge_data[('user', 'rating', 'movie')],
                                                   dtype=th.int32)

        # split labels
        th.manual_seed(42)
        idx = th.randperm(inids.shape[0])
        # 0.7 for train
        num_train = int(inids.shape[0] * 0.7)
        train_mask = th.full(inids.shape, False, dtype=th.bool)
        train_idx = idx[0:num_train]
        train_mask[train_idx] = True
        # 0.1 for val
        num_val = int(inids.shape[0] * 0.1)
        valid_mask = th.full(inids.shape, False, dtype=th.bool)
        valid_idx = idx[num_train:num_train+num_val]
        valid_mask[valid_idx] = True
        # 0.2 for test
        test_mask = th.full(inids.shape, False, dtype=th.bool)
        test_idx = idx[num_train+num_val:]
        test_mask[test_idx] = True
        g.nodes['movie'].data['train_mask'] = train_mask
        g.nodes['movie'].data['val_mask'] = valid_mask
        g.nodes['movie'].data['test_mask'] = test_mask

        if self.user_age_as_label:
            g.nodes['user'].data['age'] = age
            idx = th.randperm(unids.shape[0])
            # 0.7 for train
            num_train = int(unids.shape[0] * 0.7)
            train_mask = th.full(unids.shape, False, dtype=th.bool)
            train_idx = idx[0:num_train]
            train_mask[train_idx] = True
            # 0.1 for val
            num_val = int(unids.shape[0] * 0.1)
            valid_mask = th.full(unids.shape, False, dtype=th.bool)
            valid_idx = idx[num_train:num_train+num_val]
            valid_mask[valid_idx] = True
            # 0.2 for test
            test_mask = th.full(unids.shape, False, dtype=th.bool)
            test_idx = idx[num_train+num_val:]
            test_mask[test_idx] = True
            g.nodes['user'].data['train_mask'] = train_mask
            g.nodes['user'].data['val_mask'] = valid_mask
            g.nodes['user'].data['test_mask'] = test_mask

        # edge masks
        # edge_pct has to be between 0.2 and 1 since we will use by default 0.1 for validation
        # and 0.1 for testing as the smallest possible.
        assert self.edge_pct <= 1 and  self.edge_pct >= 0.2
        int_edges = g.number_of_edges("rating")
        if self.edge_pct == 1:
            g.edges["rating"].data['train_mask'] = th.full((int_edges,), True, dtype=th.bool)
            g.edges["rating-rev"].data['train_mask'] = th.full((int_edges,), True, dtype=th.bool)
        else:
            # the validation pct is 0.1
            val_pct = 0.1
            train_pct = self.edge_pct - val_pct
            # the test is 1 - the rest
            g.edges["rating"].data['train_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges["rating-rev"].data['train_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges["rating"].data['val_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges["rating-rev"].data['val_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges["rating"].data['test_mask'] = th.full((int_edges,), False, dtype=th.bool)
            g.edges["rating-rev"].data['test_mask'] = th.full((int_edges,), False, dtype=th.bool)

            g.edges["rating"].data['train_mask'][: int(int_edges*train_pct)] = True
            g.edges["rating-rev"].data['train_mask'][: int(int_edges * train_pct)] = True
            g.edges["rating"].data['val_mask'][int(int_edges*train_pct):
                                               int(int_edges*self.edge_pct)] = True
            g.edges["rating-rev"].data['val_mask'][int(int_edges*train_pct):
                                                   int(int_edges*self.edge_pct)] = True
            g.edges["rating"].data['test_mask'][int(int_edges*self.edge_pct):] = True
            g.edges["rating-rev"].data['test_mask'][int(int_edges*self.edge_pct):] = True
        print(g)

        self._g = g
        self._raw_text_feat = text_feat

    def __getitem__(self, idx):
        r"""Gets the data object at index.
        """
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1

    @property
    def predict_category(self):
        """The node type to be predicted, which is the movie in the ML100k dataset
        """
        return 'movie'

    @property
    def num_classes(self):
        """The number of classess of labels
        """
        return self._num_classes
