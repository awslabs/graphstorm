import numpy as np
import torch as th
import dgl
import abc

from ..data.utils import return_reverse_mappings

def split_full_edge_list(g, etype, rank):
    ''' Split the full edge list of a graph.
    '''
    # We assume that the number of edges is larger than the number of processes.
    # This should always be true unless a user's training set is extremely small.
    assert g.num_edges(etype) >= th.distributed.get_world_size()
    start = g.num_edges(etype) // th.distributed.get_world_size() * rank
    end = g.num_edges(etype) // th.distributed.get_world_size() * (rank + 1)
    return th.arange(start, end)

class GSgnnLinkPredictionData():
    """ Data for link prediction tasks

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    pb: DGL partition book
        DGL partition book
    train_etypes : list
        A list of edge types who have training edges
    eval_etypes : list
        A list of edge types who have validation and testing edges
    """
    def __init__(self, g, pb, train_etypes, eval_etypes):
        self._train_etypes = train_etypes
        self._eval_etypes = eval_etypes

        self._train_idxs = None
        self._val_idxs = None
        self._test_idxs = None
        self._do_validation = False

        self.prepare_data(g, pb)

    @abc.abstractmethod
    def prepare_data(self, g, pb):
        """
        Prepare the dataset.

        Arguement
        ---------
        g: Dist DGLGraph
        pb: Partition book
        """

    @property
    def train_etypes(self):
        return self._train_etypes

    @property
    def eval_etypes(self):
        return self._eval_etypes

    @property
    def train_idxs(self):
        return self._train_idxs

    @property
    def val_idxs(self):
        return self._val_idxs

    @property
    def test_idxs(self):
        return self._test_idxs

    @property
    def do_validation(self):
        return self._do_validation

class GSgnnLinkPredictionTrainData(GSgnnLinkPredictionData):
    """ Link prediction training data

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    pb: DGL partition book
    train_etypes : list
        A list of edge types who have training edges
    eval_etypes : list
        A list of edge types who have validation and testing edges
    full_graph_training : boolean
        If set true, train_mask is applied to all edges. The setting is used for tasks requiring link prediction
        pre-training, when entire graph structure is used.
    """
    def __init__(self, g, pb, train_etypes, eval_etypes, full_graph_training=False):
        self.full_graph_training = full_graph_training
        super(GSgnnLinkPredictionTrainData, self).__init__(
            g, pb, train_etypes, eval_etypes)

    def prepare_data(self, g, pb):
        """
        Prepare the training, validation and testing edge set.

        It will setup the following class fields:
        self._train_idxs: the edge indices of the local training set.
        self._val_idxs: the edge indices of the local validation set, can be empty.
        self._test_idxs: the edge indices of the local test set, can be empty.
        self._do_validation: if masks are empty, do_validation is set to False.

        Arguement
        ---------
        g: Dist DGLGraph
        pb: Partition book
        """
        train_idxs = {}
        # these are used as the target edge types for link prediction scoring function
        for canonical_train_etype in self.train_etypes:
            train_etype = canonical_train_etype[1]
            if 'train_mask' in g.edges[train_etype].data and not self.full_graph_training:
                train_idx = dgl.distributed.edge_split(g.edges[train_etype].data['train_mask'],
                                                    pb, etype=train_etype, force_even=True)
            else:
                # If there are no training masks, we assume all edges can be used for training.
                # Therefore, we use a more memory efficient way to split the edge list.
                # TODO(zhengda) we need to split the edges properly to increase the data locality.
                train_idx = split_full_edge_list(g, train_etype, g.rank())

            # We only use relation type. This is the limitation of distributed DGL
            train_idxs[train_etype] = train_idx

        eval_etypes = self.eval_etypes if self.eval_etypes is not None else self.train_etypes

        val_idxs = {}
        test_idxs = {}
        do_validation = False
        # collect validation set and test set if any
        for canonical_eval_etype in eval_etypes:
            eval_etype = canonical_eval_etype[1]
            # user must provide validation mask
            # TODO(xiangsx): do not use canonical_eval_etype[1], use canonical_eval_etype
            if 'val_mask' in g.edges[eval_etype].data:
                val_idxs[eval_etype] = dgl.distributed.edge_split(
                    g.edges[eval_etype].data['val_mask'], pb, etype=eval_etype, force_even=True)

            if 'test_mask' in g.edges[eval_etype].data:
                test_idxs[eval_etype] = dgl.distributed.edge_split(
                    g.edges[eval_etype].data['test_mask'], pb, etype=eval_etype, force_even=True)

        if (len(test_idxs) == 0 or len(val_idxs) == 0):
            print('part {}, train: {}'.format(g.rank(), np.sum([len(train_idxs[etype]) for etype in train_idxs])))
        else:
            do_validation = True
            print('part {}, train: {}, val: {}, test: {}'.format(g.rank(),
                                                            np.sum([len(train_idxs[etype]) for etype in train_idxs]),
                                                            np.sum([len(val_idxs[etype]) for etype in val_idxs]),
                                                            np.sum([len(test_idxs[etype]) for etype in test_idxs])))

        self._train_idxs = train_idxs
        self._val_idxs = val_idxs
        self._test_idxs = test_idxs
        self._do_validation = do_validation

class GSgnnLinkPredictionInferData(GSgnnLinkPredictionData):
    """ Link prediction training data

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    pb: DGL partition book
    eval_etypes : list
        A list of edge types who have validation and testing edges
    """
    def __init__(self, g, pb, eval_etypes):
        super(GSgnnLinkPredictionInferData, self).__init__(
            g, pb, None, eval_etypes)

    def prepare_data(self, g, pb):
        """
        Prepare the testing edge set if any

        It will setup self._test_idxs, the edge indices of the local test set.
        The test_idxs can be empty.

        Arguement
        ---------
        g: Dist DGLGraph
        pb: Partition book
        """

        eval_etypes = self.eval_etypes if len(self.eval_etypes) > 0 else self.train_etypes

        test_idxs = {}
        do_validation = False
        # collect validation set and test set if any
        for canonical_eval_etype in eval_etypes:
            eval_etype = canonical_eval_etype[1]
            # user must provide validation mask
            # TODO(xiangsx): do not use canonical_eval_etype[1], use canonical_eval_etype
            if 'test_mask' in g.edges[eval_etype].data:
                test_idxs[eval_etype] = dgl.distributed.edge_split(
                    g.edges[eval_etype].data['test_mask'], pb, etype=eval_etype, force_even=True)

        if (len(test_idxs) == 0):
            print("No test set. Do not need to do evaluation")
        else:
            do_validation = True
            print(f"part {g.rank()}, " \
                  f"test: {np.sum([len(test_idxs[etype]) for etype in test_idxs])}")

        self._test_idxs = test_idxs
        self._do_validation = do_validation

class GSgnnEdgePredictionTrainData():
    """ Edge prediction training data

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    pb: DGL partition book
    train_etypes : list
        A list of edge types who have training edges
    label_field : str
        The field for storing labels
    """
    def __init__(self, g, pb, target_etypes, label_field):
        # TODO not well optimized code for distributed setting
        assert len(target_etypes) == 1, "Not supported more than one edge types as targets"

        self._train_etypes = target_etypes
        self._eval_etypes = target_etypes
        self._label_field = label_field
        self._prepare_train_data(g, pb)

    def _prepare_train_data(self, g, pb):
        """
        Prepare the training, validation and testing edge set.

        It will setup the following class fields:
        self._train_idxs: the edge indices of the local training set.
        self._val_idxs: the edge indices of the local validation set, can be empty.
        self._test_idxs: the edge indices of the local test set, can be empty.
        self._do_validation: if masks are empty, do_validation is set to False.

        Arguement
        ---------
        g: Dist DGLGraph
        pb: Partition book
        """
        train_idxs = {}
        # these are used as the target edge types for link prediction scoring function
        for canonical_train_etype in self.train_etypes:
            if g.rank() == 0:
                print('Loading training idx for etype : {}'.format((canonical_train_etype)))
            train_etype = canonical_train_etype[1]
            if 'train_mask' in g.edges[train_etype].data:
                train_idx = dgl.distributed.edge_split(g.edges[train_etype].data['train_mask'],
                                                    pb, etype=train_etype, force_even=True)
            else:
                # If there are no training masks, we assume all edges can be used for training.
                # Therefore, we use a more memory efficient way to split the edge list.
                # TODO(zhengda) we need to split the edges properly to increase the data locality.
                train_idx = split_full_edge_list(g, train_etype, g.rank())

            # We only use relation type. This is the limitation of distributed DGL
            train_idxs[train_etype] = train_idx

        eval_etypes = self.eval_etypes if len(self.eval_etypes) > 0 else self.train_etypes

        val_idxs = {}
        test_idxs = {}
        do_validation = False
        # collect validation set and test set if any
        for canonical_eval_etype in eval_etypes:
            eval_etype = canonical_eval_etype[1]
            # user must provide validation mask
            if 'val_mask' in g.edges[eval_etype].data:
                val_idx = dgl.distributed.edge_split(g.edges[eval_etype].data['val_mask'],
                                                    pb, etype=eval_etype, force_even=True)
                val_idxs[eval_etype] = val_idx
            if 'test_mask' in g.edges[eval_etype].data:
                test_idx = dgl.distributed.edge_split(g.edges[eval_etype].data['test_mask'],
                                                    pb, etype=eval_etype, force_even=True)
                test_idxs[eval_etype] = test_idx

        if (len(test_idxs) == 0 or len(val_idxs) == 0):
            print('part {}, train: {}'.format(g.rank(), np.sum([len(train_idxs[etype]) for etype in train_idxs])))
        else:
            do_validation = True
            print('part {}, train: {}, val: {}, test: {}'.format(g.rank(),
                                                            np.sum([len(train_idxs[etype]) for etype in train_idxs]),
                                                            np.sum([len(val_idxs[etype]) for etype in val_idxs]),
                                                            np.sum([len(test_idxs[etype]) for etype in test_idxs])))

        # TODO need to extend the rest of the code to accept multiple etypes
        train_etype = self.train_etypes[0][1]
        # Pass a reference to the DistTensor object
        labels = g.edges[train_etype].data[self._label_field]

        self._labels = labels
        self._train_idxs = train_idxs
        self._val_idxs = val_idxs
        self._test_idxs = test_idxs
        self._do_validation = do_validation

        # Next generates the node ids needed for evaluation purposes to speed up the evaluation process.
        # since there are no negative pairs this procedure can be optimized compared to link prediction

        val_src_dst_pairs = {}
        test_src_dst_pairs = {}
        val_test_nodes = {}
        # Collect all the nodes to be valuated
        for etype in val_idxs.keys():
            (src_type, e_type, dest_type) = g.to_canonical_etype(etype=etype)
            val_src_dst_pairs[etype] = g.find_edges(val_idxs[etype], etype=etype)
            if src_type not in val_test_nodes:
                val_test_nodes[src_type] = val_src_dst_pairs[etype][0]
            else:
                val_test_nodes[src_type] = th.cat([val_test_nodes[src_type], val_src_dst_pairs[etype][0]])
            if dest_type not in val_test_nodes:
                val_test_nodes[dest_type] = val_src_dst_pairs[etype][1]
            else:
                val_test_nodes[dest_type] = th.cat([val_test_nodes[dest_type], val_src_dst_pairs[etype][1]])

        for etype in test_idxs.keys():
            (src_type, e_type, dest_type) = g.to_canonical_etype(etype=etype)
            test_src_dst_pairs[etype] = g.find_edges(test_idxs[etype], etype=etype)
            if src_type not in val_test_nodes:
                val_test_nodes[src_type] = test_src_dst_pairs[etype][0]
            else:
                val_test_nodes[src_type] = th.cat([val_test_nodes[src_type], test_src_dst_pairs[etype][0]])
            if dest_type not in val_test_nodes:
                val_test_nodes[dest_type] = test_src_dst_pairs[etype][1]
            else:
                val_test_nodes[dest_type] = th.cat([val_test_nodes[dest_type], test_src_dst_pairs[etype][1]])

        # Need to keep only the unique node ids to generate embeddings
        for ntype in val_test_nodes:
            val_test_nodes[ntype] = th.unique(val_test_nodes[ntype])

        self._val_test_nodes = val_test_nodes

        for ntype in val_test_nodes:
            print('Number of validation and testing nodes for ntype {} required for edge classification inference'
                  ': {} at trainer {}'.format(ntype, len(val_test_nodes[ntype]), g.rank()))
        self._val_src_dst_pairs = val_src_dst_pairs
        self._test_src_dst_pairs = test_src_dst_pairs

    @property
    def val_test_nodes(self):
        return self._val_test_nodes

    @property
    def val_src_dst_pairs(self):
        return self._val_src_dst_pairs

    @property
    def test_src_dst_pairs(self):
        return self._test_src_dst_pairs

    @property
    def train_etypes(self):
        return self._train_etypes

    @property
    def eval_etypes(self):
        return self._eval_etypes

    @property
    def train_idxs(self):
        return self._train_idxs

    @property
    def val_idxs(self):
        return self._val_idxs

    @property
    def test_idxs(self):
        return self._test_idxs

    @property
    def labels(self):
        return self._labels

    @property
    def do_validation(self):
        return self._do_validation

class GSgnnEdgePredictionInferData():
    """ Edge prediction inference data

    Parameters
    ----------
    g: DGLGraph
        The graph used in inference
    pb: DGL partition book
    infer_etypes : list
        A list of edge types who have inference edges
    label_field : str
        The field for storing labels
    """
    def __init__(self, g, pb, infer_etypes, label_field):
        # TODO not well optimized code for distributed setting
        assert len(infer_etypes) == 1, "Not supported more than one edge types as targets"

        self._target_etypes = infer_etypes
        self._label_field = label_field
        self._prepare_eval_data(g, pb)

    def _prepare_eval_data(self, g, pb):
        """
        Prepare the testing node set if any

        It will setup self._test_idxs, the node indices of the local test set.
        The test_idxs can be empty.

        Arguement
        ---------
        g: Dist DGLGraph
        pb: Partition book
        """
        test_idxs = {}
        test_src_dst_pairs = {}
        do_validation = False
        labels = None

        target_etypes = self.target_etypes

        # TODO need to extend the rest of the code to accept multiple etypes
        assert len(target_etypes) == 1, \
            "Currently, we only support doing edge prediction on one edge type."
        canonical_target_etype = target_etypes[0]
        target_etype = canonical_target_etype[1]

        # test_mask exists
        if 'test_mask' in g.edges[target_etype].data:
            test_idx = dgl.distributed.edge_split(
                g.edges[target_etype].data['test_mask'],
                pb, etype=target_etype, force_even=True)
            test_idxs[target_etype] = test_idx
            print("test_mask exists")
        else:
            print("test mast not exist")


        if len(test_idxs) == 0:
            print("No need to do testing")
        else:
            do_validation = True
            print(f"part {g.rank()}, " \
                f"test: {np.sum([len(test_idxs[etype]) for etype in test_idxs])}")

            # Pass a reference to the DistTensor object
            labels = g.edges[target_etype].data[self._label_field]

            for etype, _ in test_idxs.items():
                test_src_dst_pairs[etype] = g.find_edges(test_idxs[etype], etype=etype)

        self._labels = labels
        self._test_idxs = test_idxs
        self._do_validation = do_validation

        target_ntypes = set()
        src_ntype, _, dst_ntype = canonical_target_etype
        target_ntypes.add(src_ntype)
        target_ntypes.add(dst_ntype)

        self._target_ntypes = target_ntypes
        self._test_src_dst_pairs = test_src_dst_pairs

    @property
    def target_etypes(self):
        return self._target_etypes

    @property
    def test_idxs(self):
        return self._test_idxs

    @property
    def test_src_dst_pairs(self):
        return self._test_src_dst_pairs

    @property
    def labels(self):
        return self._labels

    @property
    def do_validation(self):
        return self._do_validation

    @property
    def target_ntypes(self):
        return self._target_ntypes

#### Node classification/regression Task Data ####
class GSgnnNodeData():
    """ Data for node tasks

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    pb: DGL partition book
    predict_ntype : str
        Target node type
    label_field : str
        The field for storing labels
    """
    def __init__(self, g, pb, predict_ntype, label_field):
        self._predict_ntype = predict_ntype
        self._label_field = label_field
        self._train_idx = None
        self._val_idx = None
        self._test_idx = None
        self._labels = None
        self._do_validation = False

        self.prepare_data(g, pb)

    @abc.abstractmethod
    def prepare_data(self, g, pb):
        """
        Prepare the dataset.

        Arguement
        ---------
        g: Dist DGLGraph
        pb: Partition book
        """

    @property
    def predict_ntype(self):
        return self._predict_ntype

    @property
    def train_idx(self):
        return self._train_idx

    @property
    def val_idx(self):
        return self._val_idx

    @property
    def test_idx(self):
        return self._test_idx

    @property
    def labels(self):
        return self._labels

    @property
    def do_validation(self):
        return self._do_validation

class GSgnnNodeTrainData(GSgnnNodeData):
    """ Training data for node tasks

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    pb: DGL partition book
    predict_ntype : str
        Target node type
    label_field : str
        The field for storing labels
    """
    def __init__(self, g, pb, predict_ntype, label_field):
        super(GSgnnNodeTrainData, self).__init__(
            g, pb, predict_ntype, label_field)

    def prepare_data(self, g, pb):
        assert 'train_mask' in g.nodes[self.predict_ntype].data, \
            "For training dataset, train_mask must be provided."

        if 'trainer_id' in g.nodes[self.predict_ntype].data:
            print('over-partitioning')
            node_trainer_ids = g.nodes[self.predict_ntype].data['trainer_id']
            train_idx = dgl.distributed.node_split(g.nodes[self.predict_ntype].data['train_mask'],
                                                pb, ntype=self.predict_ntype, force_even=True,
                                                node_trainer_ids=node_trainer_ids)
        else:
            print('random-partitioning')
            node_trainer_ids = None
            train_idx = dgl.distributed.node_split(g.nodes[self.predict_ntype].data['train_mask'],
                                                pb, ntype=self.predict_ntype, force_even=True)
        val_idx = None
        test_idx = None
        if 'val_mask' in g.nodes[self.predict_ntype].data:
            val_idx = dgl.distributed.node_split(
                g.nodes[self.predict_ntype].data['val_mask'],
                pb, ntype=self.predict_ntype, force_even=True,
                node_trainer_ids=node_trainer_ids)
        if 'test_mask' in g.nodes[self.predict_ntype].data:
            test_idx = dgl.distributed.node_split(
                g.nodes[self.predict_ntype].data['test_mask'],
                pb, ntype=self.predict_ntype, force_even=True,
                node_trainer_ids=node_trainer_ids)

        if val_idx is None and test_idx is None:
            do_validation = False
        else:
            do_validation = True

        local_nid = pb.partid2nids(pb.partid, self.predict_ntype).detach().numpy()
        print('part {}, train: {} (local: {}), val: {}, test: {}'.format(
            g.rank(), len(train_idx), len(np.intersect1d(train_idx.numpy(), local_nid)),
            len(val_idx) if val_idx is not None else 0, len(test_idx) if test_idx is not None else 0))
        labels = g.nodes[self.predict_ntype].data[self._label_field][np.arange(g.number_of_nodes(self.predict_ntype))]

        self._train_idx = train_idx
        self._val_idx = val_idx
        self._test_idx = test_idx
        self._labels = labels
        self._do_validation = do_validation

class GSgnnNodeInferData(GSgnnNodeData):
    """ Inference data for node tasks

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    pb: DGL partition book
    predict_ntype : str
        Target node type
    label_field : str
        The field for storing labels
    """
    def __init__(self, g, pb, predict_ntype, label_field):
        super(GSgnnNodeInferData, self).__init__(
            g, pb, predict_ntype, label_field)

    def prepare_data(self, g, pb):
        """
        Prepare the testing node set if any

        It will setup self._test_idxs, the node indices of the local test set.
        The test_idxs can be empty.

        Arguement
        ---------
        g: Dist DGLGraph
        pb: Partition book
        """
        test_idx = None
        if 'test_mask' in g.nodes[self.predict_ntype].data:
            node_trainer_ids = g.nodes[self.predict_ntype].data['trainer_id'] \
                if 'trainer_id' in g.nodes[self.predict_ntype].data \
                else None

            test_idx = dgl.distributed.node_split(
                g.nodes[self.predict_ntype].data['test_mask'],
                pb, ntype=self.predict_ntype, force_even=True,
                node_trainer_ids=node_trainer_ids)

        if (len(test_idx) == 0):
            print("No test set. Do not need to do evaluation")
        else:
            do_validation = True

        print(f"part {g.rank()}, "\
              f"test: {len(test_idx) if test_idx is not None else 0}")

        labels = g.nodes[self.predict_ntype].data[self._label_field][np.arange(g.number_of_nodes(self.predict_ntype))]

        self._test_idx = test_idx
        self._labels = labels
        self._do_validation = do_validation

################ GNN MLM dataset #######################

class GSgnnMLMTrainData():
    """ Training data for Masked-Language Modeling finetuning

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    pb: DGL partition book
    tune_ntype : str
        Target node type
    """
    def __init__(self, g, pb, tune_ntype):
        self._tune_ntype = tune_ntype

        self._prepare_train_data(g, pb)

    def _prepare_train_data(self, g, pb):
        num_nodes = g.num_nodes(self.tune_ntype)
        mlm_mask = dgl.distributed.DistTensor((num_nodes,),
                                              name="mlm_mask",
                                              dtype=th.int8,
                                              part_policy=g.get_node_partition_policy(self.tune_ntype),
                                              persistent=True)
        g.barrier()

        # split nodes in tune_ntype into train and valid sets
        # 90% for train and 10% for validation
        if g.rank() == 0:
            num_train = int(num_nodes * 0.9)
            train_mask = th.full((num_nodes,), False, dtype=th.bool)
            rand_idx = th.randperm(num_nodes)
            train_idx = rand_idx[:num_train]
            train_mask[train_idx] = True
            mlm_mask[th.arange(num_nodes)] = train_mask.type(th.int8)

        g.barrier()
        train_idx = dgl.distributed.node_split(mlm_mask,
                                               pb, ntype=self.tune_ntype, force_even=True)

        val_idx = th.nonzero(1 - mlm_mask[th.arange(num_nodes)]).squeeze()
        print(val_idx)
        do_validation = True

        local_nid = pb.partid2nids(pb.partid, self.tune_ntype).detach().numpy()
        print('part {}, train: {} (local: {}), val: {}'.format(
            g.rank(), len(train_idx), len(np.intersect1d(train_idx.numpy(), local_nid)),
            len(val_idx) if val_idx is not None else 0))

        self._train_idx = train_idx
        self._val_idx = val_idx
        self._do_validation = do_validation

    @property
    def tune_ntype(self):
        return self._tune_ntype

    @property
    def train_idx(self):
        return self._train_idx

    @property
    def val_idx(self):
        return self._val_idx

    @property
    def do_validation(self):
        return self._do_validation

