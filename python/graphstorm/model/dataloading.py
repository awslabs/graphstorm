import math

import dgl
import numpy as np
import torch as th
from .sampler import LocalUniform, JointUniform
from ..data.utils import return_reverse_mappings
from .utils import trim_data, modify_fanout_for_target_etype
from dgl.dataloading import DistDataLoader
from dgl.dataloading import EdgeCollator
from dgl.dataloading.dist_dataloader import _remove_kwargs_dist

####################### GNN dataset ############################

def split_full_edge_list(g, etype, rank):
    ''' Split the full edge list of a graph.
    '''
    # We assume that the number of edges is larger than the number of processes.
    # This should always be true unless a user's training set is extremely small.
    assert g.num_edges(etype) >= th.distributed.get_world_size()
    start = g.num_edges(etype) // th.distributed.get_world_size() * rank
    end = g.num_edges(etype) // th.distributed.get_world_size() * (rank + 1)
    return th.arange(start, end)

class GSgnnLinkPredictionTrainData():
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
        self._train_etypes = train_etypes
        self._eval_etypes = eval_etypes
        self.full_graph_training = full_graph_training

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

class GSgnnLinkPredictionInferData():
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
        self._eval_etypes = eval_etypes

        self._prepare_eval_data(g, pb)

    def _prepare_eval_data(self, g, pb):
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

    @property
    def eval_etypes(self):
        return self._eval_etypes

    @property
    def train_idxs(self):
        return None

    @property
    def val_idxs(self):
        return None

    @property
    def test_idxs(self):
        return self._test_idxs

    @property
    def do_validation(self):
        return self._do_validation

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

class GSgnnNodeTrainData():
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
        self._predict_ntype = predict_ntype
        self._label_field = label_field

        self._prepare_train_data(g, pb)

    def _prepare_train_data(self, g, pb):
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

class GSgnnNodeInferData():
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
        self._predict_ntype = predict_ntype
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

    @property
    def predict_ntype(self):
        return self._predict_ntype

    @property
    def train_idx(self):
        return None

    @property
    def val_idx(self):
        return None

    @property
    def test_idx(self):
        return self._test_idx

    @property
    def labels(self):
        return self._labels

    @property
    def do_validation(self):
        return self._do_validation

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

################ Minibatch DataLoader (Edge Prediction) #######################

class GSgnnEdgePredictionDataLoader():
    """ Edge prediction minibatch dataloader
        The dataloader we use here will
    Argument
    --------
    g: DGLGraph
    train_dataset: GSgnnEdgePredictionDataLoader
        The dataset used in training. It must includes train_idxs.
    fanout: neighbor sample fanout
    n_layers: number of GNN layers
    batch_size: minibatch size
    reverse_edge_types_map: A map for reverse edge type
    exclude_training_targets: Whether to exclude training edges during neighbor sampling
    remove_target_edge: This boolean controls whether we will exclude all edges of the target during message passing.
    device : the device for which the sampling will be performed.
    """
    def __init__(self, g, train_dataset, fanout, n_layers, batch_size,
                 reverse_edge_types_map, remove_target_edge=True, exclude_training_targets=False, device='cpu'):
        assert len(fanout) == n_layers
        # set up dictionary fanout
        # remove the target edge type from computational graph

        target_etypes = train_dataset.train_etypes
        for e in target_etypes:
            if e in reverse_edge_types_map and reverse_edge_types_map[e] not in target_etypes:
                target_etypes.append(reverse_edge_types_map[e])
        if g.rank() == 0:
            if remove_target_edge:
                print("Target edge will be removed from message passing graph")
            else:
                print("Target edge will not be removed from message passing graph")

        if remove_target_edge:
            edge_fanout_lis = modify_fanout_for_target_etype(g=g, fanout=fanout, target_etypes=target_etypes)
        else:
            edge_fanout_lis = fanout

        self.dataloader = self._prepare_train_dataloader(g,
                                                         train_dataset.train_idxs,
                                                         edge_fanout_lis,
                                                         batch_size,
                                                         exclude_training_targets=exclude_training_targets,
                                                         reverse_edge_types_map=reverse_edge_types_map,
                                                         device='cpu')

    def _prepare_train_dataloader(self, g, train_idxs, fanout, batch_size, device='cpu',
                                  exclude_training_targets=False, reverse_edge_types_map=None):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)

        # edge loader
        loader = dgl.dataloading.DistEdgeDataLoader(g,
                                                    train_idxs,
                                                    sampler,
                                                    batch_size=batch_size,
                                                    device=device,
                                                    shuffle=True,
                                                    drop_last=False,
                                                    num_workers=0,
                                                    exclude='reverse_types' if exclude_training_targets else None,
                                                    reverse_etypes=reverse_edge_types_map
                                                    if exclude_training_targets else None)
        return loader

    def __iter__(self):
        return self.dataloader.__iter__()

    def __next__(self):
        return self.dataloader.__next__()


################ Minibatch DataLoader (Link Prediction) #######################

BUILTIN_LP_UNIFORM_NEG_SAMPLER = 'uniform'
BUILTIN_LP_JOINT_NEG_SAMPLER = 'joint'
BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER = 'localuniform'
BUILTIN_LP_ALL_ETYPE_UNIFORM_NEG_SAMPLER = 'all_etype_uniform'
BUILTIN_LP_ALL_ETYPE_JOINT_NEG_SAMPLER = 'all_etype_joint'

class GSgnnLinkPredictionDataLoader():
    """ Link prediction minibatch dataloader

    The negative edges are sampled uniformly.

    Argument
    --------
    g: DGLGraph
    train_dataset: GSgnnLinkPredictionTrainData
        The dataset used in training. It must includes train_idxs.
    fanout: neighbor sample fanout
    n_layers: number of GNN layers
    batch_size: minibatch size
    num_negative_edges: num of negative edges per positive edge
    device : the device trainer is running on.
    exclude_training_targets: Whether to exclude training edges during neighbor sampling
    reverse_edge_types_map: A map for reverse edge type
    """
    def __init__(self, g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device,
        exclude_training_targets=False, reverse_edge_types_map=None):
        assert len(fanout) == n_layers
        self.dataloader = self._prepare_train_dataloader(g,
                                                         train_dataset.train_idxs,
                                                         fanout,
                                                         num_negative_edges,
                                                         batch_size,
                                                         device,
                                                         exclude_training_targets,
                                                         reverse_edge_types_map)

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = dgl.dataloading.negative_sampler.Uniform(num_negative_edges)
        return negative_sampler

    def _prepare_train_dataloader(self, g, train_idxs, fanout, num_negative_edges, batch_size, device,
                                  exclude_training_targets=False, reverse_edge_types_map=None):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        negative_sampler = self._prepare_negative_sampler(num_negative_edges)

        # edge loader
        if isinstance(train_idxs, dict):
            for etype in train_idxs:
                train_idxs[etype] = trim_data(train_idxs[etype], device)
        else:
            train_idxs = trim_data(train_idxs, device)
        # latest documentation https://docs.dgl.ai/generated/dgl.dataloading.as_edge_prediction_sampler.html?highlight=as_edge_prediction_sampler
        loader = dgl.dataloading.DistEdgeDataLoader(g,
                                                    train_idxs,
                                                    sampler,
                                                    batch_size=batch_size,
                                                    negative_sampler=negative_sampler,
                                                    device="cpu", # Only cpu is supported in DistGraph
                                                    shuffle=True,
                                                    drop_last=False,
                                                    num_workers=0,
                                                    exclude='reverse_types' if exclude_training_targets else None,
                                                    reverse_etypes=reverse_edge_types_map \
                                                        if exclude_training_targets else None)
        return loader

    def __iter__(self):
        return self.dataloader.__iter__()

    def __next__(self):
        return self.dataloader.__next__()

class GSgnnLPJointNegDataLoader(GSgnnLinkPredictionDataLoader):
    """ Link prediction dataloader with joint negative sampler

    """
    def __init__(self, g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device,
        exclude_training_targets=False, reverse_edge_types_map=None):
        super(GSgnnLPJointNegDataLoader, self).__init__(
            g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device, exclude_training_targets, reverse_edge_types_map)

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = JointUniform(num_negative_edges)
        return negative_sampler

class GSgnnLPLocalUniformNegDataLoader(GSgnnLinkPredictionDataLoader):
    """ Link prediction dataloader with local uniform negative sampler

    """
    def __init__(self, g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device,
        exclude_training_targets=False, reverse_edge_types_map=None):
        super(GSgnnLPLocalUniformNegDataLoader, self).__init__(
            g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device, exclude_training_targets, reverse_edge_types_map)

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = LocalUniform(num_negative_edges)
        return negative_sampler

######## Per etype sampler ########
import inspect

class AllEtypeDistEdgeDataLoader(DistDataLoader):
    """ Distributed edge data sampler that samples at least one
        edge for each edge type in a minibatch

        Parameters
        ----------
        g: DistGraph
            Input graph
        eids: dict
            Target edge ids
        graph_sampler:
            Graph neighbor sampler
        device: str:
            Device
        kwargs: list
            Other arguments
    """
    def __init__(self, g, eids, graph_sampler, device=None, **kwargs):
        collator_kwargs = {}
        dataloader_kwargs = {}
        _collator_arglist = inspect.getfullargspec(EdgeCollator).args
        for k, v in kwargs.items():
            if k in _collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v

        if device is None:
            # for the distributed case default to the CPU
            device = 'cpu'
        assert device == 'cpu', 'Only cpu is supported in the case of a DistGraph.'
        self.collator = EdgeCollator(g, eids, graph_sampler, **collator_kwargs)
        _remove_kwargs_dist(dataloader_kwargs)
        super().__init__(eids,
                         collate_fn=self.collator.collate,
                         **dataloader_kwargs)

        self._reinit_dataset()
        self.device = device

    def _reinit_dataset(self):
        """ Reinitialize the dataset

            Here we record the edge ids of different edge types separately.
            When doing sampling we will sample edges per edge type.
        """
        batch_size = self.batch_size
        data_idx = {}
        total_edges = 0
        for key, val in self.dataset.items():
            data_idx[key] = th.arange(0, len(val))
            total_edges += len(val)
        self.data_idx = data_idx

        max_expected_idxs = 0
        bs_per_type = {}
        for etype, idxs in self.data_idx.items():
            # compute the number of edges to be sampled for
            # each edge type in a minibatch.
            # If batch_size * num_edges / total_edges < 0, then set 1.
            #
            # Note: The resulting batch size of a mini batch may be larger
            #       than the batch size set by a user.
            bs = math.ceil(batch_size * len(idxs) / total_edges)
            bs_per_type[etype] = bs
            exp_idxs = len(idxs) // bs
            if not self.drop_last and len(idxs) % bs != 0:
                exp_idxs += 1

            # Get the maximum expected idx across all edge types.
            # The epoch size is decided by max_expected_idxs
            max_expected_idxs = max(max_expected_idxs, exp_idxs)
        self.bs_per_type = bs_per_type
        self.expected_idxs = max_expected_idxs

        self.current_pos = {etype:0 for etype, _ in self.data_idx.items()}

    def __iter__(self):
        if self.shuffle:
            self.data_idx = {etype: th.randperm(len(idxs)) \
                for etype, idxs in self.data_idx.items()}
        self.current_pos = {etype:0 for etype, _ in self.data_idx.items()}
        self.recv_idxs = 0
        self.num_pending = 0
        return self

    def _next_data_etype(self, etype):
        """ Get postive edges for the next iteration for a specific edge type

            Return a tensor of eids. If return None, we will randomly
            generate an eid for the etype if the dataloader does not
            reach the end of epoch.
        """
        if self.current_pos[etype] == len(self.dataset[etype]):
            return None

        end_pos = 0
        if self.current_pos[etype] + self.bs_per_type[etype] > len(self.dataset[etype]):
            if self.drop_last:
                return None
            else:
                end_pos = len(self.dataset[etype])
        else:
            end_pos = self.current_pos[etype] + self.bs_per_type[etype]
        idx = self.data_idx[etype][self.current_pos[etype]:end_pos].tolist()
        ret = self.dataset[etype][idx]

        # Sharing large number of tensors between processes will consume too many
        # file descriptors, so let's convert each tensor to scalar value beforehand.
        self.current_pos[etype] = end_pos

        return ret

    def _rand_gen(self, etype):
        """ Randomly select one edge for a specific edge type
        """
        return self.dataset[etype][th.randint(len(self.dataset[etype]), (1,))]

    def _next_data(self):
        """ Get postive edges for the next iteration
        """
        end = True
        ret = []
        for etype, _ in self.dataset.items():
            next_data = self._next_data_etype(etype)
            # Only if all etypes reach end of iter,
            # the current iter is done
            end = (next_data is None) & end
            ret.append((etype, next_data))

        if end:
            return None
        else:
            new_ret = []
            for i in range(len(ret)):
                # ret[i] is (etype, eids)
                if ret[i][1] is None:
                    # if eids is None, randomly generate one more data point
                    new_ret.append((ret[i][0], self._rand_gen(ret[i][0]).item()))
                else:
                    for data in ret[i][1]:
                        new_ret.append((ret[i][0], data.item()))
            return new_ret

class GSgnnAllEtypeLinkPredictionDataLoader(GSgnnLinkPredictionDataLoader):
    """ Link prediction minibatch dataloader. In each minibatch,
        at least one edge is sampled from each etype.

        Note: using this dataloader with a graph with massive etypes
        may cause memory issue, as the batch_size will be implicitly
        increased.

        The negative edges are sampled uniformly.

        Note: The resulting batch size of a mini batch may be larger
              than the batch size set by a user.

    Parameters
    ----------
    g: DGLGraph
        Input graph
    train_dataset: GSgnnLinkPredictionTrainData
        The dataset used in training. It must includes train_idxs.
    fanout: list of int
        Neighbor sample fanout
    n_layers: int
        Number of GNN layers
    batch_size: int
        Minibatch size
    num_negative_edges: int
        Num of negative edges per positive edge
    device : device
        The device trainer is running on.
    exclude_training_targets:
        Whether to exclude training edges during neighbor sampling
    reverse_edge_types_map:
        A map for reverse edge type
    """
    def __init__(self, g, train_dataset, fanout, n_layers,
        batch_size, num_negative_edges, device,
        exclude_training_targets=False, reverse_edge_types_map=None):
        super(GSgnnAllEtypeLinkPredictionDataLoader, self).__init__(
            g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device, exclude_training_targets, reverse_edge_types_map)

    def _prepare_train_dataloader(self, g, train_idxs, fanout, num_negative_edges, batch_size, device,
                                  exclude_training_targets=False, reverse_edge_types_map=None):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        negative_sampler = self._prepare_negative_sampler(num_negative_edges)

        # edge loader
        if isinstance(train_idxs, dict):
            for etype in train_idxs:
                train_idxs[etype] = trim_data(train_idxs[etype], device)
        else:
            train_idxs = trim_data(train_idxs, device)
        # latest documentation https://docs.dgl.ai/generated/dgl.dataloading.as_edge_prediction_sampler.html?highlight=as_edge_prediction_sampler
        loader = AllEtypeDistEdgeDataLoader(g,
                                            train_idxs,
                                            sampler,
                                            batch_size=batch_size,
                                            negative_sampler=negative_sampler,
                                            device="cpu", # Only cpu is supported in DistGraph
                                            shuffle=True,
                                            drop_last=False,
                                            num_workers=0,
                                            exclude='reverse_types' if exclude_training_targets else None,
                                            reverse_etypes=reverse_edge_types_map \
                                                if exclude_training_targets else None)
        return loader

    def __iter__(self):
        return self.dataloader.__iter__()

    def __next__(self):
        return self.dataloader.__next__()

class GSgnnAllEtypeLPJointNegDataLoader(GSgnnAllEtypeLinkPredictionDataLoader):
    """ Link prediction dataloader with joint negative sampler.
        In each minibatch, at least one edge is sampled from each etype.

    """
    def __init__(self, g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device,
        exclude_training_targets=False, reverse_edge_types_map=None):
        super(GSgnnAllEtypeLPJointNegDataLoader, self).__init__(
            g, train_dataset, fanout, n_layers, batch_size, num_negative_edges, device, exclude_training_targets, reverse_edge_types_map)

    def _prepare_negative_sampler(self, num_negative_edges):
        # the default negative sampler is uniform sampler
        negative_sampler = JointUniform(num_negative_edges)
        return negative_sampler

################ Minibatch DataLoader (Node classification) #######################

class GSgnnNodeDataLoader():
    """ Minibatch dataloader for node tasks

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    train_dataset: GSgnnNodeTrainData
        Training ataset
    fanout: list of int
        Neighbor sample fanout
    n_layers: int
        Number of GNN layers
    batch_size: int
        Batch size
    device: torch.device
        the device trainer is running on.
    """
    def __init__(self, g, train_dataset, fanout, n_layers, batch_size, device):
        assert len(fanout) == n_layers
        self.dataloader = self._prepare_train_dataloader(g,
                                                         train_dataset.predict_ntype,
                                                         train_dataset.train_idx,
                                                         fanout,
                                                         batch_size,
                                                         device)

    def _prepare_train_dataloader(self, g, predict_ntype, train_idx, fanout, batch_size, device):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        train_idx = trim_data(train_idx, device)
        loader = dgl.dataloading.DistNodeDataLoader(g, {predict_ntype: train_idx}, sampler,
            batch_size=batch_size, shuffle=True, num_workers=0)

        return loader

    def __iter__(self):
        return self.dataloader.__iter__()

    def __next__(self):
        return self.dataloader.__next__()
