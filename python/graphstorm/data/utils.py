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

    Utility functions for dataset and data processing
"""
import numpy as np
import torch as th
import torch.distributed as dist

from sklearn.preprocessing import LabelBinarizer

def get_id(nid_dict, key):
    """ Convert Raw node id into integer ids.

    Parameters
    ----------
    dict : dict
        Raw node id to integer id mapping.
    key : str
        Raw node id
    """
    nid = nid_dict.get(key, None)
    is_new = False
    if nid is None:
        is_new = True
        nid = len(nid_dict)
        nid_dict[key] = nid
    return nid, is_new

def reverse_etype(etype):
    """Add reversed edges for the given edge type.
       If the given edge type is a canonical type, use the first and second elements as edge
    """
    if isinstance(etype, tuple):
        return (etype[2], "rev-{}".format(etype[1]), etype[0])
    else:
        return "rev-{}".format(etype)


def return_reverse_mappings(etypes, g):
    """ Retrieve all edge types' IDs for reverse edge mapping.
    These mappings are needed by the edge dataloader to exclude the reverse edges in the
    computational graph. Here we assume that the reverse edge id has the same id as the
    original edge id. This is a result of the add_reverse_edges() function of the data
    utils files.
    """
    reverse_eids = {g.to_canonical_etype(etype): th.arange(g.number_of_edges(etype)) \
                    for etype in etypes}
    return reverse_eids

def add_reverse_edges(graph_edges):
    """Add reverse edge for all edge types.
    """
    new_graph_edges = {}
    for etype, edges in graph_edges.items():
        new_graph_edges[etype] = edges
        new_graph_edges[reverse_etype(etype)] = (edges[1], edges[0])
    return new_graph_edges

def adjust_eval_mapping_for_partition(original_idx, original_eval_idx, original_eval_text):
    '''
    This function corrects the mappings for the evaluation entity. It will search over the mapped
    idx where the original eval idx is contained and will returned the position and the
    corresponding text.

    Returns
    -------

    '''
    original_eval_idx = np.array(original_eval_idx)
    original_idx= original_idx[0:len(original_idx)]
    original_idx= original_idx.cpu().numpy()
    _, x_ind, y_ind =np.intersect1d(original_idx, original_eval_idx, \
                                    assume_unique=True, return_indices=True)
    # xy are the common and unique elements

    # x_ind is the index of the common entries in the original_idx
    # essentially this index is the node idx in the distgraph
    # to recover the corresponding embeddings for saving
    # original_node_idx_to_distributed_node_idx= {original_eval_idx : x_ind[i] \
    #                                             for i in range(len(x_ind))}

    # y_ind is the index of the common entries in the original_eval_idx
    # essentially this will give the position to return the corresponding
    # original_eval_text entries that correspond to the x_ind

    original_eval_text = np.array(original_eval_text)
    return th.tensor(x_ind), original_eval_text[y_ind].tolist()

def parse_category_single_feat(category_inputs, classes=None):
    # Add the 'r' before any docstring that includes math functions to avoid pylint errors.
    r""" Parse categorical features and convert it into onehot encoding.

    Each entity of category_inputs should only contain only one category.

    Parameters
    ----------
    category_inputs : list of str
        input categorical features
    norm: str, optional
        Which kind of normalization is applied to the features.
        Supported normalization ops include:

        (1) None, do nothing.
        (2) `col`, column-based normalization. Normalize the data
        for each column:

        .. math::
            x_{ij} = \frac{x_{ij}}{\sum_{i=0}^N{x_{ij}}}

        (3) `row`, sane as None
    classes : list
        predefined class list

    Note
    ----
    sklearn.preprocessing.LabelBinarizer is used to convert
    categorical features into a onehot encoding format.

    Return
    ------
    numpy.array
        The features in numpy array
    list
        Labels for each class

    Examples
    --------

    >>> inputs = ['A', 'B', 'C', 'A']
    >>> feats = parse_category_single_feat(inputs)
    >>> feats
        array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])

    """
    if classes is not None:
        lb = LabelBinarizer()
        lb.fit(classes)
        feat = lb.transform(category_inputs)
    else:
        lb = LabelBinarizer()
        feat = lb.fit_transform(category_inputs)

    # if there are only 2 catebories,
    # fit_transform only create a array of [0, 1, ...]
    if feat.shape[1] == 1:
        f = np.zeros((feat.shape[0], 2))
        f[range(f.shape[0]),feat.squeeze()] = 1.
        feat = f

    return feat, lb.classes_


def generated_train_valid_test_splits(g, train_pct, valid_pct, test_pct,
                                      use_non_selected_edges=False, seed=None):
    """Generate the train/validation/test splits
    """
    # Use seed to ensure consistent train split
    if seed is not None:
        th.manual_seed(seed)

    train_eids = {}
    val_eids = {}
    test_eids = {}
    train_graph_mask_dic = {}
    val_graph_mask_dic = {}
    test_graph_mask_dic = {}
    for etype in g.canonical_etypes:
        number_edges_etype = g.number_of_edges(etype)
        edge_ids = th.randperm(number_edges_etype)

        train_eids[etype] = edge_ids[:int(train_pct*number_edges_etype)]
        val_eids[etype] = edge_ids[int(train_pct*number_edges_etype):
                                   int((train_pct+valid_pct)*number_edges_etype)]
        test_eids[etype] = edge_ids[int((train_pct+valid_pct)*number_edges_etype):
                                    int((train_pct+valid_pct+test_pct)*number_edges_etype)]

        if use_non_selected_edges:
            train_graph_mask = th.ones(number_edges_etype)
            train_graph_mask[val_eids[etype]] = 0
            train_graph_mask[test_eids[etype]] = 0
            val_graph_mask = th.ones(number_edges_etype)
            val_graph_mask[test_eids[etype]] = 0
            test_graph_mask = th.ones(number_edges_etype)
        else:
            train_graph_mask = th.zeros(number_edges_etype)
            train_graph_mask[train_eids[etype]] = 1
            val_graph_mask = th.zeros(number_edges_etype)
            val_graph_mask[train_eids[etype]] = 1
            val_graph_mask[val_eids[etype]] = 1
            test_graph_mask = th.zeros(number_edges_etype)
            test_graph_mask[train_eids[etype]] = 1
            test_graph_mask[val_eids[etype]] = 1
            test_graph_mask[test_eids[etype]] = 1
        train_graph_mask_dic[etype]=train_graph_mask
        val_graph_mask_dic[etype] = val_graph_mask
        test_graph_mask_dic[etype] = test_graph_mask

        print('Edge type : {}: |train|={}, |val|={}, |test|={}'.format(str(etype),
                                                                       len(train_eids[etype]),
                                                                       len(val_eids[etype]),
                                                                       len(test_eids[etype])))

    return train_eids, val_eids, test_eids, \
           train_graph_mask_dic, val_graph_mask_dic, test_graph_mask_dic

def alltoall_cpu(rank, world_size, output_tensor_list, input_tensor_list):
    """ Each process scatters list of input tensors to all processes in a cluster
    and return gathered list of tensors in output list. The tensors should have the same shape.
    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire
    output_tensor_list : List of tensor
        The received tensors
    input_tensor_list : List of tensor
        The tensors to exchange
    """
    input_tensor_list = [tensor.to(th.device('cpu')) for tensor in input_tensor_list]
    for i in range(world_size):
        dist.scatter(output_tensor_list[i], input_tensor_list if i == rank else None, src=i)

def alltoallv_cpu(rank, world_size, output_tensor_list, input_tensor_list):
    """Each process scatters list of input tensors to all processes in a cluster
    and return gathered list of tensors in output list.

    Note: for gloo backend

    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire
    output_tensor_list : List of tensor
        The received tensors
    input_tensor_list : List of tensor
        The tensors to exchange
    """
    # send tensor to each target trainer using torch.distributed.isend
    # isend is async
    senders = []
    for i in range(world_size):
        if i == rank:
            output_tensor_list[i] = input_tensor_list[i].to(th.device('cpu'))
        else:
            sender = dist.isend(input_tensor_list[i].to(th.device('cpu')), dst=i)
            senders.append(sender)

    for i in range(world_size):
        if i != rank:
            dist.recv(output_tensor_list[i], src=i)

    th.distributed.barrier()

def alltoallv_nccl(rank, world_size, output_tensor_list, input_tensor_list):
    """Each process scatters list of input tensors to all processes in a cluster
    and return gathered list of tensors in output list.

    Note: for NCCL backend

    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire
    output_tensor_list : List of tensor
        The received tensors
    input_tensor_list : List of tensor
        The tensors to exchange
    """
    # send tensor to each target trainer using torch.distributed.isend
    # isend is async
    senders = []
    for i in range(world_size):
        if i == rank:
            output_tensor_list[i] = input_tensor_list[i]
        else:
            sender = dist.isend(input_tensor_list[i], dst=i)
            senders.append(sender)

    for i in range(world_size):
        if i != rank:
            dist.recv(output_tensor_list[i], src=i)

    th.distributed.barrier()

def all_reduce_sum(tensor):
    """Use a specific dist.all_reduce function
    """
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
