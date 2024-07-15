import torch
import torch.nn as nn

import re
from dgl import create_block

def get_trainable_params(in_dims, out_dims):
    """
    Create trainable parameters

    Parameters
    ----------
    in_dims: int
        The number of input dimensions
    out_dims: int
        The number of output dimensions

    Returns
    -------
        Tensor of size (in_dims, out_dims)
    """
    params = nn.Parameter(torch.Tensor(in_dims, out_dims))
    nn.init.xavier_uniform_(params, gain=nn.init.calculate_gain("relu"))
    return params

def to_per_field_nfeats(inputs, fields):
    """
    This function splits tensors in `inputs` into multiple sub-tensors column-wise,
    then name each sub-tensor according to the field name in `fields`.

        inputs = {
            'ntype': concat([tensor1, tensor2], dim=1)
        }
        fields = [field1, field2]
        outputs = {
            'ntype': {
                'field1': tensor1,
                'field2': tensor2,
            }
        }

    Parameters
    ----------
    inputs: dict of Tensor
        A dictionary that maps the ntype to its ntype feature
    fields: list of String
        The fields that contain the node features

    Returns
    -------
        dict of dict of Tensor
    """
    outputs = {}
    for ntype, all_feats in inputs.items():
        all_feats = torch.nan_to_num(all_feats, 0).float()
        all_feats = all_feats.tensor_split(len(fields), dim=1)
        outputs[ntype] = {f"{t}_feat": feat for t, feat in zip(fields, all_feats)}
    return outputs

def rel_name_map(rel_name):
    """
    Remove the number suffix from relation name.

        {rel_name}_{i}-rev --> {rel_name}-rev
        {rel_name}_{i} --> {rel_name}

    Parameters
    ----------
    rel_name: string
        relation name with number suffix

    Returns
    -------
        string: relation name without number suffix
    """
    if "-rev" in rel_name:
        return (
            rel_name.split("-")[0].rstrip("_0123456789") + "-rev"
        )
    return rel_name.rstrip("_0123456789")


def get_unique_rel_names(rel_names):
    """
    Given a list of relation name, remove the number suffix from each relation name.
    This function calls `rel_name_map` on each relation name in the list.

    Parameters
    ----------
    rel_names: list of string
        list of relation name with number suffix

    Returns
    -------
        list of string: list of relation name without number suffix
    """
    rel_names = [rel_name_map(rel_name) for rel_name in rel_names]
    rel_names = list(set(rel_names))
    rel_names.sort()
    return rel_names

def get_unique_etype_triplet(etype_triplet):
    """
    Given a list of etype triplet (src, relation_name, dst), remove the number suffix from each relation name.
    This function calls `rel_name_map` on the relation name of each etype triplet in the list.

    Parameters
    ----------
    etype_triplet: list of triplet
        list of etype triplet (src, relation_name, dst) with number suffix

    Returns
    -------
        list of triplet: list of etype triplet (src, relation_name, dst) without number suffix
    """
    # remove the timestamp from relation name, then form as triplet (src, rel, dst)
    etype_triplet = [(src, rel_name_map(rel), dst) for src, rel, dst in etype_triplet]
    etype_triplet = list(set(etype_triplet))
    etype_triplet.sort()
    return etype_triplet

def rel_field_map(rel_name):
    """
    Return the number suffix from relation name.

        {rel_name}_{i}-rev --> i
        {rel_name}_{i} --> i

    Parameters
    ----------
    rel_name: string
        relation name with number suffix

    Returns
    -------
        int: number suffix of each relation name
    """
    t = re.findall(r"\d+", rel_name)[-1]
    assert t.isnumeric()
    return t


def get_unique_nfields(rel_names):
    """
    Given a list of relation name, return the number suffix from each relation name.
    This function calls `rel_field_map` on each relation name in the list.

    Parameters
    ----------
    rel_names: list of string
        list of relation name with number suffix

    Returns
    -------
        list of int: list of sorted number suffix of each relation name
    """
    field_ids = [rel_field_map(rel_name) for rel_name in rel_names]
    field_ids = list(set(field_ids))
    field_ids.sort()
    return field_ids

def get_temporal_ordered_etypes(etypes):
    """
    This function is used to construct the DGLBlock for temporal aggregation,
    which is later used with `merge_multi_blocks` to construct a new DGLBlock for temporal aggregation.

    Parameters
    ----------
    inputs: list of etypes. For example,
        inputs = [
            ('paper', 'cite_04', 'paper'),
            ('paper', 'cite_05', 'paper'),
            ('paper', 'cite_06', 'paper')
        ]

    Returns
    -------
        dict that map each etype to a list of etypes. For example,
            outputs = {
                ('paper', 'cite_04', 'paper'): [
                    ('paper', 'cite_04', 'paper')
                ],
                ('paper', 'cite_05', 'paper'): [
                    ('paper', 'cite_04', 'paper'),
                    ('paper', 'cite_05', 'paper')
                ],
                ('paper', 'cite_06', 'paper'): [
                    ('paper', 'cite_04', 'paper'),
                    ('paper', 'cite_05', 'paper'),
                    ('paper', 'cite_06', 'paper')
                ]
            }

    """
    etype_group = {
        etype: [] for etype in get_unique_etype_triplet(etypes)
    }

    for src, rel, dst in etypes:
        etype = (src, rel_name_map(rel), dst)
        etype_group[etype].append((src, rel, dst))

    mapping = {}
    for k, v in etype_group.items():
        v.sort()
        v = [v[:i] for i in range(1, len(v) + 1)]

        for etype_list in v:
            mapping[etype_list[-1]] = etype_list
    return mapping

def merge_multi_blocks(block, embs, merge_canonical_etypes):
    """
    Given a DGLBlock and a list of edge types, this function construct a new DGLBlocks
    by merging serveral edge types in the DGLBlock based on the edge type list. For example,
    let us assume `block` contains three edge types [
            ('paper', 'cite_04', 'paper'),
            ('paper', 'cite_05', 'paper'),
            ('paper', 'cite_06', 'paper')
        ] and `merge_canonical_etypes` is [
            ('paper', 'cite_04', 'paper'),
            ('paper', 'cite_05', 'paper')
        ]
    Then, this function return a new DGLBlock that considers all edges in `merge_canonical_etypes`
    to construct the edges in ('paper', 'cite_05', 'paper'). By doing so, day 5's output only get information
    from day 4's input embeddings and day 5's input embeddings.

    Parameters
    ----------
    block: DGLBlock
        This is the DGLBlock returned by GraphStorm data loader.

    embs: dict of Tensor
        This is the hidden embs, e.g., {ntype1: Tensor1, ntype2: Tensor2}

    merge_canonical_etypes: list of edge types
        This is the return of function `get_temporal_ordered_etypes`

    Returns
    -------
        DGLBlock.
    """

    # check format correctiveness
    srctypes, dsttypes, merge_rel_types = set(), set(), list()
    for src, rel, dst in merge_canonical_etypes:
        srctypes.add(src)
        dsttypes.add(dst)
        merge_rel_types.append(rel)
    assert len(srctypes) == 1 and len(dsttypes) == 1
    srctypes, dsttypes = list(srctypes)[0], list(dsttypes)[0]

    # fetch useful information
    merge_rel_types = [etype[1] for etype in merge_canonical_etypes]
    merge_fields = get_unique_nfields(merge_rel_types)
    num_merge = len(merge_canonical_etypes)
    last_canonical_etypes = merge_canonical_etypes[-1]

    num_src_nodes = block.num_src_nodes(srctypes)
    num_dst_nodes = block.num_dst_nodes(dsttypes)

    # create message passing block
    new_src_nodes, new_dst_nodes = [], []
    for i, canonical_etypes in enumerate(merge_canonical_etypes):
        src, dst = block[canonical_etypes].edges()
        new_src_nodes.append(src + num_src_nodes * i)
        new_dst_nodes.append(dst)

    new_block = create_block(
        num_src_nodes={
            srctypes: num_src_nodes * num_merge
        },
        num_dst_nodes={
            dsttypes: num_dst_nodes
        },
        data_dict={
            last_canonical_etypes: (
                torch.cat(new_src_nodes),
                torch.cat(new_dst_nodes)
            )
        }
    )

    # copy src & dst _ID informaiton
    sg = block[last_canonical_etypes]
    for k, v in sg.srcnodes[srctypes].data.items():
        new_block.srcnodes[srctypes].data[k] = v.repeat(num_merge)
    for k, v in sg.dstnodes[dsttypes].data.items():
        new_block.dstnodes[dsttypes].data[k] = v

    # prepare src & dst node feats
    src_feats = torch.cat([embs[srctypes][f'{field}_feat'] for field in merge_fields], dim=0)
    dst_feats = embs[dsttypes][f'{merge_fields[-1]}_feat'][:num_dst_nodes]

    return new_block, src_feats, dst_feats

def get_specific_field(inputs, nfield):
    """
    Select a subset of Tensors from `inputs` that is associated with `nfield`. For example,

    inputs = {
        node_type_1: {
            node_type_1_field_1: node_type_1_field_1_values,
            node_type_1_field_2: node_type_1_field_2_values,
            node_type_1_field_3: node_type_1_field_3_values
        },
        node_type_2: {
            node_type_2_field_1: node_type_2_field_1_values,
            node_type_2_field_2: node_type_2_field_2_values,
            node_type_2_field_3: node_type_2_field_3_values
        }
    }
    nfield = "field_3"

    outputs = {
        node_type_1: node_type_1_field_3,
        node_type_2: node_type_2_field_3
    }

    Parameters
    ----------
    inputs: dict of dict of Tensor
        map a node type to the the per node type field tensor.

    nfield: string
        the field tensor we would like to remain

    Returns
    -------
        dict of Tensor

    """
    outputs = {}
    for ntype, ntype_inputs in inputs.items():
        outputs[ntype] = ntype_inputs[nfield]
    return outputs


def average_over_fields(inputs):
    """
    For each node type, average all its field tensors.

    inputs = {
        node_type_1: {
            node_type_1_field_1: node_type_1_field_1_values,
            node_type_1_field_2: node_type_1_field_2_values,
            node_type_1_field_3: node_type_1_field_3_values
        },
        node_type_2: {
            node_type_2_field_1: node_type_1_field_1_values,
            node_type_2_field_2: node_type_1_field_2_values,
            node_type_2_field_3: node_type_1_field_3_values
        }
    }

    outputs = {
        node_type_1: node_type_1_values,
        node_type_2: node_type_2_values
    }

    Parameters
    ----------
    inputs: dict of dict of Tensor
        map a node type to the the per node type field tensor.

    Returns
    -------
        dict of Tensor
    """
    outputs = {}
    for ntype, ntype_inputs in inputs.items():
        outputs[ntype] = list(ntype_inputs.values())
        outputs[ntype] = torch.stack(outputs[ntype], dim=0).mean(dim=0)
    return outputs

def concat_over_fields(inputs, fields):
    """
    For each node type, concatenate all its field tensors,
    where the field tensors are sorted by their tensor name.

    inputs = {
        node_type_1: {
            node_type_1_field_1: node_type_1_field_1_values,
            node_type_1_field_2: node_type_1_field_2_values,
            node_type_1_field_3: node_type_1_field_3_values
        },
        node_type_2: {
            node_type_2_field_1: node_type_1_field_1_values,
            node_type_2_field_2: node_type_1_field_2_values,
            node_type_2_field_3: node_type_1_field_3_values
        }
    }

    outputs = {
        node_type_1: node_type_1_values,
        node_type_2: node_type_2_values
    }

    Parameters
    ----------
    inputs: dict of dict of Tensor
        map a node type to the the per node type field tensor.

    Returns
    -------
        dict of Tensor
    """
    outputs = {}
    for ntype, ntype_inputs in inputs.items():
        outputs[ntype] = torch.cat([ntype_inputs[f'{t}_feat'] for t in fields], dim=1)
    return outputs
