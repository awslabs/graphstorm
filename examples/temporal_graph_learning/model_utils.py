import torch
import torch.nn as nn

import re
from dgl import create_block

def get_trainable_params(in_dims, out_dims):
    params = nn.Parameter(torch.Tensor(in_dims, out_dims))
    nn.init.xavier_uniform_(params, gain=nn.init.calculate_gain("relu"))
    return params

def to_per_field_nfeats(inputs, fields):
    outputs = {}
    for ntype, all_feats in inputs.items():
        all_feats = torch.nan_to_num(all_feats, 0).float()
        all_feats = all_feats.tensor_split(len(fields), dim=1)
        outputs[ntype] = {f"{t}_feat": feat for t, feat in zip(fields, all_feats)}
    return outputs

def rel_name_map(rel_name):
    # remove the timestamp from relation name
    if "-rev" in rel_name:
        # {rel_name}_{field_id}-rev --> {rel_name}-rev
        return (
            rel_name.split("-")[0].rstrip("_0123456789") + "-rev"
        )
    # {rel_name}_{field_id} --> {rel_name}
    return rel_name.rstrip("_0123456789")


def get_unique_rel_names(rel_names):
    # remove the timestamp from relation name
    rel_names = [rel_name_map(rel_name) for rel_name in rel_names]
    rel_names = list(set(rel_names))
    rel_names.sort()
    return rel_names

def rel_field_map(rel_name):
    # {rel_name}_{field_id} and {rel_name}_{field_id}-rev
    t = re.findall(r"\d+", rel_name)[-1]
    assert t.isnumeric() 
    return t


def get_unique_nfields(rel_names):
    field_ids = [rel_field_map(rel_name) for rel_name in rel_names]
    field_ids = list(set(field_ids))
    field_ids.sort()
    return field_ids


def get_unique_etype_triplet(etype_triplet):
    # remove the timestamp from relation name, then form as triplet (src, rel, dst)
    etype_triplet = [(src, rel_name_map(rel), dst) for src, rel, dst in etype_triplet]
    etype_triplet = list(set(etype_triplet))
    etype_triplet.sort()
    return etype_triplet

def get_trainable_params(in_dims, out_dims):
    params = nn.Parameter(torch.Tensor(in_dims, out_dims))
    nn.init.xavier_uniform_(params, gain=nn.init.calculate_gain("relu"))
    return params


def get_temporal_restrict_fields(fields):
    temporal_restrict_fields = {}
    for i, fn in enumerate(fields):
        temporal_restrict_fields[f"{fn}_feat"] = [
            f"{prev_fn}_feat" for prev_fn in fields[i:]
        ]
    return temporal_restrict_fields

def get_merge_canonical_etype_mapping(etypes):
    """
    inputs = [
        ('paper', 'cite_04', 'paper'),
        ('paper', 'cite_05', 'paper'),
        ('paper', 'cite_06', 'paper')
    ]
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
        node_type_1: node_type_1_field_3,
        node_type_2: node_type_1_field_3
    }
    """
    outputs = {}
    for ntype, ntype_inputs in inputs.items():
        outputs[ntype] = ntype_inputs[nfield]
    return outputs


def average_over_fields(inputs):
    """
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
    """
    outputs = {}
    for ntype, ntype_inputs in inputs.items():
        outputs[ntype] = list(ntype_inputs.values())
        outputs[ntype] = torch.stack(outputs[ntype], dim=0).mean(dim=0)
    return outputs

def concat_over_fields(inputs, fields):
    """
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
    """
    outputs = {}
    for ntype, ntype_inputs in inputs.items():
        outputs[ntype] = torch.cat([ntype_inputs[f'{t}_feat'] for t in fields], dim=1)
    return outputs
