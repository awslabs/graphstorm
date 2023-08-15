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
"""
import torch as th
import random

import torch.nn.functional as F

from ..lm_model import TOKEN_IDX, VALID_LEN, ATT_MASK_IDX

DFS_TRANSVERSE = "dfs"
BFS_TRANSVERSE = "bfs"

def pad_seq(sequence, max_sequence_len, value=0):
    return F.pad(sequence, (0, max_sequence_len-len(sequence)), value=value)

def sequence_dfs2bfs(sequences, idx, max_sentence_len):
                    sequences = sequences.chunk(len(sequences) // max_sentence_len)
                    sequences = sequences[idx]
                    return th.cat(sequences)

def prepare_hat_node_centric(data, input_nodes, seeds, blocks,
                             max_sentence_len,
                             max_sequence_len,
                             transverse_format=DFS_TRANSVERSE,
                             shuffle_neighbor_order=True):
    """ Convert sampled subgraphs (blocks) into sequences of sentence sequences

        Parameters
        ----------
        data: GSgnnData
            The dataset
        input_nodes: th.Tensor
            All the nodes in blocks
        seeds: dict of th.Tensor
            Target nodes
        blocks: dgl.Block
            DGL computation graphs (layer by layer)
        max_sentence_len: int
            Max sequence length of a single node.
        max_sequence_len: int
            Max length of output token sequence.
        transverse_format: str
            How graph is tranversed and stored. "dfs" for depth first.
            "bfs" for breadth first.
        shuffle_neighbor_order: bool
            Whether we shuffle the order of a node's neighbors
    """
    g = data.g
    if not isinstance(input_nodes, dict):
        assert len(g.ntypes) == 1, \
                "We don't know the input node type, but the graph has more than one node type."
        input_nodes = {g.ntypes[0]: input_nodes}

    token_ids = data.get_node_feat(input_nodes, TOKEN_IDX)
    attention_masks = data.get_node_feat(input_nodes, ATT_MASK_IDX)

    # create a list for each seed node.
    batch_input = {seed_ntype: th.unsqueeze(seed_ids, 1) for seed_ntype, seed_ids in seeds.items()}
    # TODO(xiang): Need to support encoding Edge Type into sequences.
    layers = [{} for _ in range(len(blocks))]
    for i, block in enumerate(blocks):
        for dst_ntype in block.dsttypes:
            dstnodes = block.dstnodes(dst_ntype)
            # layers[i][dst_ntype] stores {dst_id: neighbor info of dst_id}
            layers[i][dst_ntype] = {}

            # handle all edges in block whose target ntype is dst_ntype
            # per_etype_info has the following structure:
            # [(src_ntype_0, {dst_id: [src_id, ...]}),
            #  ...
            #  (src_ntype_N, {dst_id: [src_id, ...]})]
            per_etype_info = []
            for etype in block.canonical_etypes:
                if etype[2] == dst_ntype:
                    src_ntype = etype[0]
                    src, dst = block.in_edges(dstnodes, etype=etype)
                    neighbor_info = {dst_id:[] for dst_id in dstnodes.tolist()}
                    for src_id, dst_id in zip(src.tolist(), dst.tolist()):
                        neighbor_info[dst_id].append(src_id)
                    per_etype_info.append((src_ntype, neighbor_info))

            for src_ntype, edges in per_etype_info:
                for dst_id, src_id_list in edges.items():
                    # Now layers[i][dst_ntype] looks like:
                    # { dst_id_0: [(src_ntype, [src_id, ...]), ...],
                    #   ...
                    #   dst_id_N: [(src_ntype, [src_id, ...]), ...]}
                    if dst_id not in layers[i][dst_ntype]:
                        layers[i][dst_ntype][dst_id] = [(src_ntype, src_id_list)]
                    else:
                        layers[i][dst_ntype][dst_id].append((src_ntype, src_id_list))
    assert len(layers[-1]) == len(batch_input)
    for ntype in batch_input.keys():
        assert len(layers[-1][ntype]) == len(batch_input[ntype])

    # construct sequences
    ordered_token_ids = [{} for _ in range(len(layers))]
    ordered_atten_mask = [{} for _ in range(len(layers))]
    shuffled_token_ids = [{} for _ in range(len(layers))]
    shuffled_atten_mask = [{} for _ in range(len(layers))]
    position_info = [{} for _ in range(len(layers))]
    for i, layer in enumerate(layers):
        for dst_ntype, dst_info in layer.items():
            ordered_token_ids[i][dst_ntype] = {}
            ordered_atten_mask[i][dst_ntype] = {}
            shuffled_token_ids[i][dst_ntype] = {}
            shuffled_atten_mask[i][dst_ntype] = {}
            position_info[i][dst_ntype] = {}
            for dst_id, neighbor_info in dst_info.items():
                tokens = token_ids[dst_ntype][dst_id]
                attention_mask = attention_masks[dst_ntype][dst_id]
                if len(attention_masks[dst_ntype].shape) == 1:
                    attention_mask = attention_mask.long()
                    att_mask = th.arange(0, len(tokens), device=tokens.device)
                    att_mask = att_mask < attention_mask
                else:
                    assert attention_mask.shape[1] == tokens.shape[1], \
                        "The shape of token should equal to the shape of attention mask"
                    att_mask = attention_mask
                ordered_token_ids[i][dst_ntype][dst_id] = [tokens]
                ordered_atten_mask[i][dst_ntype][dst_id] = [att_mask]
                position_info[i][dst_ntype][dst_id] = [len(layers)-i-1]
                if shuffle_neighbor_order: # shuffled neighbors
                    shuffled_token_ids[i][dst_ntype][dst_id] = [tokens]
                    shuffled_atten_mask[i][dst_ntype][dst_id] = [att_mask]

                def get_neighbor_info(src_ntype, src_id):
                    if i == 0: # first layer
                        tokens = token_ids[src_ntype][src_id]
                        attention_mask = attention_masks[src_ntype][src_id]
                        if len(attention_masks[src_ntype].shape) == 1:
                            attention_mask = attention_mask.long()
                            att_mask = th.arange(0, len(tokens), device=tokens.device)
                            att_mask = att_mask < attention_mask
                        else:
                            att_mask = attention_mask
                        pos = len(layers)-i
                    else:
                        tokens = ordered_token_ids[i-1][src_ntype][src_id]
                        att_mask = ordered_atten_mask[i-1][src_ntype][src_id]
                        pos = position_info[i-1][src_ntype][src_id]
                    return tokens, att_mask, pos

                for (src_ntype, src_list) in neighbor_info:
                    for src_id in src_list:
                        tokens, att_mask, pos = get_neighbor_info(src_ntype, src_id)
                        if i == 0: # first layer
                            ordered_token_ids[i][dst_ntype][dst_id].append(tokens)
                            ordered_atten_mask[i][dst_ntype][dst_id].append(att_mask)
                            position_info[i][dst_ntype][dst_id].append(pos)
                        else:
                            ordered_token_ids[i][dst_ntype][dst_id].extend(tokens)
                            ordered_atten_mask[i][dst_ntype][dst_id].extend(att_mask)
                            position_info[i][dst_ntype][dst_id].extend(pos)

                if shuffle_neighbor_order:
                    # shuffle the order of different etype
                    random.shuffle(neighbor_info)
                    for (src_ntype, src_list) in neighbor_info:
                        # shuffle the order of neighbors.
                        random.shuffle(src_list)
                        for src_id in src_list:
                            tokens, att_mask, _ = get_neighbor_info(src_ntype, src_id)
                            if i == 0: # first layer
                                shuffled_token_ids[i][dst_ntype][dst_id].append(tokens)
                                shuffled_atten_mask[i][dst_ntype][dst_id].append(att_mask)
                            else:
                                shuffled_token_ids[i][dst_ntype][dst_id].extend(tokens)
                                shuffled_atten_mask[i][dst_ntype][dst_id].extend(att_mask)

    # create a minibatch
    ret_ordered_token_ids = {}
    ret_ordered_atten_mask = {}
    ret_shuffled_token_ids = {}
    ret_shuffled_atten_mask = {}
    ret_position_info = {}
    for dst_ntype in ordered_token_ids[-1].keys():
        # ordered_token_ids[-1][dst_ntype] stores:
        # { dst_id_0: sequences of the ego network of dst_id_0)
        #   ...
        #   dst_id_N: sequences of the ego network of dst_id_N}
        ret_position_info[dst_ntype] = [th.tensor(pos) \
            for pos in position_info[-1][dst_ntype].values()]
        ret_ordered_token_ids[dst_ntype] = [th.cat(sequences) \
            for sequences in ordered_token_ids[-1][dst_ntype].values()]
        ret_ordered_atten_mask[dst_ntype] = [th.cat(sequences) \
            for sequences in ordered_atten_mask[-1][dst_ntype].values()]
        if shuffle_neighbor_order:
            ret_shuffled_token_ids[dst_ntype] = [th.cat(sequences) \
                for sequences in shuffled_token_ids[-1][dst_ntype].values()]
            ret_shuffled_atten_mask[dst_ntype] = [th.cat(sequences) \
                for sequences in shuffled_atten_mask[-1][dst_ntype].values()]

        # originally, graph are stored in DFS format
        if transverse_format == BFS_TRANSVERSE:
            for i, pos_info in enumerate(ret_position_info[dst_ntype]):
                new_idx = []
                for j in range(len(layers)):
                    idx = th.nonzero(pos_info == j, as_tuple=True).tolist()
                    new_idx.extend(idx)
                new_idx = th.tensor(new_idx)

                ret_ordered_token_ids[dst_ntype][i] = \
                    sequence_dfs2bfs(ret_ordered_token_ids[dst_ntype][i],
                                     new_idx, max_sentence_len)
                ret_ordered_atten_mask[dst_ntype][i] = \
                    sequence_dfs2bfs(ret_ordered_atten_mask[dst_ntype][i],
                                     new_idx, max_sentence_len)
                if shuffle_neighbor_order:
                    ret_shuffled_token_ids[dst_ntype][i] = \
                        sequence_dfs2bfs(ret_shuffled_token_ids[dst_ntype][i],
                                        new_idx, max_sentence_len)
                    ret_shuffled_atten_mask[dst_ntype][i] = \
                        sequence_dfs2bfs(ret_shuffled_atten_mask[dst_ntype][i],
                                        new_idx, max_sentence_len)
                ret_position_info[dst_ntype][i] = ret_position_info[dst_ntype][i][new_idx]
        else:
            assert transverse_format == DFS_TRANSVERSE, \
                f"Unsupported graph tranverse method {transverse_format}")

        ret_ordered_token_ids[dst_ntype] = th.stack( \
            [pad_seq(sequence, max_sequence_len) \
            if len(sequence) <= max_sequence_len else sequence[:max_sequence_len] \
            for sequence in ret_ordered_token_ids[dst_ntype]])
        ret_ordered_atten_mask[dst_ntype] = th.stack( \
            [pad_seq(sequence, max_sequence_len) \
            if len(sequence) <= max_sequence_len else sequence[:max_sequence_len] \
            for sequence in ret_ordered_atten_mask[dst_ntype]])
        max_num_setence = max_sequence_len // max_sentence_len
        ret_position_info[dst_ntype] = th.stack( \
            [pad_seq(th.tensor(pos_info), max_num_setence, -1) \
            if len(pos_info) <= max_num_setence else th.tensor(pos_info[:max_num_setence]) \
            for pos_info in ret_position_info[dst_ntype]])

        if shuffle_neighbor_order:
            ret_shuffled_token_ids[dst_ntype] = th.stack( \
                [pad_seq(sequence, max_sequence_len) \
                if len(sequence) <= max_sequence_len else sequence[:max_sequence_len] \
                for sequence in ret_shuffled_token_ids[dst_ntype]])
            ret_shuffled_atten_mask[dst_ntype] = th.stack( \
                [pad_seq(sequence, max_sequence_len) \
                if len(sequence) <= max_sequence_len else sequence[:max_sequence_len] \
                for sequence in ret_shuffled_atten_mask[dst_ntype]])

    return ret_ordered_token_ids, \
        ret_ordered_atten_mask, \
        ret_shuffled_token_ids, \
        ret_shuffled_atten_mask, \
        ret_position_info

def get_prepare_lm_input(dataloader):
    if dataloader == "lm_hat_node_centric":
        return prepare_hat_node_centric

    raise RuntimeError(f"Unknow dataloader type {dataloader}")
