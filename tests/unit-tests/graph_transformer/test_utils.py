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
import os
import tempfile
import pytest
import multiprocessing as mp
from numpy.testing import assert_equal

import torch as th
import dgl

from graphstorm.model.lm_model import TOKEN_IDX, ATT_MASK_IDX
from graphstorm.model.graph_transformer import get_prepare_lm_input
from graphstorm.model.graph_transformer.utils import (pad_seq,
                                                      sequence_dfs2bfs,
                                                      prepare_hat_node_centric)


def test_get_prepare_lm_input():
    loader = get_prepare_lm_input("lm_hat_node_centric")
    assert callable(loader)

    error = False
    try:
        loader = get_prepare_lm_input("other")
    except:
        # unknow lm data loading type
        error = True
    assert error is True

def test_pad_seq():
    seq = th.arange(10)
    new_seq = pad_seq(seq, 16, value=0)
    assert_equal(seq.numpy(), new_seq[:10].numpy())
    assert th.count_nonzero(new_seq[10:]) == 0

    new_seq = pad_seq(seq, 16, value=1)
    assert_equal(seq.numpy(), new_seq[:10].numpy())
    assert th.sum(new_seq[10:]) == 6

class DummpyTrainData():
    def __init__(self, node_data):
        self._g = None # dummy
        self._node_data = node_data

    def get_node_feat(self, input_nodes, feat_name):
        return {ntype: self._node_data[ntype][feat_name][nid] \
                for ntype, nid in input_nodes.items()}


    @property
    def g(self):
        return self._g

def test_prepare_hat_node_centric_input():
    num_ntype0 = 20
    num_ntype1 = 40
    max_sentence_len = 16
    max_sequence_len = 64 * max_sentence_len
    node_data = {
        "ntype0" : {
            TOKEN_IDX: th.tensor(([[i] * max_sentence_len for i in range(num_ntype0)])),
            ATT_MASK_IDX: th.randint(1, max_sentence_len, (num_ntype0,)),
        },
        "ntype1" : {
            TOKEN_IDX: th.tensor(([[i] * max_sentence_len for i in range(num_ntype1)])),
            ATT_MASK_IDX: th.randint(1, max_sentence_len, (num_ntype1,)),
        }
    }
    batch_size = 4

    # generate a graph
    num_block_ntype0 = 16
    num_block_ntype1 = 20
    input_nodes = {
        "ntype0": th.arange(num_block_ntype0),
        "ntype1": th.arange(num_block_ntype1) + 4
    }

    block1 = dgl.create_block({
        ("ntype1", "rel0", "ntype0"): ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], \
                                       [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]),
        ("ntype0", "rel1", "ntype0"): ([5, 6, 7, 8, 9], [0, 0, 1, 2, 3])},
        num_src_nodes={"ntype0": 10, "ntype1": 10},
        num_dst_nodes={"ntype0": 4})
    block0 = dgl.create_block({
        ("ntype1", "rel0", "ntype0"): ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], \
                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("ntype1", "rel1", "ntype1"): ([i for i in range(20)] * 2, [i//4 for i in range(40)]),
        ("ntype0", "rel1", "ntype0"): ([10, 11, 12, 13, 14, 15], [0, 1, 2, 3, 4, 5])},
        num_src_nodes={"ntype0": num_block_ntype0, "ntype1": num_block_ntype1},
        num_dst_nodes={"ntype0": 10, "ntype1": 10})

    blocks = [block0, block1]
    seeds = {"ntype0": th.tensor([1,2,3,4])}

    data = DummpyTrainData(node_data)

    ordered_token_ids, ordered_atten_mask, \
        shuffled_token_ids, shuffled_atten_mask, \
        position_info = prepare_hat_node_centric_input(
            data, input_nodes, seeds, blocks, max_sentence_len, max_sequence_len)

    # check position encoding first
    pos_info = position_info["ntype0"][0]
    assert th.sum(pos_info == 0).item() == 1
    assert th.sum(pos_info == 1).item() == 6
    assert th.sum(pos_info == 2).item() == 21

    pos_info = position_info["ntype0"][1]
    assert th.sum(pos_info == 0).item() == 1
    assert th.sum(pos_info == 1).item() == 5
    assert th.sum(pos_info == 2).item() == 18

    pos_info = position_info["ntype0"][2]
    assert th.sum(pos_info == 0).item() == 1
    assert th.sum(pos_info == 1).item() == 5
    assert th.sum(pos_info == 2).item() == 18

    pos_info = position_info["ntype0"][3]
    assert th.sum(pos_info == 0).item() == 1
    assert th.sum(pos_info == 1).item() == 9
    assert th.sum(pos_info == 2).item() == 34


if __name__ == '__main__':
    test_get_prepare_lm_input()
    test_pad_seq()
    test_prepare_hat_node_centric_input()