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
import random
import multiprocessing as mp
from numpy.testing import assert_equal, assert_raises

import torch as th
import dgl

from graphstorm.model.lm_model import TOKEN_IDX, ATT_MASK_IDX
from graphstorm.model.graph_transformer import get_prepare_lm_input
from graphstorm.dataloading.graph_lm_dataloading import BFS_TRANSVERSE
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

def test_sequence_dfs2bfs():
    sentence_len = 16
    num_sentence = 8
    sentences = [th.rand((sentence_len,)) for i in range(num_sentence)]
    idxs = th.randperm(num_sentence).long()
    sequence = th.cat(sentences)
    seq_after_shuffler = sequence_dfs2bfs(sequence, idxs, sentence_len)
    ground_truth = th.cat([sentences[idx] for idx in idxs.tolist()])
    assert_equal(ground_truth.numpy(), seq_after_shuffler.numpy())

    is_dividable_sentence_len = False
    try:
        sequence_dfs2bfs(sequence, idxs, 15)
    except:
        is_dividable_sentence_len = True
    assert is_dividable_sentence_len

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
            TOKEN_IDX: th.tensor(([[i+num_ntype0] * max_sentence_len for i in range(num_ntype1)])),
            ATT_MASK_IDX: th.randint(1, max_sentence_len, (num_ntype1,)),
        }
    }

    # generate a graph
    num_block_ntype0 = 16
    num_block_ntype1 = 20
    input_nodes = {
        "ntype0": th.arange(num_block_ntype0),
        "ntype1": th.arange(num_block_ntype1) + 4
    }

    block0 = dgl.create_block({
        ("ntype1", "rel0", "ntype0"): ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5,
                                        4, 3, 2, 1, 0], \
                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4,
                                        5, 6, 7, 8, 9]),
        ("ntype1", "rel1", "ntype1"): ([i for i in range(20)] * 2, [i//4 for i in range(40)]),
        ("ntype0", "rel1", "ntype0"): ([10, 11, 12, 13, 14, 15], [0, 1, 2, 3, 4, 5])},
        num_src_nodes={"ntype0": num_block_ntype0, "ntype1": num_block_ntype1},
        num_dst_nodes={"ntype0": 10, "ntype1": 10})
    block1 = dgl.create_block({
        ("ntype1", "rel0", "ntype0"): ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], \
                                       [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]),
        ("ntype0", "rel1", "ntype0"): ([5, 6, 7, 8, 9], [0, 0, 1, 2, 3])},
        num_src_nodes={"ntype0": 10, "ntype1": 10},
        num_dst_nodes={"ntype0": 4})

    blocks = [block0, block1]
    seeds = {"ntype0": th.tensor([1,2,3,4])}

    data = DummpyTrainData(node_data)

    th.manual_seed(0)
    random.seed(0)
    dfs_ordered_token_ids, dfs_ordered_atten_mask, \
        dfs_shuffled_token_ids, dfs_shuffled_atten_mask, \
        _, dfs_position_info = prepare_hat_node_centric(
            data, input_nodes, seeds, blocks, max_sentence_len, max_sequence_len)

    # check position encoding first
    pos_info = dfs_position_info["ntype0"][0]
    assert th.sum(pos_info == 0).item() == 1
    assert th.sum(pos_info == 1).item() == 6
    assert th.sum(pos_info == 2).item() == 21

    pos_info = dfs_position_info["ntype0"][1]
    assert th.sum(pos_info == 0).item() == 1
    assert th.sum(pos_info == 1).item() == 5
    assert th.sum(pos_info == 2).item() == 18

    pos_info = dfs_position_info["ntype0"][2]
    assert th.sum(pos_info == 0).item() == 1
    assert th.sum(pos_info == 1).item() == 5
    assert th.sum(pos_info == 2).item() == 18

    pos_info = dfs_position_info["ntype0"][3]
    assert th.sum(pos_info == 0).item() == 1
    assert th.sum(pos_info == 1).item() == 9
    assert th.sum(pos_info == 2).item() == 34

    # seed node will be 0,1,2,3
    for i in range(4):
        assert_equal(dfs_ordered_token_ids["ntype0"][i][:max_sentence_len].numpy(),
                    node_data["ntype0"][TOKEN_IDX][i].numpy())
        assert_equal(dfs_shuffled_token_ids["ntype0"][i][:max_sentence_len].numpy(),
                    node_data["ntype0"][TOKEN_IDX][i].numpy())
        assert th.sum(dfs_ordered_atten_mask["ntype0"][i][:max_sentence_len]) == \
            node_data["ntype0"][ATT_MASK_IDX][i]
        assert th.sum(dfs_shuffled_atten_mask["ntype0"][i][:max_sentence_len]) == \
            node_data["ntype0"][ATT_MASK_IDX][i]
    # check first dfs neighbors
    # 0 -> ntype0: 5 -> ntype0: 15, ntype1: 5+4, ntype1: 4+4 (ntype1 with offset 4)
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len:max_sentence_len*2].numpy(),
        node_data["ntype0"][TOKEN_IDX][5].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len:max_sentence_len*2]) == \
            node_data["ntype0"][ATT_MASK_IDX][5]
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*2:max_sentence_len*3].numpy(),
        node_data["ntype0"][TOKEN_IDX][15].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*2:max_sentence_len*3]) == \
            node_data["ntype0"][ATT_MASK_IDX][15]
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*3:max_sentence_len*4].numpy(),
        node_data["ntype1"][TOKEN_IDX][9].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*3:max_sentence_len*4]) == \
            node_data["ntype1"][ATT_MASK_IDX][9]
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*4:max_sentence_len*5].numpy(),
        node_data["ntype1"][TOKEN_IDX][8].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*4:max_sentence_len*5]) == \
            node_data["ntype1"][ATT_MASK_IDX][8]
    # check second dfs neighbors
    # 0 -> ntype0: 6 -> ntype1: 6+4, ntype1: 3+4 (ntype1 with offset 4)
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*5:max_sentence_len*6].numpy(),
        node_data["ntype0"][TOKEN_IDX][6].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*5:max_sentence_len*6]) == \
            node_data["ntype0"][ATT_MASK_IDX][6]
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*6:max_sentence_len*7].numpy(),
        node_data["ntype1"][TOKEN_IDX][10].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*6:max_sentence_len*7]) == \
            node_data["ntype1"][ATT_MASK_IDX][10]
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*7:max_sentence_len*8].numpy(),
        node_data["ntype1"][TOKEN_IDX][7].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*7:max_sentence_len*8]) == \
            node_data["ntype1"][ATT_MASK_IDX][7]

    # check third dfs neighbors
    # 0 -> ntype1: 0+4 -> ntype1: 0+4, ntype1: 1+4, ntype1: 2+4, ntype1: 3+4,
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*8:max_sentence_len*9].numpy(),
        node_data["ntype1"][TOKEN_IDX][4].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*8:max_sentence_len*9]) == \
            node_data["ntype1"][ATT_MASK_IDX][4]
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*9:max_sentence_len*10].numpy(),
        node_data["ntype1"][TOKEN_IDX][4].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*9:max_sentence_len*10]) == \
            node_data["ntype1"][ATT_MASK_IDX][4]
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*10:max_sentence_len*11].numpy(),
        node_data["ntype1"][TOKEN_IDX][5].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*10:max_sentence_len*11]) == \
            node_data["ntype1"][ATT_MASK_IDX][5]
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*11:max_sentence_len*12].numpy(),
        node_data["ntype1"][TOKEN_IDX][6].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*11:max_sentence_len*12]) == \
            node_data["ntype1"][ATT_MASK_IDX][6]
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*12:max_sentence_len*13].numpy(),
        node_data["ntype1"][TOKEN_IDX][7].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*12:max_sentence_len*13]) == \
            node_data["ntype1"][ATT_MASK_IDX][7]
    # check last dfs neighbors
    # 0 -> ntype1: 3+4 -> ntype1: 12+4, ntype1: 13+4, ntype1: 14+4, ntype1: 15+4,
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*23:max_sentence_len*24].numpy(),
        node_data["ntype1"][TOKEN_IDX][7].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*23:max_sentence_len*24]) == \
            node_data["ntype1"][ATT_MASK_IDX][7]
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*24:max_sentence_len*25].numpy(),
        node_data["ntype1"][TOKEN_IDX][16].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*24:max_sentence_len*25]) == \
            node_data["ntype1"][ATT_MASK_IDX][16]
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*25:max_sentence_len*26].numpy(),
        node_data["ntype1"][TOKEN_IDX][17].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*25:max_sentence_len*26]) == \
            node_data["ntype1"][ATT_MASK_IDX][17]
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*26:max_sentence_len*27].numpy(),
        node_data["ntype1"][TOKEN_IDX][18].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*26:max_sentence_len*27]) == \
            node_data["ntype1"][ATT_MASK_IDX][18]
    assert_equal(
        dfs_ordered_token_ids["ntype0"][0][max_sentence_len*27:max_sentence_len*28].numpy(),
        node_data["ntype1"][TOKEN_IDX][19].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0][max_sentence_len*27:max_sentence_len*28]) == \
            node_data["ntype1"][ATT_MASK_IDX][19]
    # check overall statistics starting from node0: 0
    def check_overall_statistics_0(token_ids, att_mask):
        assert th.sum(token_ids["ntype0"][0]==0) == \
            1 * max_sentence_len + max_sequence_len - max_sentence_len*28
        assert th.sum(token_ids["ntype0"][0]==1) == 0
        assert th.sum(token_ids["ntype0"][0]==2) == 0
        assert th.sum(token_ids["ntype0"][0]==3) == 0
        assert th.sum(token_ids["ntype0"][0]==4) == 0
        assert th.sum(token_ids["ntype0"][0]==5) == max_sentence_len
        assert th.sum(token_ids["ntype0"][0]==6) == max_sentence_len
        assert th.sum(token_ids["ntype0"][0]==7) == 0
        assert th.sum(token_ids["ntype0"][0]==15) == max_sentence_len
        assert th.sum(token_ids["ntype0"][0]==23) == 0
        assert th.sum(token_ids["ntype0"][0]==24) == 2 * max_sentence_len # ntype1: 0+4
        assert th.sum(token_ids["ntype0"][0]==25) == 2 * max_sentence_len # ntype1: 1+4
        assert th.sum(token_ids["ntype0"][0]==26) == 2 * max_sentence_len # ntype1: 2+4
        assert th.sum(token_ids["ntype0"][0]==27) == 3 * max_sentence_len # ntype1: 3+4
        assert th.sum(token_ids["ntype0"][0]==28) == 2 * max_sentence_len # ntype1: 4+4
        assert th.sum(token_ids["ntype0"][0]==29) == 2 * max_sentence_len # ntype1: 5+4
        assert th.sum(token_ids["ntype0"][0]==30) == 2 * max_sentence_len # ntype1: 6+4
        for i in range(20+7+4, 20+16+4):
            # ntype1: 7+4 ~ 15+4
            assert th.sum(token_ids["ntype0"][0]==i) == max_sentence_len

        # check mask
        assert th.sum(att_mask["ntype0"][0]) == \
            th.sum(node_data["ntype0"][ATT_MASK_IDX][0]).item() + \
            th.sum(node_data["ntype0"][ATT_MASK_IDX][5]).item() + \
            th.sum(node_data["ntype0"][ATT_MASK_IDX][6]).item() + \
            th.sum(node_data["ntype0"][ATT_MASK_IDX][15]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][4]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][5]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][6]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][7]).item() * 3 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][8]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][9]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][10]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][11]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][12]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][13]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][14]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][15]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][16]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][17]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][18]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][19]).item()
    check_overall_statistics_0(dfs_ordered_token_ids, dfs_ordered_atten_mask)
    check_overall_statistics_0(dfs_shuffled_token_ids, dfs_shuffled_atten_mask)

    # check overall statistics starting from node0: 1
    def check_overall_statistics_1(token_ids, att_mask):
        assert th.sum(token_ids["ntype0"][1]==0) == \
            max_sequence_len - max_sentence_len*24
        assert th.sum(token_ids["ntype0"][1]==1) == max_sentence_len
        assert th.sum(token_ids["ntype0"][1]==2) == 0
        assert th.sum(token_ids["ntype0"][1]==6) == 0
        assert th.sum(token_ids["ntype0"][1]==7) == max_sentence_len
        assert th.sum(token_ids["ntype0"][1]==8) == 0
        assert th.sum(token_ids["ntype0"][1]==24) == max_sentence_len # ntype1: 0+4
        assert th.sum(token_ids["ntype0"][1]==25) == max_sentence_len # ntype1: 1+4
        assert th.sum(token_ids["ntype0"][1]==26) == 2 * max_sentence_len # ntype1: 2+4
        assert th.sum(token_ids["ntype0"][1]==27) == max_sentence_len # ntype1: 3+4
        assert th.sum(token_ids["ntype0"][1]==28) == 2 * max_sentence_len # ntype1: 4+4
        assert th.sum(token_ids["ntype0"][1]==29) == 2 * max_sentence_len # ntype1: 5+4
        assert th.sum(token_ids["ntype0"][1]==30) == 2 * max_sentence_len # ntype1: 6+4
        assert th.sum(token_ids["ntype0"][1]==31) == 3 * max_sentence_len # ntype1: 7+4
        for i in range(20+16+4, 20+20+4):
            # ntype1: 16+4 ~ 19+4
            assert th.sum(token_ids["ntype0"][1]==i) == max_sentence_len
        for i in range(20+8+4, 20+12+4):
            # ntype1: 8+4 ~ 11+4
            assert th.sum(token_ids["ntype0"][1]==i) == max_sentence_len

        # check mask
        assert th.sum(att_mask["ntype0"][1]) == \
            th.sum(node_data["ntype0"][ATT_MASK_IDX][1]).item() + \
            th.sum(node_data["ntype0"][ATT_MASK_IDX][7]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][4]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][5]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][6]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][7]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][8]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][9]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][10]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][11]).item() * 3 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][12]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][13]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][14]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][15]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][20]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][21]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][22]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][23]).item()

    check_overall_statistics_1(dfs_ordered_token_ids, dfs_ordered_atten_mask)
    check_overall_statistics_1(dfs_shuffled_token_ids, dfs_shuffled_atten_mask)

    # check overall statistics starting from node0: 2
    def check_overall_statistics_2(token_ids, att_mask):
        assert th.sum(token_ids["ntype0"][2]==0) == \
            max_sequence_len - max_sentence_len*24
        assert th.sum(token_ids["ntype0"][2]==1) == 0
        assert th.sum(token_ids["ntype0"][2]==2) == max_sentence_len
        assert th.sum(token_ids["ntype0"][2]==7) == 0
        assert th.sum(token_ids["ntype0"][2]==8) == max_sentence_len
        assert th.sum(token_ids["ntype0"][2]==9) == 0
        assert th.sum(token_ids["ntype0"][2]==24) == 0
        assert th.sum(token_ids["ntype0"][2]==25) == max_sentence_len # ntype1: 1+4
        assert th.sum(token_ids["ntype0"][2]==32) == 3 * max_sentence_len # ntype1: 8+4
        assert th.sum(token_ids["ntype0"][2]==33) == 2 * max_sentence_len # ntype1: 9+4
        for i in range(20+12+4, 20+20+4):
            # ntype1: 12+4 ~ 19+4
            assert th.sum(token_ids["ntype0"][2]==i) == 2 * max_sentence_len

        # check mask
        assert th.sum(att_mask["ntype0"][2]) == \
            th.sum(node_data["ntype0"][ATT_MASK_IDX][2]).item() + \
            th.sum(node_data["ntype0"][ATT_MASK_IDX][8]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][5]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][12]).item() * 3 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][13]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][16]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][17]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][18]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][19]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][20]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][21]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][22]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][23]).item() * 2

    check_overall_statistics_2(dfs_ordered_token_ids, dfs_ordered_atten_mask)
    check_overall_statistics_2(dfs_shuffled_token_ids, dfs_shuffled_atten_mask)

    # check overall statistics starting from node0: 3
    def check_overall_statistics_3(token_ids, att_mask):
        assert th.sum(token_ids["ntype0"][3]==0) == \
            max_sequence_len - max_sentence_len*44
        assert th.sum(token_ids["ntype0"][3]==2) == 0
        assert th.sum(token_ids["ntype0"][3]==3) == max_sentence_len
        assert th.sum(token_ids["ntype0"][3]==8) == 0
        assert th.sum(token_ids["ntype0"][3]==9) == max_sentence_len
        assert th.sum(token_ids["ntype0"][3]==10) == 0
        assert th.sum(token_ids["ntype0"][3]==24) == 4 * max_sentence_len # ntype1: 0+4
        assert th.sum(token_ids["ntype0"][3]==25) == 3 * max_sentence_len # ntype1: 1+4
        assert th.sum(token_ids["ntype0"][3]==26) == 3 * max_sentence_len # ntype1: 2+4
        assert th.sum(token_ids["ntype0"][3]==27) == 3 * max_sentence_len # ntype1: 3+4
        assert th.sum(token_ids["ntype0"][3]==28) == 3 * max_sentence_len # ntype1: 4+4
        assert th.sum(token_ids["ntype0"][3]==29) == 3 * max_sentence_len # ntype1: 5+4
        assert th.sum(token_ids["ntype0"][3]==30) == 3 * max_sentence_len # ntype1: 6+4
        assert th.sum(token_ids["ntype0"][3]==31) == 3 * max_sentence_len # ntype1: 7+4
        assert th.sum(token_ids["ntype0"][3]==32) == 2 * max_sentence_len # ntype1: 8+4
        assert th.sum(token_ids["ntype0"][3]==33) == 3 * max_sentence_len # ntype1: 9+4
        assert th.sum(token_ids["ntype0"][3]==34) == 2 * max_sentence_len # ntype1: 10+4
        assert th.sum(token_ids["ntype0"][3]==35) == 2 * max_sentence_len # ntype1: 11+4
        for i in range(20+12+4, 20+20+4):
            # ntype1: 12+4 ~ 19+4
            assert th.sum(token_ids["ntype0"][3]==i) == max_sentence_len

        # check mask
        assert th.sum(att_mask["ntype0"][3]) == \
            th.sum(node_data["ntype0"][ATT_MASK_IDX][3]).item() + \
            th.sum(node_data["ntype0"][ATT_MASK_IDX][9]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][4]).item() * 4+ \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][5]).item() * 3 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][6]).item() * 3 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][7]).item() * 3 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][8]).item() * 3 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][9]).item() * 3 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][10]).item() * 3 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][11]).item() * 3 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][12]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][13]).item() * 3 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][14]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][15]).item() * 2 + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][16]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][17]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][18]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][19]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][20]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][21]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][22]).item() + \
            th.sum(node_data["ntype1"][ATT_MASK_IDX][23]).item()
    check_overall_statistics_3(dfs_ordered_token_ids, dfs_ordered_atten_mask)
    check_overall_statistics_3(dfs_shuffled_token_ids, dfs_shuffled_atten_mask)

    with assert_raises(AssertionError):
        assert_equal(dfs_ordered_token_ids["ntype0"][0].numpy(),
                     dfs_shuffled_token_ids["ntype0"][0].numpy())
        assert_equal(dfs_ordered_atten_mask["ntype0"][0].numpy(),
                     dfs_shuffled_atten_mask["ntype0"][0].numpy())
        assert_equal(dfs_ordered_token_ids["ntype0"][1].numpy(),
                     dfs_shuffled_token_ids["ntype0"][1].numpy())
        assert_equal(dfs_ordered_atten_mask["ntype0"][1].numpy(),
                     dfs_shuffled_atten_mask["ntype0"][1].numpy())
        assert_equal(dfs_ordered_token_ids["ntype0"][2].numpy(),
                     dfs_shuffled_token_ids["ntype0"][2].numpy())
        assert_equal(dfs_ordered_atten_mask["ntype0"][2].numpy(),
                     dfs_shuffled_atten_mask["ntype0"][2].numpy())
        assert_equal(dfs_ordered_token_ids["ntype0"][3].numpy(),
                     dfs_shuffled_token_ids["ntype0"][3].numpy())
        assert_equal(dfs_ordered_atten_mask["ntype0"][3].numpy(),
                     dfs_shuffled_atten_mask["ntype0"][3].numpy())

    bfs_ordered_token_ids, bfs_ordered_atten_mask, \
        bfs_shuffled_token_ids, bfs_shuffled_atten_mask, \
        _, bfs_position_info = prepare_hat_node_centric(
            data, input_nodes, seeds, blocks, max_sentence_len, max_sequence_len,
            transverse_format=BFS_TRANSVERSE)

    # check position encoding first
    pos_info = bfs_position_info["ntype0"][0]
    assert th.sum(pos_info == 0).item() == 1
    assert th.sum(pos_info == 1).item() == 6
    assert th.sum(pos_info == 2).item() == 21

    pos_info = bfs_position_info["ntype0"][1]
    assert th.sum(pos_info == 0).item() == 1
    assert th.sum(pos_info == 1).item() == 5
    assert th.sum(pos_info == 2).item() == 18

    pos_info = bfs_position_info["ntype0"][2]
    assert th.sum(pos_info == 0).item() == 1
    assert th.sum(pos_info == 1).item() == 5
    assert th.sum(pos_info == 2).item() == 18

    pos_info = bfs_position_info["ntype0"][3]
    assert th.sum(pos_info == 0).item() == 1
    assert th.sum(pos_info == 1).item() == 9
    assert th.sum(pos_info == 2).item() == 34

    # seed node will be 0,1,2,3
    for i in range(4):
        assert_equal(bfs_ordered_token_ids["ntype0"][i][:max_sentence_len].numpy(),
                    node_data["ntype0"][TOKEN_IDX][i].numpy())
        assert_equal(bfs_shuffled_token_ids["ntype0"][i][:max_sentence_len].numpy(),
                    node_data["ntype0"][TOKEN_IDX][i].numpy())
        assert th.sum(bfs_ordered_atten_mask["ntype0"][i][:max_sentence_len]) == \
            node_data["ntype0"][ATT_MASK_IDX][i]

    # check first two layers
    assert_equal(
        bfs_ordered_token_ids["ntype0"][0][max_sentence_len:max_sentence_len*2].numpy(),
        node_data["ntype0"][TOKEN_IDX][5].numpy())
    assert th.sum(bfs_ordered_atten_mask["ntype0"][0][max_sentence_len:max_sentence_len*2]) == \
            node_data["ntype0"][ATT_MASK_IDX][5]
    assert_equal(
        bfs_ordered_token_ids["ntype0"][0][max_sentence_len*2:max_sentence_len*3].numpy(),
        node_data["ntype0"][TOKEN_IDX][6].numpy())
    assert th.sum(bfs_ordered_atten_mask["ntype0"][0][max_sentence_len*2:max_sentence_len*3]) == \
            node_data["ntype0"][ATT_MASK_IDX][6]
    assert_equal(
        bfs_ordered_token_ids["ntype0"][0][max_sentence_len*3:max_sentence_len*4].numpy(),
        node_data["ntype1"][TOKEN_IDX][4].numpy())
    assert th.sum(bfs_ordered_atten_mask["ntype0"][0][max_sentence_len*3:max_sentence_len*4]) == \
            node_data["ntype1"][ATT_MASK_IDX][4]
    assert_equal(
        bfs_ordered_token_ids["ntype0"][0][max_sentence_len*4:max_sentence_len*5].numpy(),
        node_data["ntype1"][TOKEN_IDX][5].numpy())
    assert th.sum(bfs_ordered_atten_mask["ntype0"][0][max_sentence_len*4:max_sentence_len*5]) == \
            node_data["ntype1"][ATT_MASK_IDX][5]
    assert_equal(
        bfs_ordered_token_ids["ntype0"][0][max_sentence_len*5:max_sentence_len*6].numpy(),
        node_data["ntype1"][TOKEN_IDX][6].numpy())
    assert th.sum(bfs_ordered_atten_mask["ntype0"][0][max_sentence_len*5:max_sentence_len*6]) == \
            node_data["ntype1"][ATT_MASK_IDX][6]
    assert_equal(
        bfs_ordered_token_ids["ntype0"][0][max_sentence_len*6:max_sentence_len*7].numpy(),
        node_data["ntype1"][TOKEN_IDX][7].numpy())
    assert th.sum(bfs_ordered_atten_mask["ntype0"][0][max_sentence_len*6:max_sentence_len*7]) == \
            node_data["ntype1"][ATT_MASK_IDX][7]

    check_overall_statistics_0(bfs_ordered_token_ids, bfs_ordered_atten_mask)
    check_overall_statistics_0(bfs_shuffled_token_ids, bfs_shuffled_atten_mask)
    check_overall_statistics_1(bfs_ordered_token_ids, bfs_ordered_atten_mask)
    check_overall_statistics_1(bfs_shuffled_token_ids, bfs_shuffled_atten_mask)
    check_overall_statistics_2(bfs_ordered_token_ids, bfs_ordered_atten_mask)
    check_overall_statistics_2(bfs_shuffled_token_ids, bfs_shuffled_atten_mask)
    check_overall_statistics_3(bfs_ordered_token_ids, bfs_ordered_atten_mask)
    check_overall_statistics_3(bfs_shuffled_token_ids, bfs_shuffled_atten_mask)

    max_sequence_len = 26 * max_sentence_len
    th.manual_seed(0)
    random.seed(0)
    dfs_ordered_token_ids_2, dfs_ordered_atten_mask_2, \
        dfs_shuffled_token_ids_2, dfs_shuffled_atten_mask_2, \
        _, dfs_position_info_2 = prepare_hat_node_centric(
            data, input_nodes, seeds, blocks, max_sentence_len, max_sequence_len)
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0]) != th.sum(dfs_ordered_atten_mask_2["ntype0"][0])
    assert th.sum(dfs_ordered_atten_mask["ntype0"][0]) != th.sum(dfs_shuffled_atten_mask_2["ntype0"][0])
    assert th.sum(dfs_ordered_atten_mask_2["ntype0"][0]) != th.sum(dfs_shuffled_atten_mask_2["ntype0"][0])
    with assert_raises(AssertionError):
        assert_equal(dfs_ordered_token_ids["ntype0"][0][:max_sequence_len].numpy(),
                     dfs_ordered_token_ids_2["ntype0"][0].numpy())
        assert_equal(dfs_ordered_token_ids_2["ntype0"][0].numpy(),
                     dfs_shuffled_token_ids_2["ntype0"][0].numpy())
        assert_equal(dfs_position_info["ntype0"][0][:26].numpy(),
                     dfs_position_info_2["ntype0"][0].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][1]) == th.sum(dfs_ordered_atten_mask_2["ntype0"][1])
    assert th.sum(dfs_ordered_atten_mask["ntype0"][1]) == th.sum(dfs_shuffled_atten_mask_2["ntype0"][1])
    assert_equal(dfs_ordered_token_ids["ntype0"][1][:max_sequence_len].numpy(),
                 dfs_ordered_token_ids_2["ntype0"][1].numpy())
    assert_equal(dfs_shuffled_token_ids["ntype0"][1][:max_sequence_len].numpy(),
                 dfs_shuffled_token_ids_2["ntype0"][1].numpy())
    assert_equal(dfs_position_info["ntype0"][1][:26].numpy(),
                 dfs_position_info_2["ntype0"][1].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][2]) == th.sum(dfs_ordered_atten_mask_2["ntype0"][2])
    assert th.sum(dfs_ordered_atten_mask["ntype0"][2]) == th.sum(dfs_shuffled_atten_mask_2["ntype0"][2])
    assert_equal(dfs_ordered_token_ids["ntype0"][2][:max_sequence_len].numpy(),
                 dfs_ordered_token_ids_2["ntype0"][2].numpy())
    assert_equal(dfs_shuffled_token_ids["ntype0"][2][:max_sequence_len].numpy(),
                 dfs_shuffled_token_ids_2["ntype0"][2].numpy())
    assert_equal(dfs_position_info["ntype0"][2][:26].numpy(),
                 dfs_position_info_2["ntype0"][2].numpy())
    assert th.sum(dfs_ordered_atten_mask["ntype0"][3]) != th.sum(dfs_ordered_atten_mask_2["ntype0"][3])
    assert th.sum(dfs_ordered_atten_mask["ntype0"][3]) != th.sum(dfs_shuffled_atten_mask_2["ntype0"][3])
    assert th.sum(dfs_ordered_atten_mask_2["ntype0"][3]) != th.sum(dfs_shuffled_atten_mask_2["ntype0"][3])
    with assert_raises(AssertionError):
        assert_equal(dfs_ordered_token_ids["ntype0"][3][:max_sequence_len].numpy(),
                     dfs_ordered_token_ids_2["ntype0"][3].numpy())
        assert_equal(dfs_ordered_token_ids_2["ntype0"][3].numpy(),
                     dfs_shuffled_token_ids_2["ntype0"][3].numpy())
        assert_equal(dfs_position_info["ntype0"][3][:26].numpy(),
                     dfs_position_info_2["ntype0"][3].numpy())

    bfs_ordered_token_ids_2, bfs_ordered_atten_mask_2, \
        bfs_shuffled_token_ids_2, bfs_shuffled_atten_mask_2, \
        _, bfs_position_info_2 = prepare_hat_node_centric(
            data, input_nodes, seeds, blocks, max_sentence_len, max_sequence_len,
            transverse_format=BFS_TRANSVERSE)
    assert th.sum(bfs_ordered_atten_mask["ntype0"][0]) != th.sum(bfs_ordered_atten_mask_2["ntype0"][0])
    assert th.sum(bfs_ordered_atten_mask["ntype0"][0]) != th.sum(bfs_shuffled_atten_mask_2["ntype0"][0])
    assert th.sum(bfs_ordered_atten_mask_2["ntype0"][0]) != th.sum(bfs_shuffled_atten_mask_2["ntype0"][0])
    with assert_raises(AssertionError):
        assert_equal(bfs_ordered_token_ids["ntype0"][0][:max_sequence_len].numpy(),
                     bfs_ordered_token_ids_2["ntype0"][0].numpy())
        assert_equal(bfs_ordered_token_ids_2["ntype0"][0].numpy(),
                     bfs_shuffled_token_ids_2["ntype0"][0].numpy())
        assert_equal(bfs_position_info["ntype0"][0][:26].numpy(),
                    bfs_position_info_2["ntype0"][0].numpy())
    assert th.sum(bfs_ordered_atten_mask["ntype0"][1]) == th.sum(bfs_ordered_atten_mask_2["ntype0"][1])
    assert th.sum(bfs_ordered_atten_mask["ntype0"][1]) == th.sum(bfs_shuffled_atten_mask_2["ntype0"][1])
    assert_equal(bfs_ordered_token_ids["ntype0"][1][:max_sequence_len].numpy(),
                 bfs_ordered_token_ids_2["ntype0"][1].numpy())
    assert_equal(bfs_shuffled_token_ids["ntype0"][1][:max_sequence_len].numpy(),
                 bfs_shuffled_token_ids_2["ntype0"][1].numpy())
    assert_equal(bfs_position_info["ntype0"][1][:26].numpy(),
                 bfs_position_info_2["ntype0"][1].numpy())
    assert th.sum(bfs_ordered_atten_mask["ntype0"][2]) == th.sum(bfs_ordered_atten_mask_2["ntype0"][2])
    assert th.sum(bfs_ordered_atten_mask["ntype0"][2]) == th.sum(bfs_shuffled_atten_mask_2["ntype0"][2])
    assert_equal(bfs_ordered_token_ids["ntype0"][2][:max_sequence_len].numpy(),
                 bfs_ordered_token_ids_2["ntype0"][2].numpy())
    assert_equal(bfs_shuffled_token_ids["ntype0"][2][:max_sequence_len].numpy(),
                 bfs_shuffled_token_ids_2["ntype0"][2].numpy())
    assert_equal(bfs_position_info["ntype0"][2][:26].numpy(),
                 bfs_position_info_2["ntype0"][2].numpy())
    assert th.sum(bfs_ordered_atten_mask["ntype0"][3]) != th.sum(bfs_ordered_atten_mask_2["ntype0"][3])
    assert th.sum(bfs_ordered_atten_mask["ntype0"][3]) != th.sum(bfs_shuffled_atten_mask_2["ntype0"][3])
    assert th.sum(bfs_ordered_atten_mask_2["ntype0"][3]) != th.sum(bfs_shuffled_atten_mask_2["ntype0"][3])
    with assert_raises(AssertionError):
        assert_equal(bfs_ordered_token_ids["ntype0"][3][:max_sequence_len].numpy(),
                     bfs_ordered_token_ids_2["ntype0"][3].numpy())
        assert_equal(bfs_ordered_token_ids_2["ntype0"][3].numpy(),
                     bfs_shuffled_token_ids_2["ntype0"][3].numpy())
        assert_equal(bfs_position_info["ntype0"][3][:26].numpy(),
                     bfs_position_info_2["ntype0"][3].numpy())

if __name__ == '__main__':
    test_get_prepare_lm_input()
    test_pad_seq()
    test_sequence_dfs2bfs()
    test_prepare_hat_node_centric_input()
