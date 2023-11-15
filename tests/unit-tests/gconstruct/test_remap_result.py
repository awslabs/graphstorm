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
import pytest
import os
import tempfile
import pytest

import torch as th
import numpy as np
from numpy.testing import assert_equal

from graphstorm.gconstruct import remap_result
from graphstorm.gconstruct.file_io import read_data_parquet
from graphstorm.gconstruct.id_map import IdMap, IdReverseMap
from graphstorm.gconstruct.remap_result import _get_file_range
from graphstorm.gconstruct.remap_result import (worker_remap_edge_pred,
                                                worker_remap_node_data)

def gen_id_maps(num_ids=1000):
    nid0 = np.random.permutation(num_ids).tolist()
    nid0 = np.array([f"n0_{i}" for i in nid0])
    map0 = IdMap(nid0)

    nid1 = np.random.permutation(num_ids).tolist()
    nid1 = np.array([f"n1_{i}" for i in nid1])
    map1 = IdMap(nid1)

    return {"n0": map0,
            "n1": map1}

def gen_edge_preds(num_ids=1000, num_preds=2000):
    pred = th.rand((num_preds, 10))
    src_nids = th.randint(num_ids, (num_preds,))
    dst_nids = th.randint(num_ids, (num_preds,))

    return pred, src_nids, dst_nids

def gen_node_data(num_ids=1000, num_preds=2000):
    data = th.rand((num_preds, 10))
    nids = th.randint(num_ids, (num_preds,))

    return data, nids

@pytest.mark.parametrize("data_col", ["pred", "emb"])
def test_worker_remap_node_data(data_col):
    with tempfile.TemporaryDirectory() as tmpdirname:
        num_ids = 1000
        num_data = 1000
        mappings = gen_id_maps(num_ids)
        map_files = {}
        ntypes = []
        for ntype, map in mappings.items():
            map_files[ntype] = map.save(os.path.join(tmpdirname, ntype + "_id_remap.parquet"))
            ntypes.append(ntype)

        data, nids = gen_node_data(num_ids, num_data)
        data_path = os.path.join(tmpdirname, f"{data_col}-00000.pt")
        nid_path = os.path.join(tmpdirname, "nid-00000.pt")
        output_path_prefix = os.path.join(tmpdirname, f"out-{data_col}")
        th.save(data, data_path)
        th.save(nids, nid_path)
        chunk_size = 256

        for ntype in ntypes:
            remap_result.id_maps[ntype] = IdReverseMap(os.path.join(tmpdirname, ntype + "_id_remap.parquet"))

        worker_remap_node_data(data_path, nid_path, ntypes[0], data_col,
                               output_path_prefix, chunk_size, preserve_input=True)
        assert os.path.exists(f"{output_path_prefix}_00000.parquet")
        assert os.path.exists(f"{output_path_prefix}_00001.parquet")
        assert os.path.exists(f"{output_path_prefix}_00002.parquet")
        assert os.path.exists(f"{output_path_prefix}_00003.parquet")

        data0 = read_data_parquet(f"{output_path_prefix}_00000.parquet",
                                  [data_col, "nid"])
        data1 = read_data_parquet(f"{output_path_prefix}_00001.parquet",
                                  [data_col, "nid"])
        data2 = read_data_parquet(f"{output_path_prefix}_00002.parquet",
                                  [data_col, "nid"])
        data3 = read_data_parquet(f"{output_path_prefix}_00003.parquet",
                                  [data_col, "nid"])
        assert len(data0[data_col]) == 256
        assert len(data1[data_col]) == 256
        assert len(data2[data_col]) == 256
        assert len(data3[data_col]) == 232

        data_ = [data0[data_col], data1[data_col], data2[data_col], data3[data_col]]
        nids_ = [data0["nid"], data1["nid"], data2["nid"], data3["nid"]]

        data_ = np.concatenate(data_, axis=0)
        nids_ = np.concatenate(nids_, axis=0)
        revserse_mapping = {}
        revserse_mapping[ntypes[0]] = {val: key for key, val in mappings[ntypes[0]]._ids.items()}

        for i in range(num_data):
            assert_equal(data_[i], data[i].numpy())
            assert_equal(nids_[i], revserse_mapping[ntypes[0]][int(nids[i])])

def test_worker_remap_edge_pred():
    with tempfile.TemporaryDirectory() as tmpdirname:
        num_ids = 1000
        num_preds = 1000
        mappings = gen_id_maps(num_ids)
        map_files = {}
        ntypes = []
        for ntype, map in mappings.items():
            map_files[ntype] = map.save(os.path.join(tmpdirname, ntype + "_id_remap.parquet"))
            ntypes.append(ntype)
        preds, src_nids, dst_nids = gen_edge_preds(num_ids, num_preds)
        pred_path = os.path.join(tmpdirname, "pred-00000.pt")
        src_nid_path = os.path.join(tmpdirname, "src-nid-00000.pt")
        dst_nid_path = os.path.join(tmpdirname, "dst-nid-00000.pt")
        output_path_prefix = os.path.join(tmpdirname, "out-pred")
        th.save(preds, pred_path)
        th.save(src_nids, src_nid_path)
        th.save(dst_nids, dst_nid_path)
        chunk_size = 256

        for ntype in ntypes:
            remap_result.id_maps[ntype] = IdReverseMap(os.path.join(tmpdirname, ntype + "_id_remap.parquet"))

        worker_remap_edge_pred(pred_path, src_nid_path, dst_nid_path,
                               ntypes[0], ntypes[1], output_path_prefix,
                               chunk_size, preserve_input=True)

        assert os.path.exists(f"{output_path_prefix}_00000.parquet")
        assert os.path.exists(f"{output_path_prefix}_00001.parquet")
        assert os.path.exists(f"{output_path_prefix}_00002.parquet")
        assert os.path.exists(f"{output_path_prefix}_00003.parquet")
        data0 = read_data_parquet(f"{output_path_prefix}_00000.parquet",
                                  ["pred", "src_nid", "dst_nid"])
        data1 = read_data_parquet(f"{output_path_prefix}_00001.parquet",
                                  ["pred", "src_nid", "dst_nid"])
        data2 = read_data_parquet(f"{output_path_prefix}_00002.parquet",
                                  ["pred", "src_nid", "dst_nid"])
        data3 = read_data_parquet(f"{output_path_prefix}_00003.parquet",
                                  ["pred", "src_nid", "dst_nid"])
        assert len(data0["pred"]) == 256
        assert len(data1["pred"]) == 256
        assert len(data2["pred"]) == 256
        assert len(data3["pred"]) == 232
        preds_ = [data0["pred"], data1["pred"], data2["pred"], data3["pred"]]
        src_nids_ = [data0["src_nid"], data1["src_nid"], data2["src_nid"], data3["src_nid"]]
        dst_nids_ = [data0["dst_nid"], data1["dst_nid"], data2["dst_nid"], data3["dst_nid"]]
        preds_ = np.concatenate(preds_, axis=0)
        src_nids_ = np.concatenate(src_nids_, axis=0)
        dst_nids_ = np.concatenate(dst_nids_, axis=0)
        revserse_mapping = {}
        revserse_mapping[ntypes[0]] = {val: key for key, val in mappings[ntypes[0]]._ids.items()}
        revserse_mapping[ntypes[1]] = {val: key for key, val in mappings[ntypes[1]]._ids.items()}

        for i in range(num_preds):
            assert_equal(preds_[i], preds[i].numpy())
            assert_equal(src_nids_[i], revserse_mapping[ntypes[0]][int(src_nids[i])])
            assert_equal(dst_nids_[i], revserse_mapping[ntypes[1]][int(dst_nids[i])])

def test__get_file_range():
    start, end = _get_file_range(10, 0, 0)
    assert start == 0
    assert end == 10

    start, end = _get_file_range(10, 0, 1)
    assert start == 0
    assert end == 10

    start, end = _get_file_range(10, 0, 2)
    assert start == 0
    assert end == 5
    start, end = _get_file_range(10, 1, 2)
    assert start == 5
    assert end == 10

    start, end = _get_file_range(10, 0, 3)
    assert start == 0
    assert end == 3
    start, end = _get_file_range(10, 1, 3)
    assert start == 3
    assert end == 6
    start, end = _get_file_range(10, 2, 3)
    assert start == 6
    assert end == 10

    start, end = _get_file_range(10, 0, 4)
    assert start == 0
    assert end == 2
    start, end = _get_file_range(10, 1, 4)
    assert start == 2
    assert end == 4
    start, end = _get_file_range(10, 2, 4)
    assert start == 4
    assert end == 7
    start, end = _get_file_range(10, 3, 4)
    assert start == 7
    assert end == 10

if __name__ == '__main__':
    test__get_file_range()
    test_worker_remap_edge_pred()
    test_worker_remap_node_data("pred")
