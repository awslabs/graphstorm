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
import json

import dgl
import numpy as np
import torch as th
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from numpy.testing import assert_almost_equal

from graphstorm.gconstruct.utils import _estimate_sizeof, _to_numpy_array, _to_shared_memory
from graphstorm.gconstruct.utils import HDF5Array, ExtNumpyWrapper
from graphstorm.gconstruct.utils import convert_to_ext_mem_numpy, _to_ext_memory
from graphstorm.gconstruct.utils import multiprocessing_data_read
from graphstorm.gconstruct.utils import (save_maps,
                                         load_maps,
                                         get_hard_edge_negs_feats,
                                         shuffle_hard_nids)
from graphstorm.gconstruct.file_io import (write_data_hdf5,
                                           read_data_hdf5,
                                           get_in_files,
                                           write_data_parquet)
from graphstorm.gconstruct.file_io import read_index, write_index_json
from graphstorm.gconstruct.file_io import (read_data_csv,
                                           read_data_json,
                                           read_data_parquet)
from graphstorm.gconstruct.transform import HardEdgeDstNegativeTransform

def gen_data():
    data_th = th.zeros((1024, 16), dtype=th.float32)
    data_np = np.ndarray(shape=(1024, 16), dtype=np.single)

    data_dict = {
        "th": data_th,
        "np": data_np
    }

    data_list = [data_th, data_np]
    data_tuple = (data_th, data_np)

    data_mixed = {
        "list": [data_dict, data_dict],
        "tuple": (data_list, data_tuple),
    }

    return data_th, data_np, data_dict, data_list, data_tuple, data_mixed

def test_estimate_sizeof():
    data_th, data_np, data_dict, data_list, data_tuple, data_mixed = gen_data()
    # object is torch tensor
    data_th_size = data_th.nelement() * 4
    assert _estimate_sizeof(data_th) == data_th_size

    # object is numpy array
    data_np_size = data_np.size * 4
    assert _estimate_sizeof(data_np) == data_np_size

    # object is a dict
    data_dict_size = data_th_size + data_np_size
    assert _estimate_sizeof(data_dict) == data_dict_size

    # object is a list
    data_list_size = data_th_size + data_np_size
    assert _estimate_sizeof(data_list) == data_list_size

    # object is a tuple
    data_tuple_size = data_th_size + data_np_size
    assert _estimate_sizeof(data_tuple) == data_tuple_size

    # object is recursive obj
    data_mixed_size = data_dict_size * 2 + data_list_size + data_tuple_size
    assert _estimate_sizeof(data_mixed) == data_mixed_size

def test_object_conversion():
    data_th, data_np, data_dict, data_list, data_tuple, data_mixed = gen_data()

    def check_is_shared(data):
        assert th.is_tensor(data)
        assert data.is_shared()
    def check_is_numpy(data):
        assert isinstance(data, np.ndarray)

    new_data = _to_shared_memory(data_th)
    check_is_shared(new_data)
    new_data = _to_numpy_array(new_data)
    check_is_numpy(new_data)

    new_data = _to_shared_memory(data_np)
    check_is_shared(new_data)
    new_data = _to_numpy_array(new_data)
    check_is_numpy(new_data)

    new_data = _to_shared_memory(data_dict)
    assert isinstance(new_data, dict)
    check_is_shared(new_data["th"])
    check_is_shared(new_data["np"])
    new_data = _to_numpy_array(new_data)
    assert isinstance(new_data, dict)
    check_is_numpy(new_data["th"])
    check_is_numpy(new_data["np"])

    new_data = _to_shared_memory(data_list)
    assert isinstance(new_data, list)
    check_is_shared(new_data[0])
    check_is_shared(new_data[1])
    new_data = _to_numpy_array(new_data)
    assert isinstance(new_data, list)
    check_is_numpy(new_data[0])
    check_is_numpy(new_data[1])

    new_data = _to_shared_memory(data_tuple)
    assert isinstance(new_data, tuple)
    check_is_shared(new_data[0])
    check_is_shared(new_data[1])
    new_data = _to_numpy_array(new_data)
    assert isinstance(new_data, tuple)
    check_is_numpy(new_data[0])
    check_is_numpy(new_data[1])

    new_data = _to_shared_memory(data_mixed)
    assert isinstance(new_data, dict)
    assert isinstance(new_data["list"], list)
    assert isinstance(new_data["list"][0], dict)
    check_is_shared(new_data["list"][0]["th"])
    check_is_shared(new_data["list"][0]["np"])
    assert isinstance(new_data["list"][1], dict)
    check_is_shared(new_data["list"][1]["th"])
    check_is_shared(new_data["list"][1]["np"])
    assert isinstance(new_data["tuple"], tuple)
    assert isinstance(new_data["tuple"][0], list)
    check_is_shared(new_data["tuple"][0][0])
    check_is_shared(new_data["tuple"][0][1])
    assert isinstance(new_data["tuple"][1], tuple)
    check_is_shared(new_data["tuple"][0][0])
    check_is_shared(new_data["tuple"][0][1])
    new_data = _to_numpy_array(new_data)
    assert isinstance(new_data, dict)
    assert isinstance(new_data["list"], list)
    assert isinstance(new_data["list"][0], dict)
    check_is_numpy(new_data["list"][0]["th"])
    check_is_numpy(new_data["list"][0]["np"])
    assert isinstance(new_data["list"][1], dict)
    check_is_numpy(new_data["list"][1]["th"])
    check_is_numpy(new_data["list"][1]["np"])
    assert isinstance(new_data["tuple"], tuple)
    assert isinstance(new_data["tuple"][0], list)
    check_is_numpy(new_data["tuple"][0][0])
    check_is_numpy(new_data["tuple"][0][1])
    assert isinstance(new_data["tuple"][1], tuple)
    check_is_numpy(new_data["tuple"][0][0])
    check_is_numpy(new_data["tuple"][0][1])

def check_ext_mem_array(arr, orig_arr):
    assert len(arr) == orig_arr.shape[0]
    assert arr.shape == orig_arr.shape
    assert arr.dtype == orig_arr.dtype
    idx = np.array([1, 3, 4])
    assert np.all(arr[idx] == orig_arr[idx])
    assert np.all(arr.to_numpy() == orig_arr)

    new_arr = arr.astype(np.float16)
    assert arr.dtype == np.float32
    assert new_arr.dtype == np.float16
    assert new_arr[idx].dtype == np.float16

def test_ext_mem_array():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data = np.random.uniform(size=(1000, 10)).astype(np.float32)
        tensor_path = os.path.join(tmpdirname, "tmp1.npy")
        check_ext_mem_array(convert_to_ext_mem_numpy(tensor_path, data), data)

        data1 = np.random.uniform(size=(1000, 10)).astype(np.float32)
        data2 = np.random.uniform(size=(1000,)).astype(np.float32)
        data3 = np.random.uniform(size=(1000, 10)).astype(np.float32)
        data4 = np.random.uniform(size=(1000,)).astype(np.float32)
        data5 = np.random.uniform(size=(1000,)).astype(str)
        arr_dict = {
                "test1": (data1, data2),
                "test2": [data3, data4, data5],
        }
        arr_dict1 = _to_ext_memory(None, arr_dict, tmpdirname)
        assert isinstance(arr_dict1, dict)
        assert "test1" in arr_dict1
        assert "test2" in arr_dict1
        assert isinstance(arr_dict1["test1"], tuple)
        assert isinstance(arr_dict1["test2"], list)
        assert isinstance(arr_dict1["test1"][0], ExtNumpyWrapper)
        assert isinstance(arr_dict1["test1"][1], ExtNumpyWrapper)
        assert isinstance(arr_dict1["test2"][0], ExtNumpyWrapper)
        assert isinstance(arr_dict1["test2"][1], ExtNumpyWrapper)
        assert isinstance(arr_dict1["test2"][2], np.ndarray)
        assert np.all(arr_dict1["test1"][0].to_numpy() == data1)
        assert np.all(arr_dict1["test1"][1].to_numpy() == data2)
        assert np.all(arr_dict1["test2"][0].to_numpy() == data3)
        assert np.all(arr_dict1["test2"][1].to_numpy() == data4)

        tensor_path = os.path.join(tmpdirname, "tmp2.hdf5")
        write_data_hdf5({"test": data}, tensor_path)
        data1 = read_data_hdf5(tensor_path, in_mem=False)
        check_ext_mem_array(data1['test'], data)

def dummy_read(in_file):
    assert False

def test_multiprocessing_read():
    try:
        multiprocessing_data_read([str(i) for i in range(10)], 2, dummy_read)
    except RuntimeError as e:
        print(e)
        return
    assert False

def test_read_empty_parquet():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_file = os.path.join(tmpdirname, "test.parquet")
        fields = ["a", "b"]
        empty_df = pd.DataFrame(columns=fields)
        empty_table = pa.Table.from_pandas(empty_df)
        pq.write_table(empty_table, data_file)

        pass_test = False
        try:
            read_data_parquet(data_file, fields)
        except:
            pass_test = True
        assert pass_test

def test_read_empty_csv():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_file = os.path.join(tmpdirname, "test.parquet")
        fields = ["a", "b"]
        empty_df = pd.DataFrame(columns=fields)
        empty_df.to_csv(data_file, index=True, sep=",")

        pass_test = False
        try:
            read_data_csv(data_file, fields, ",")
        except:
            pass_test = True
        assert pass_test

def test_read_empty_json():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_file = os.path.join(tmpdirname, "test.json")
        data = {}
        with open(data_file, 'w', encoding="utf8") as json_file:
            json.dump(data, json_file)

        pass_test = False
        try:
            read_data_json(data_file)
        except:
            pass_test = True
        assert pass_test

def test_read_empty_parquet():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_file = os.path.join(tmpdirname, "test.parquet")
        fields = ["a", "b"]
        empty_df = pd.DataFrame(columns=fields)
        empty_table = pa.Table.from_pandas(empty_df)
        pq.write_table(empty_table, data_file)

        pass_test = False
        try:
            read_data_parquet(data_file, fields)
        except:
            pass_test = True
        assert pass_test

def test_get_in_files():
    with tempfile.TemporaryDirectory() as tmpdirname:
        files = [os.path.join(tmpdirname, f"test{i}.parquet") for i in range(10)]
        for i in range(10):
            data = {"test": np.random.rand(10)}
            write_data_parquet(data, files[i])

        in_files = get_in_files(os.path.join(tmpdirname,"*.parquet"))
        assert len(in_files) == 10
        files.sort()
        assert files == in_files

        in_files = get_in_files(os.path.join(tmpdirname,"test9.parquet"))
        assert len(in_files) == 1
        assert os.path.join(tmpdirname,"test9.parquet") == in_files[0]

        pass_test = False
        try:
            in_files = get_in_files(os.path.join(tmpdirname,"test10.parquet"))
        except:
            pass_test = True
        assert pass_test

def test_get_hard_edge_negs_feats():
    hard_trans0 = HardEdgeDstNegativeTransform("hard_neg", "hard_neg")
    hard_trans0.set_target_etype(("src", "rel0", "dst"))

    hard_trans1 = HardEdgeDstNegativeTransform("hard_neg", "hard_neg1")
    hard_trans1.set_target_etype(("src", "rel0", "dst"))

    hard_trans2 = HardEdgeDstNegativeTransform("hard_neg", "hard_neg")
    hard_trans2.set_target_etype(("src", "rel1", "dst"))

    hard_edge_neg_feats = get_hard_edge_negs_feats([hard_trans0, hard_trans1, hard_trans2])
    assert len(hard_edge_neg_feats) == 2
    assert len(hard_edge_neg_feats[("src", "rel0", "dst")]) == 1
    assert len(hard_edge_neg_feats[("src", "rel0", "dst")]["dst"]) == 2
    assert set(hard_edge_neg_feats[("src", "rel0", "dst")]["dst"]) == set(["hard_neg", "hard_neg1"])
    assert len(hard_edge_neg_feats[("src", "rel1", "dst")]) == 1
    assert len(hard_edge_neg_feats[("src", "rel1", "dst")]["dst"]) == 1

def test_save_load_maps():
    with tempfile.TemporaryDirectory() as tmpdirname:
        map_data = {"a": th.randint(100, (10,)),
                    "b": th.randint(100, (10,))}
        save_maps(tmpdirname, "node_mapping", map_data)
        node_mapping = load_maps(tmpdirname, "node_mapping")
        assert_almost_equal(node_mapping["a"].numpy(), map_data["a"].numpy())
        assert_almost_equal(node_mapping["b"].numpy(), map_data["b"].numpy())

def test_shuffle_hard_nids():
    with tempfile.TemporaryDirectory() as tmpdirname:
        num_parts = 2
        # generate node_mapping
        node_mapping = {
            "dst0" : 99 - th.arange(100),
            "dst1" : 199 - th.arange(200)
        }
        save_maps(tmpdirname, "node_mapping", node_mapping)

        reverse_map_dst0 = {gid: i for i, gid in enumerate(node_mapping["dst0"].tolist())}
        reverse_map_dst0[-1] = -1
        reverse_map_dst1 = {gid: i for i, gid in enumerate(node_mapping["dst1"].tolist())}
        reverse_map_dst1[-1] = -1

        static_feat = th.rand((100, 10))
        # generate edge features
        etype0 = ("src", "rel0", "dst0")
        etype1 = ("src", "rel0", "dst1")
        p0_etype0_neg0 = th.randint(100, (100, 5))
        p0_etype0_neg1 = th.randint(100, (100, 10))
        p0_etype1_neg0 = th.randint(100, (100, 5))

        p0_etype0_neg0_shuffled = [
            [reverse_map_dst0[nid] for nid in negs] \
                for negs in p0_etype0_neg0.tolist()]
        p0_etype0_neg0_shuffled = np.array(p0_etype0_neg0_shuffled)
        p0_etype0_neg1_shuffled = [
            [reverse_map_dst0[nid] for nid in negs] \
                for negs in p0_etype0_neg1.tolist()]
        p0_etype0_neg1_shuffled = np.array(p0_etype0_neg1_shuffled)

        p0_etype1_neg0_shuffled = [
            [reverse_map_dst1[nid] for nid in negs] \
                for negs in p0_etype1_neg0.tolist()]
        p0_etype1_neg0_shuffled = np.array(p0_etype1_neg0_shuffled)

        edge_feats_part0 = {
            ":".join(etype0)+"/static_feat": static_feat,
            ":".join(etype0)+"/hard_neg0": p0_etype0_neg0,
            ":".join(etype0)+"/hard_neg1": p0_etype0_neg1,
            ":".join(etype1)+"/hard_neg0": p0_etype1_neg0
        }
        part_path = os.path.join(tmpdirname, f"part{0}")
        os.mkdir(part_path)
        edge_feat_path = os.path.join(part_path, "edge_feat.dgl")
        dgl.data.utils.save_tensors(edge_feat_path, edge_feats_part0)


        p1_etype0_neg0 = th.randint(100, (100, 5))
        p1_etype0_neg1 = th.randint(100, (100, 10))
        p1_etype1_neg0 = th.randint(100, (100, 5))
        p1_etype0_neg1[:,-2:] = -1
        p1_etype1_neg0[0][-1] = -1

        p1_etype0_neg0_shuffled = [
            [reverse_map_dst0[nid] for nid in negs] \
                for negs in p1_etype0_neg0.tolist()]
        p1_etype0_neg0_shuffled = np.array(p1_etype0_neg0_shuffled)

        p1_etype0_neg1_shuffled = [
            [reverse_map_dst0[nid] for nid in negs ] \
                for negs in p1_etype0_neg1.tolist()]
        p1_etype0_neg1_shuffled = np.array(p1_etype0_neg1_shuffled)

        p1_etype1_neg0_shuffled = [
            [reverse_map_dst1[nid] for nid in negs] \
                for negs in p1_etype1_neg0.tolist()]
        p1_etype1_neg0_shuffled = np.array(p1_etype1_neg0_shuffled)

        edge_feats_part1 = {
            ":".join(etype0)+"/static_feat": static_feat,
            ":".join(etype0)+"/hard_neg0": p1_etype0_neg0,
            ":".join(etype0)+"/hard_neg1": p1_etype0_neg1,
            ":".join(etype1)+"/hard_neg0": p1_etype1_neg0
        }

        part_path = os.path.join(tmpdirname, f"part{1}")
        os.mkdir(part_path)
        edge_feat_path = os.path.join(part_path, "edge_feat.dgl")
        dgl.data.utils.save_tensors(edge_feat_path, edge_feats_part1)

        hard_edge_neg_feats = {
            etype0: {"dst0": ["hard_neg0", "hard_neg1"]},
            etype1: {"dst1": ["hard_neg0"]}
        }
        # test
        shuffle_hard_nids(tmpdirname, 2, hard_edge_neg_feats)

        part_path = os.path.join(tmpdirname, f"part{0}")
        edge_feat_path = os.path.join(part_path, "edge_feat.dgl")
        p0_new_edge_feats = dgl.data.utils.load_tensors(edge_feat_path)

        assert_almost_equal(p0_new_edge_feats[":".join(etype0)+"/static_feat"].numpy(),
                            static_feat.numpy())
        assert_almost_equal(p0_new_edge_feats[":".join(etype0)+"/hard_neg0"].numpy(),
                            p0_etype0_neg0_shuffled)
        assert_almost_equal(p0_new_edge_feats[":".join(etype0)+"/hard_neg1"].numpy(),
                            p0_etype0_neg1_shuffled)
        assert_almost_equal(p0_new_edge_feats[":".join(etype1)+"/hard_neg0"].numpy(),
                            p0_etype1_neg0_shuffled)

        part_path = os.path.join(tmpdirname, f"part{1}")
        edge_feat_path = os.path.join(part_path, "edge_feat.dgl")
        p1_new_edge_feats = dgl.data.utils.load_tensors(edge_feat_path)
        assert_almost_equal(p1_new_edge_feats[":".join(etype0)+"/static_feat"].numpy(),
                            static_feat.numpy())
        assert_almost_equal(p1_new_edge_feats[":".join(etype0)+"/hard_neg0"].numpy(),
                            p1_etype0_neg0_shuffled)
        assert_almost_equal(p1_new_edge_feats[":".join(etype0)+"/hard_neg1"].numpy(),
                            p1_etype0_neg1_shuffled)
        assert_almost_equal(p1_new_edge_feats[":".join(etype1)+"/hard_neg0"].numpy(),
                            p1_etype1_neg0_shuffled)


def test_read_index():
    write_index_json([(3, 3)], "/tmp/test_idx.json")
    split_info = {"valid": "/tmp/test_idx.json"}
    _, json_content, _ = read_index(split_info)
    assert json_content == [(3, 3)]

    write_index_json(np.arange(3), "/tmp/test_idx.json")
    split_info = {"train": "/tmp/test_idx.json"}
    json_content, _, _ = read_index(split_info)
    assert json_content == [0, 1, 2]

    data = ["p70", "p71", "p72", "p73", "p74", "p75"]
    df = pd.DataFrame(data, columns=['ID'])
    df.to_parquet('/tmp/test_idx.parquet')
    split_info = {"train": "/tmp/test_idx.parquet", "column": ["ID"]}
    parquet_content, _, _ = read_index(split_info)
    assert np.array_equal(parquet_content, data)

    data_multi = ["p702", "p712", "p722", "p732", "p742", "p752"]
    df = pd.DataFrame(data_multi, columns=['ID'])
    # test with wildcard
    df.to_parquet('/tmp/test_idx_multi.parquet')
    split_info = {"train": "/tmp/test_idx*.parquet", "column": ["ID"]}
    parquet_content, _, _ = read_index(split_info)
    assert np.array_equal(parquet_content, data + data_multi)
    # test with list input
    split_info = {"train": ["/tmp/test_idx.parquet", "/tmp/test_idx_multi.parquet"],
                  "column": ["ID"]}
    parquet_content, _, _ = read_index(split_info)
    assert np.array_equal(parquet_content, data + data_multi)

    data, data2 = ["p1", "p2"], ["p3", "p4"]
    df = pd.DataFrame({'src': data, 'dst': data2})
    df.to_parquet('/tmp/train_idx.parquet')
    split_info = {"train": "/tmp/train_idx.parquet", "column": ["src", "dst"]}
    parquet_content, _, _ = read_index(split_info)
    assert parquet_content == [("p1", "p3"), ("p2", "p4")]

    data3, data4 = ["p5", "p6"], ["p7", "p8"]
    df = pd.DataFrame({'src': data3, 'dst': data4})
    df.to_parquet('/tmp/test_idx.parquet')
    split_info = {"train": "/tmp/train_idx.parquet",
                  "test": "/tmp/test_idx.parquet", "column": ["src", "dst"]}
    train_content, _, test_content = read_index(split_info)
    assert train_content == [("p1", "p3"), ("p2", "p4")]
    assert test_content == [("p5", "p7"), ("p6", "p8")]



if __name__ == '__main__':
    test_shuffle_hard_nids()
    test_save_load_maps()
    test_get_hard_edge_negs_feats()
    test_get_in_files()
    test_read_empty_parquet()
    test_read_empty_json()
    test_read_empty_csv()
    test_estimate_sizeof()
    test_object_conversion()
    test_ext_mem_array()
    test_multiprocessing_read()
    test_read_index()