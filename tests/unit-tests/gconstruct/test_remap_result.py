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
import argparse
import tempfile
import pytest
from functools import partial
from pathlib import Path

import pandas as pd
import torch as th
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from graphstorm.config import GSConfig
from graphstorm.config.config import get_mttask_id
from graphstorm.config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                               BUILTIN_TASK_NODE_REGRESSION,
                               BUILTIN_TASK_EDGE_CLASSIFICATION,
                               BUILTIN_TASK_EDGE_REGRESSION,
                               BUILTIN_TASK_LINK_PREDICTION,
                               BUILTIN_TASK_RECONSTRUCT_NODE_FEAT)
from graphstorm.gconstruct import remap_result
from graphstorm.gconstruct.file_io import read_data_parquet
from graphstorm.gconstruct.id_map import IdMap, IdReverseMap
from graphstorm.gconstruct.remap_result import (_get_file_range,
                                                _parse_gs_config)
from graphstorm.gconstruct.remap_result import (worker_remap_edge_pred,
                                                worker_remap_node_data)
from graphstorm.gconstruct.remap_result import (write_data_parquet_file,
                                                write_data_csv_file)

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
        ntypes = []
        for ntype, id_map in mappings.items():
            id_map.save(os.path.join(tmpdirname, ntype))
            ntypes.append(ntype)

        data, nids = gen_node_data(num_ids, num_data)
        data_path = os.path.join(tmpdirname, f"{data_col}-00000.pt")
        nid_path = os.path.join(tmpdirname, "nid-00000.pt")
        output_path_prefix = os.path.join(tmpdirname, f"out-{data_col}")
        th.save(data, data_path)
        th.save(nids, nid_path)
        chunk_size = 256

        for ntype in ntypes:
            remap_result.id_maps[ntype] = IdReverseMap(os.path.join(tmpdirname, ntype))

        worker_remap_node_data(data_path, nid_path, ntypes[0], data_col,
                               output_path_prefix, chunk_size,
                               write_data_parquet_file)
        worker_remap_node_data(data_path, nid_path, ntypes[0], data_col,
                               output_path_prefix, chunk_size,
                               partial(write_data_csv_file, delimiter=","))
        def read_csv(file, delimiter=","):
            data = pd.read_csv(file, delimiter=delimiter)
            nid = data["nid"].to_numpy()
            data_ = [np.array(d.split(";")).astype(np.float32) for d in data[data_col]]

            return {"nid": nid,
                    data_col: data_}

        assert os.path.exists(f"{output_path_prefix}_00000.parquet")
        assert os.path.exists(f"{output_path_prefix}_00001.parquet")
        assert os.path.exists(f"{output_path_prefix}_00002.parquet")
        assert os.path.exists(f"{output_path_prefix}_00003.parquet")
        assert os.path.exists(f"{output_path_prefix}_00000.csv")
        assert os.path.exists(f"{output_path_prefix}_00001.csv")
        assert os.path.exists(f"{output_path_prefix}_00002.csv")
        assert os.path.exists(f"{output_path_prefix}_00003.csv")

        data0 = read_data_parquet(f"{output_path_prefix}_00000.parquet",
                                  [data_col, "nid"])
        data1 = read_data_parquet(f"{output_path_prefix}_00001.parquet",
                                  [data_col, "nid"])
        data2 = read_data_parquet(f"{output_path_prefix}_00002.parquet",
                                  [data_col, "nid"])
        data3 = read_data_parquet(f"{output_path_prefix}_00003.parquet",
                                  [data_col, "nid"])
        data0_csv = read_csv(f"{output_path_prefix}_00000.csv")
        data1_csv = read_csv(f"{output_path_prefix}_00001.csv")
        data2_csv = read_csv(f"{output_path_prefix}_00002.csv")
        data3_csv = read_csv(f"{output_path_prefix}_00003.csv")
        assert len(data0[data_col]) == 256
        assert len(data1[data_col]) == 256
        assert len(data2[data_col]) == 256
        assert len(data3[data_col]) == 232
        assert len(data0_csv[data_col]) == 256
        assert len(data1_csv[data_col]) == 256
        assert len(data2_csv[data_col]) == 256
        assert len(data3_csv[data_col]) == 232

        data_ = [data0[data_col], data1[data_col], data2[data_col], data3[data_col]]
        nids_ = [data0["nid"], data1["nid"], data2["nid"], data3["nid"]]
        csv_data_ = [data0_csv[data_col], data1_csv[data_col],
                     data2_csv[data_col], data3_csv[data_col]]
        csv_nids_ = [data0_csv["nid"], data1_csv["nid"],
                     data2_csv["nid"], data3_csv["nid"]]

        data_ = np.concatenate(data_, axis=0)
        nids_ = np.concatenate(nids_, axis=0)
        csv_data_ = np.concatenate(csv_data_, axis=0)
        csv_nids_ = np.concatenate(csv_nids_, axis=0)
        assert_almost_equal(data_, csv_data_, decimal=5)
        assert_equal(nids_, csv_nids_)
        revserse_mapping = {}
        revserse_mapping[ntypes[0]] = {val: key for key, val in mappings[ntypes[0]]._ids.items()}

        for i in range(num_data):
            assert_almost_equal(data_[i], data[i].numpy(), decimal=4)
            assert_equal(nids_[i], revserse_mapping[ntypes[0]][int(nids[i])])

def test_worker_remap_edge_pred():
    with tempfile.TemporaryDirectory() as tmpdirname:
        num_ids = 1000
        num_preds = 1000
        mappings = gen_id_maps(num_ids)
        ntypes = []
        for ntype, map in mappings.items():
            map.save(os.path.join(tmpdirname, ntype))
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
            remap_result.id_maps[ntype] = IdReverseMap(os.path.join(tmpdirname, ntype))

        worker_remap_edge_pred(pred_path, src_nid_path, dst_nid_path,
                               ntypes[0], ntypes[1], output_path_prefix,
                               chunk_size, write_data_parquet_file)
        worker_remap_edge_pred(pred_path, src_nid_path, dst_nid_path,
                               ntypes[0], ntypes[1], output_path_prefix,
                               chunk_size, partial(write_data_csv_file, delimiter=","))
        def read_csv(file, delimiter=","):
            data = pd.read_csv(file, delimiter=delimiter)
            src_nid = data["src_nid"].to_numpy()
            dst_nid = data["dst_nid"].to_numpy()
            pred = [np.array(p.split(";")).astype(np.float32) for p in data["pred"]]

            return {"src_nid": src_nid,
                    "dst_nid": dst_nid,
                    "pred": pred}

        assert os.path.exists(f"{output_path_prefix}_00000.parquet")
        assert os.path.exists(f"{output_path_prefix}_00001.parquet")
        assert os.path.exists(f"{output_path_prefix}_00002.parquet")
        assert os.path.exists(f"{output_path_prefix}_00003.parquet")
        assert os.path.exists(f"{output_path_prefix}_00000.csv")
        assert os.path.exists(f"{output_path_prefix}_00001.csv")
        assert os.path.exists(f"{output_path_prefix}_00002.csv")
        assert os.path.exists(f"{output_path_prefix}_00003.csv")
        data0 = read_data_parquet(f"{output_path_prefix}_00000.parquet",
                                  ["pred", "src_nid", "dst_nid"])
        data1 = read_data_parquet(f"{output_path_prefix}_00001.parquet",
                                  ["pred", "src_nid", "dst_nid"])
        data2 = read_data_parquet(f"{output_path_prefix}_00002.parquet",
                                  ["pred", "src_nid", "dst_nid"])
        data3 = read_data_parquet(f"{output_path_prefix}_00003.parquet",
                                  ["pred", "src_nid", "dst_nid"])
        data0_csv = read_csv(f"{output_path_prefix}_00000.csv")
        data1_csv = read_csv(f"{output_path_prefix}_00001.csv")
        data2_csv = read_csv(f"{output_path_prefix}_00002.csv")
        data3_csv = read_csv(f"{output_path_prefix}_00003.csv")
        assert len(data0["pred"]) == 256
        assert len(data1["pred"]) == 256
        assert len(data2["pred"]) == 256
        assert len(data3["pred"]) == 232
        assert len(data0_csv["pred"]) == 256
        assert len(data1_csv["pred"]) == 256
        assert len(data2_csv["pred"]) == 256
        assert len(data3_csv["pred"]) == 232
        preds_ = [data0["pred"], data1["pred"], data2["pred"], data3["pred"]]
        src_nids_ = [data0["src_nid"], data1["src_nid"], data2["src_nid"], data3["src_nid"]]
        dst_nids_ = [data0["dst_nid"], data1["dst_nid"], data2["dst_nid"], data3["dst_nid"]]
        csv_preds_ = [data0_csv["pred"], data1_csv["pred"],
                      data2_csv["pred"], data3_csv["pred"]]
        csv_src_nids_ = [data0_csv["src_nid"], data1_csv["src_nid"],
                         data2_csv["src_nid"], data3_csv["src_nid"]]
        csv_dst_nids_ = [data0_csv["dst_nid"], data1_csv["dst_nid"],
                         data2_csv["dst_nid"], data3_csv["dst_nid"]]
        preds_ = np.concatenate(preds_, axis=0)
        src_nids_ = np.concatenate(src_nids_, axis=0)
        dst_nids_ = np.concatenate(dst_nids_, axis=0)
        csv_preds_ = np.concatenate(csv_preds_, axis=0)
        csv_src_nids_ = np.concatenate(csv_src_nids_, axis=0)
        csv_dst_nids_ = np.concatenate(csv_dst_nids_, axis=0)
        assert_almost_equal(preds_, csv_preds_, decimal=5)
        assert_equal(src_nids_, csv_src_nids_)
        assert_equal(dst_nids_, csv_dst_nids_)
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

def test_write_data_parquet_file():
    data = {"emb": np.random.rand(10, 10),
            "nid": np.arange(10),
            "pred": np.random.rand(10, 10)}

    def check_write_content(fname, col_names):
        # col_names should in order of emb, nid and pred
        parq_data = read_data_parquet(fname, col_names)
        assert_almost_equal(data["emb"], parq_data[col_names[0]])
        assert_equal(data["nid"], parq_data[col_names[1]])
        assert_almost_equal(data["pred"], parq_data[col_names[2]])

    # without renaming columns
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_prefix = os.path.join(tmpdirname, "test")
        write_data_parquet_file(data, file_prefix, None)
        output_fname = f"{file_prefix}.parquet"

        check_write_content(output_fname, ["emb", "nid", "pred"])

    # rename all column names
    with tempfile.TemporaryDirectory() as tmpdirname:
        col_name_map = {
            "emb": "new_emb",
            "nid": "new_nid",
            "pred": "new_pred"
        }
        file_prefix = os.path.join(tmpdirname, "test")
        write_data_parquet_file(data, file_prefix, col_name_map)
        output_fname = f"{file_prefix}.parquet"

        check_write_content(output_fname, ["new_emb", "new_nid", "new_pred"])

    # rename part of column names
    with tempfile.TemporaryDirectory() as tmpdirname:
        col_name_map = {
            "emb": "new_emb",
            "nid": "new_nid",
        }
        file_prefix = os.path.join(tmpdirname, "test")
        write_data_parquet_file(data, file_prefix, col_name_map)
        output_fname = f"{file_prefix}.parquet"

        check_write_content(output_fname, ["new_emb", "new_nid", "pred"])

def test_write_data_csv_file():
    data = {"emb": np.random.rand(10, 10),
            "nid": np.arange(10),
            "pred": np.random.rand(10, 10)}

    def check_write_content(fname, col_names):
        # col_names should in order of emb, nid and pred
        csv_data = pd.read_csv(fname, delimiter=",")
        # emb
        assert col_names[0] in csv_data
        csv_emb_data = csv_data[col_names[0]].values.tolist()
        csv_emb_data = [d.split(";") for d in csv_emb_data]
        csv_emb_data = np.array(csv_emb_data, dtype=np.float32)
        assert_almost_equal(data["emb"], csv_emb_data)

        # nid
        assert col_names[1] in csv_data
        csv_nid_data = csv_data[col_names[1]].values.tolist()
        csv_nid_data = np.array(csv_nid_data, dtype=np.int32)
        assert_equal(data["nid"], csv_nid_data)

        # pred
        assert col_names[2] in csv_data
        csv_pred_data = csv_data[col_names[2]].values.tolist()
        csv_pred_data = [d.split(";") for d in csv_pred_data]
        csv_pred_data = np.array(csv_pred_data, dtype=np.float32)
        assert_almost_equal(data["pred"], csv_pred_data)

    # without renaming columns
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_prefix = os.path.join(tmpdirname, "test")
        write_data_csv_file(data, file_prefix, col_name_map=None)
        output_fname = f"{file_prefix}.csv"

        check_write_content(output_fname, ["emb", "nid", "pred"])

    # rename all column names
    with tempfile.TemporaryDirectory() as tmpdirname:
        col_name_map = {
            "emb": "new_emb",
            "nid": "new_nid",
            "pred": "new_pred"
        }
        file_prefix = os.path.join(tmpdirname, "test")
        write_data_csv_file(data, file_prefix, col_name_map=col_name_map)
        output_fname = f"{file_prefix}.csv"

        check_write_content(output_fname, ["new_emb", "new_nid", "new_pred"])

    # rename part of column names
    with tempfile.TemporaryDirectory() as tmpdirname:
        col_name_map = {
            "emb": "new_emb",
            "nid": "new_nid",
        }
        file_prefix = os.path.join(tmpdirname, "test")
        write_data_csv_file(data, file_prefix, col_name_map=col_name_map)
        output_fname = f"{file_prefix}.csv"

        check_write_content(output_fname, ["new_emb", "new_nid", "pred"])

def test_parse_config():
    with tempfile.TemporaryDirectory() as tmpdirname:
        part_path = os.path.join(tmpdirname, "tmp.json")
        save_prediction_path = os.path.join(tmpdirname,  "predict")
        save_embed_path = os.path.join(tmpdirname, "emb")
        Path(part_path).touch()
        os.mkdir(save_prediction_path)
        os.mkdir(save_embed_path)

        # single task config
        target_ntype = "n1"
        config = GSConfig.__new__(GSConfig)
        setattr(config, "_part_config", part_path)
        setattr(config, "_save_prediction_path", save_prediction_path)
        setattr(config, "_save_embed_path", save_embed_path)
        setattr(config, "_task_type", BUILTIN_TASK_NODE_CLASSIFICATION)
        setattr(config, "_target_ntype", target_ntype)
        setattr(config, "_multi_tasks", None)
        node_id_mapping, predict_dir, emb_dir, task_emb_dirs, pred_ntypes, pred_etypes = _parse_gs_config(config)
        assert node_id_mapping == os.path.join(tmpdirname, "raw_id_mappings")
        assert predict_dir == save_prediction_path
        assert emb_dir == save_embed_path
        assert len(pred_ntypes) == 1
        assert pred_ntypes[0] == target_ntype
        assert len(pred_etypes) == 0
        assert len(task_emb_dirs) == 0

        target_etype = ["n0,r0,n1"]
        config = GSConfig.__new__(GSConfig)
        setattr(config, "_part_config", part_path)
        setattr(config, "_save_prediction_path", save_prediction_path)
        setattr(config, "_save_embed_path", save_embed_path)
        setattr(config, "_task_type", BUILTIN_TASK_EDGE_CLASSIFICATION)
        setattr(config, "_target_etype", target_etype)
        setattr(config, "_multi_tasks", None)

        node_id_mapping, predict_dir, emb_dir, task_emb_dirs, pred_ntypes, pred_etypes = _parse_gs_config(config)
        assert node_id_mapping == os.path.join(tmpdirname, "raw_id_mappings")
        assert predict_dir == save_prediction_path
        assert emb_dir == save_embed_path
        assert len(pred_ntypes) == 0
        assert len(pred_etypes) == 1
        assert pred_etypes[0] == ["n0", "r0", "n1"]
        assert len(task_emb_dirs) == 0

        # multi-task config
        multi_task_config = [
            {
                "node_classification": {
                    "target_ntype": "n0",
                    "batch_size": 128,
                    "label_field": "nc",
                    "num_classes":4
                },
            },
            {
                "node_regression": {
                    "target_ntype": "n1",
                    "batch_size": 128,
                    "label_field": "nr",
                },
            },
            {
                "edge_classification": {
                    "target_etype": ["n0,r0,r1"],
                    "batch_size": 128,
                    "label_field": "ec",
                    "num_classes":2
                },
            },
            {
                "edge_regression" : {
                    "target_etype": ["n0,r0,r2"],
                    "batch_size": 128,
                    "label_field": "er"
                },
            },
            {
                "link_prediction" : {
                    "num_negative_edges": 4,
                    "batch_size": 128,
                    "exclude_training_targets": False,
                    "lp_embed_normalizer": "l2_norm"
                }
            },
        ]

        config = GSConfig.__new__(GSConfig)
        setattr(config, "_part_config", part_path)
        setattr(config, "_save_prediction_path", save_prediction_path)
        setattr(config, "_save_embed_path", save_embed_path)
        config._parse_multi_tasks(multi_task_config)
        node_id_mapping, predict_dir, emb_dir, task_emb_dirs, pred_ntypes, pred_etypes = _parse_gs_config(config)

        assert node_id_mapping == os.path.join(tmpdirname, "raw_id_mappings")
        assert isinstance(predict_dir, tuple)
        node_predict_dirs, edge_predict_dirs = predict_dir
        assert len(node_predict_dirs) == 2
        assert len(edge_predict_dirs) == 2
        assert node_predict_dirs[0] == os.path.join(save_prediction_path, config.multi_tasks[0].task_id)
        assert node_predict_dirs[1] == os.path.join(save_prediction_path, config.multi_tasks[1].task_id)
        assert edge_predict_dirs[0] == os.path.join(save_prediction_path, config.multi_tasks[2].task_id)
        assert edge_predict_dirs[1] == os.path.join(save_prediction_path, config.multi_tasks[3].task_id)
        assert emb_dir == save_embed_path
        assert len(pred_ntypes) == 2
        assert pred_ntypes[0] == "n0"
        assert pred_ntypes[1] == "n1"
        assert len(pred_etypes) == 2
        assert pred_etypes[0] == ['n0', 'r0', 'r1']
        assert pred_etypes[1] == ['n0', 'r0', 'r2']
        print(task_emb_dirs)
        assert len(task_emb_dirs) == 1
        assert task_emb_dirs[0] == get_mttask_id(
            task_type="link_prediction",
            etype="ALL_ETYPE")

        # there is no predict path
        # it will use emb_path
        multi_task_config[4]["link_prediction"].pop("lp_embed_normalizer")
        config = GSConfig.__new__(GSConfig)
        setattr(config, "_part_config", part_path)
        setattr(config, "_save_embed_path", save_embed_path)
        config._parse_multi_tasks(multi_task_config)
        node_id_mapping, predict_dir, emb_dir, task_emb_dirs, pred_ntypes, pred_etypes = _parse_gs_config(config)
        assert node_id_mapping == os.path.join(tmpdirname, "raw_id_mappings")
        assert isinstance(predict_dir, tuple)
        node_predict_dirs, edge_predict_dirs = predict_dir
        assert len(node_predict_dirs) == 2
        assert len(edge_predict_dirs) == 2
        assert node_predict_dirs[0] == os.path.join(save_embed_path, config.multi_tasks[0].task_id)
        assert node_predict_dirs[1] == os.path.join(save_embed_path, config.multi_tasks[1].task_id)
        assert edge_predict_dirs[0] == os.path.join(save_embed_path, config.multi_tasks[2].task_id)
        assert edge_predict_dirs[1] == os.path.join(save_embed_path, config.multi_tasks[3].task_id)
        assert len(task_emb_dirs) == 0

        # there is no predict path and emb path
        config = GSConfig.__new__(GSConfig)
        setattr(config, "_part_config", part_path)
        config._parse_multi_tasks(multi_task_config)
        node_id_mapping, predict_dir, emb_dir, task_emb_dirs, pred_ntypes, pred_etypes = _parse_gs_config(config)
        assert predict_dir is None
        assert emb_dir is None

if __name__ == '__main__':
    test_parse_config()

    test_write_data_csv_file()
    test_write_data_parquet_file()
    test__get_file_range()
    test_worker_remap_edge_pred()
    test_worker_remap_node_data("pred")
