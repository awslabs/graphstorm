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

import argparse
import os
import json

from numpy.testing import assert_almost_equal
import torch as th
import numpy as np

from graphstorm.gconstruct.file_io import read_data_parquet

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("check_infer")
    argparser.add_argument("--train-embout", type=str, required=True,
                           help="Path to embedding saved by trainer")
    argparser.add_argument("--infer-embout", type=str, required=True,
                           help="Path to embedding saved by trainer")
    args = argparser.parse_args()
    with open(os.path.join(args.train_embout, "emb_info.json"), 'r', encoding='utf-8') as f:
        train_emb_info = json.load(f)

    with open(os.path.join(args.infer_embout, "emb_info.json"), 'r', encoding='utf-8') as f:
        info_emb_info = json.load(f)

    # meta info should be same
    assert len(train_emb_info["emb_name"]) == len(info_emb_info["emb_name"])

    # feats are same
    for ntype in info_emb_info["emb_name"]:
        train_emb = []
        train_nids = []
        ntype_emb_path = os.path.join(args.train_embout, ntype)
        emb_files = os.listdir(ntype_emb_path)
        ntype_emb_files = [file for file in emb_files if file.endswith(".pt") and file.startswith("embed-")]
        ntype_nid_files = [file for file in emb_files if file.endswith(".pt") and file.startswith("embed_nids-")]
        ntype_emb_files = sorted(ntype_emb_files)
        ntype_nid_files = sorted(ntype_nid_files)
        for f in ntype_emb_files:
            # Only work with torch 1.13+
            train_emb.append(th.load(os.path.join(ntype_emb_path, f),weights_only=True))
        for f in ntype_nid_files:
            train_nids.append(th.load(os.path.join(ntype_emb_path, f),weights_only=True))
        train_emb = th.cat(train_emb, dim=0)
        train_nids = th.cat(train_nids, dim=0)

        ntype_remaped_emb_files = [file for file in emb_files if file.endswith(".parquet")]
        ntype_remaped_emb_files = sorted(ntype_remaped_emb_files)
        train_remaped_emb = []
        train_remaped_nids = []
        for f in ntype_remaped_emb_files:
            data = read_data_parquet(os.path.join(ntype_emb_path, f),
                                     data_fields=["emb", "nid"])
            train_remaped_emb.append(data["emb"])
            train_remaped_nids.append(data["nid"])
        train_remaped_emb = np.concatenate(train_remaped_emb)
        train_remaped_nids = np.concatenate(train_remaped_nids)

        infer_emb = []
        infer_nids = []
        ntype_emb_path = os.path.join(args.infer_embout, ntype)
        emb_files = os.listdir(ntype_emb_path)
        ntype_emb_files = [file for file in emb_files if file.endswith(".pt") and file.startswith("embed-")]
        ntype_nid_files = [file for file in emb_files if file.endswith(".pt") and file.startswith("embed_nids-")]
        ntype_emb_files = sorted(ntype_emb_files)
        ntype_nid_files = sorted(ntype_nid_files)
        for ef, nf in zip(ntype_emb_files, ntype_nid_files):
            # Only work with torch 1.13+
            infer_emb.append(th.load(os.path.join(ntype_emb_path, ef), weights_only=True))
            infer_nids.append(th.load(os.path.join(ntype_emb_path, nf), weights_only=True))
        infer_emb = th.cat(infer_emb, dim=0)
        infer_nids = th.cat(infer_nids, dim=0)
        ntype_remaped_emb_files = [file for file in emb_files if file.endswith(".parquet")]
        ntype_remaped_emb_files = sorted(ntype_remaped_emb_files)
        infer_remaped_emb = []
        infer_remaped_nids = []
        for f in ntype_remaped_emb_files:
            data = read_data_parquet(os.path.join(ntype_emb_path, f),
                                     data_fields=["emb", "nid"])
            infer_remaped_emb.append(data["emb"])
            infer_remaped_nids.append(data["nid"])
        infer_remaped_emb = np.concatenate(infer_remaped_emb)
        infer_remaped_nids = np.concatenate(infer_remaped_nids)

        assert infer_emb.shape[0] == infer_nids.shape[0]
        assert train_emb.shape[1] == infer_emb.shape[1]
        assert infer_remaped_emb.shape[0] == infer_nids.shape[0]
        assert infer_remaped_nids.shape[0] == infer_nids.shape[0]

        # sort train embs
        # train emb  [xxx, xxx, xxx]
        # train nids [0, 1, 2, ...]
        train_emb = train_emb[th.argsort(train_nids)]
        for nid, inf_emb in zip(infer_nids, infer_emb):
            assert_almost_equal(train_emb[nid].numpy(), inf_emb.numpy(), decimal=1)

        train_remap_embs = {}
        for nid, train_emb in zip (train_remaped_nids, train_remaped_emb):
            train_remap_embs[nid] = train_emb
        for nid, inf_emb in zip(infer_remaped_nids, infer_remaped_emb):
            assert_almost_equal(train_remap_embs[nid], inf_emb, decimal=1)
