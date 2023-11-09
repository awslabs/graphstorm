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

import numpy as np
from numpy.testing import assert_almost_equal

from graphstorm.gconstruct.file_io import read_data_parquet

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("check_infer")
    argparser.add_argument("--infer-prediction", type=str, required=True,
                           help="Path to embedding saved by trainer")
    argparser.add_argument("--no-sfs-prediction", type=str, required=True,
                           help="Path to embedding saved by trainer")
    argparser.add_argument("--edge-prediction", action='store_true',
                           help="Path to embedding saved by trainer")

    args = argparser.parse_args()

    sfs_files = os.listdir(args.infer_prediction)
    no_sfs_files = os.listdir(args.no_sfs_prediction)

    sfs_files = [file for file in sfs_files if file.endswith(".parquet")]
    no_sfs_files = [file for file in no_sfs_files if file.endswith(".parquet")]

    if args.edge_prediction:
        sfs_src_nids = []
        sfs_dst_nids = []
        sfs_preds = []

        for f in sfs_files:
            data = read_data_parquet(os.path.join(args.infer_prediction, f),
                                     data_fields=["pred", "src_nid", "dst_nid"])
            sfs_src_nids.append(data["src_nid"])
            sfs_dst_nids.append(data["dst_nid"])
            sfs_preds.append(data["pred"])
        sfs_src_nids = np.concatenate(sfs_src_nids)
        sfs_dst_nids = np.concatenate(sfs_dst_nids)
        sfs_preds = np.concatenate(sfs_preds)

        dst_sort_idx = np.argsort(sfs_dst_nids)
        sfs_src_nids = sfs_src_nids[dst_sort_idx]
        sfs_dst_nids = sfs_dst_nids[dst_sort_idx]
        sfs_preds = sfs_preds[dst_sort_idx]
        src_sort_idx = np.argsort(sfs_src_nids)
        sfs_src_nids = sfs_src_nids[src_sort_idx]
        sfs_dst_nids = sfs_dst_nids[src_sort_idx]
        sfs_preds = sfs_preds[src_sort_idx]

        no_sfs_src_nids = []
        no_sfs_dst_nids = []
        no_sfs_preds = []

        for f in no_sfs_files:
            data = read_data_parquet(os.path.join(args.infer_prediction, f),
                                     data_fields=["pred", "src_nid", "dst_nid"])
            no_sfs_src_nids.append(data["src_nid"])
            no_sfs_dst_nids.append(data["dst_nid"])
            no_sfs_preds.append(data["pred"])
        no_sfs_src_nids = np.concatenate(no_sfs_src_nids)
        no_sfs_dst_nids = np.concatenate(no_sfs_dst_nids)
        no_sfs_preds = np.concatenate(no_sfs_preds)

        dst_sort_idx = np.argsort(no_sfs_dst_nids)
        no_sfs_src_nids = no_sfs_src_nids[dst_sort_idx]
        no_sfs_dst_nids = no_sfs_dst_nids[dst_sort_idx]
        no_sfs_preds = no_sfs_preds[dst_sort_idx]
        src_sort_idx = np.argsort(no_sfs_src_nids)
        no_sfs_src_nids = no_sfs_src_nids[src_sort_idx]
        no_sfs_dst_nids = no_sfs_dst_nids[src_sort_idx]
        no_sfs_preds = no_sfs_preds[src_sort_idx]

        assert_almost_equal(sfs_src_nids, no_sfs_src_nids)
        assert_almost_equal(sfs_dst_nids, no_sfs_dst_nids)
        assert_almost_equal(sfs_preds, no_sfs_preds)
    else:
        # node prediction task
        sfs_nids = []
        sfs_preds = []
        for f in sfs_files:
            data = read_data_parquet(os.path.join(args.infer_prediction, f),
                                     data_fields=["pred", "nid"])
            sfs_nids.append(data["nid"])
            sfs_preds.append(data["pred"])
        sfs_nids = np.concatenate(sfs_nids)
        sfs_preds = np.concatenate(sfs_preds)
        src_sort_idx = np.argsort(sfs_nids)
        sfs_nids = sfs_nids[src_sort_idx]
        sfs_preds = sfs_preds[src_sort_idx]

        no_sfs_nids = []
        no_sfs_preds = []
        for f in sfs_files:
            data = read_data_parquet(os.path.join(args.infer_prediction, f),
                                     data_fields=["pred", "nid"])
            no_sfs_nids.append(data["nid"])
            no_sfs_preds.append(data["pred"])
        no_sfs_nids = np.concatenate(no_sfs_nids)
        no_sfs_preds = np.concatenate(no_sfs_preds)
        src_sort_idx = np.argsort(no_sfs_nids)
        no_sfs_nids = no_sfs_nids[src_sort_idx]
        no_sfs_preds = no_sfs_preds[src_sort_idx]

        assert_almost_equal(sfs_nids, no_sfs_nids)
        assert_almost_equal(sfs_preds, no_sfs_preds)
