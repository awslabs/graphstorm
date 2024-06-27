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
import json
import torch as th

from graphstorm.gconstruct.file_io import read_data_parquet
from numpy.testing import assert_equal

def main(args):
    test_ntypes = args.test_ntypes
    predict_path = args.remap_output
    ntype0 = "n0"
    ntype1 = "n1"

    if ntype0 in test_ntypes:
        ntype0_pred_path = os.path.join(predict_path, ntype0)
        data = read_data_parquet(
            os.path.join(ntype0_pred_path, "pred.predict-00000_00000.parquet"),
            data_fields=["pred", "nid"])

        assert_equal(data["pred"][:,0].astype("str"), data["nid"])
        assert_equal(data["pred"][:,1].astype("str"), data["nid"])
        data = read_data_parquet(
            os.path.join(ntype0_pred_path, "pred.predict-00001_00000.parquet"),
            data_fields=["pred", "nid"])
        assert_equal(data["pred"][:,0].astype("str"), data["nid"])
        assert_equal(data["pred"][:,1].astype("str"), data["nid"])

    if ntype1 in test_ntypes:
        ntype1_pred_path = os.path.join(predict_path, ntype1)
        data = read_data_parquet(
            os.path.join(ntype1_pred_path, "pred.predict-00000_00000.parquet"),
            data_fields=["pred", "nid"])
        assert_equal(data["pred"][:,0].astype("str"), data["nid"])
        assert_equal(data["pred"][:,1].astype("str"), data["nid"])
        data = read_data_parquet(
            os.path.join(ntype1_pred_path, "pred.predict-00001_00000.parquet"),
            data_fields=["pred", "nid"])
        assert_equal(data["pred"][:,0].astype("str"), data["nid"])
        assert_equal(data["pred"][:,1].astype("str"), data["nid"])

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Check edge prediction remapping")
    argparser.add_argument("--remap-output", type=str, required=True,
                           help="Path to save the generated data")
    argparser.add_argument("--test-ntypes", type=str, nargs="+", default=["n0", "n1"],
                           help="ntypes with prediction results")

    args = argparser.parse_args()

    main(args)