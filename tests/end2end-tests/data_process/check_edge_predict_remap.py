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

from graphstorm.gconstruct.file_io import read_data_parquet
from numpy.testing import assert_equal

def main(args):
    test_etypes = [tuple(test_etype.split(",")) for test_etype in args.test_etypes]
    predict_path = args.remap_output
    column_names = args.column_names
    etype0 = ("n0", "access", "n1")
    etype1 = ("n1", "access", "n0")

    column_name_map = {}
    if column_names is not None:
        for col_rename_pair in column_names:
            orig_name, new_name = col_rename_pair.split(",")
            column_name_map[orig_name] = new_name

    data_fields = ["pred", "src_nid", "dst_nid"] if column_names is None \
        else [column_name_map["pred"], column_name_map["src_nid"], column_name_map["dst_nid"]]

    if etype0 in test_etypes:
        etype0_pred_path = os.path.join(predict_path, "_".join(etype0))
        data = read_data_parquet(
            os.path.join(etype0_pred_path, "pred.predict-00000_00000.parquet"),
            data_fields=data_fields)
        if column_names is not None:
            data["pred"] = data[column_name_map["pred"]]
            data["src_nid"] = data[column_name_map["src_nid"]]
            data["dst_nid"] = data[column_name_map["dst_nid"]]
        assert_equal(data["pred"][:,0].astype("str"), data["src_nid"])
        assert_equal(data["pred"][:,1].astype("str"), data["dst_nid"])

        data = read_data_parquet(
            os.path.join(etype0_pred_path, "pred.predict-00001_00000.parquet"),
            data_fields=data_fields)
        if column_names is not None:
            data["pred"] = data[column_name_map["pred"]]
            data["src_nid"] = data[column_name_map["src_nid"]]
            data["dst_nid"] = data[column_name_map["dst_nid"]]
        assert_equal(data["pred"][:,0].astype("str"), data["src_nid"])
        assert_equal(data["pred"][:,1].astype("str"), data["dst_nid"])

    if etype1 in test_etypes:
        etype1_pred_path = os.path.join(predict_path, "_".join(etype1))
        data = read_data_parquet(
            os.path.join(etype1_pred_path, "pred.predict-00000_00000.parquet"),
            data_fields=data_fields)
        if column_names is not None:
            data["pred"] = data[column_name_map["pred"]]
            data["src_nid"] = data[column_name_map["src_nid"]]
            data["dst_nid"] = data[column_name_map["dst_nid"]]
        assert_equal(data["pred"][:,0].astype("str"), data["src_nid"])
        assert_equal(data["pred"][:,1].astype("str"), data["dst_nid"])
        data = read_data_parquet(
            os.path.join(etype1_pred_path, "pred.predict-00001_00000.parquet"),
            data_fields=data_fields)
        if column_names is not None:
            data["pred"] = data[column_name_map["pred"]]
            data["src_nid"] = data[column_name_map["src_nid"]]
            data["dst_nid"] = data[column_name_map["dst_nid"]]
        assert_equal(data["pred"][:,0].astype("str"), data["src_nid"])
        assert_equal(data["pred"][:,1].astype("str"), data["dst_nid"])

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Check edge prediction remapping")
    argparser.add_argument("--remap-output", type=str, required=True,
                           help="Path to save the generated data")
    argparser.add_argument("--column-names", type=str, nargs="+", default=None,
                           help="Defines how to rename default column names to new names.")
    argparser.add_argument("--test-etypes", type=str, nargs="+", default=["n0,access,n1", "n1,access,n0"],
                           help="etypes with prediction results")

    args = argparser.parse_args()

    main(args)