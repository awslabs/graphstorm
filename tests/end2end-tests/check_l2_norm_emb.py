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
    argparser.add_argument("--embout", type=str, required=True,
                            help="Path to embedding ")
    args = argparser.parse_args()
    with open(os.path.join(args.embout, "emb_info.json"), 'r', encoding='utf-8') as f:
        emb_info = json.load(f)

     # feats are same
    for ntype in emb_info["emb_name"]:
        ntype_emb_path = os.path.join(args.embout, ntype)
        emb_files = os.listdir(ntype_emb_path)
        ntype_remaped_emb_files = [file for file in emb_files if file.endswith(".parquet")]

        for f in ntype_remaped_emb_files:
            data = read_data_parquet(os.path.join(ntype_emb_path, f),
                                     data_fields=["emb"])
            data = np.absolute(data["emb"])
            # with norm l2, each value must in (-1.0, 1.0)
            assert np.all(data < 1.0)
