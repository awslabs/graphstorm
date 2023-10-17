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

    Generate data for testing edge prediction remapping
"""

import os
import argparse
import json

import numpy as np
import torch as th

from graphstorm.gconstruct.id_map import IdMap
from graphstorm.model.utils import pad_file_index

def main(args):
    output_path = args.output
    ntype0 = "n0"
    ntype1 = "n1"

    # generate random node ids for nodes
    nid0 = np.random.randint(10000, size=1000) * 10000 + np.arange(1000)
    nid1 = np.random.randint(10000, size=1000) * 10000 + np.arange(1000)
    nid0_str = nid0.astype('str')
    nid1_str = nid1.astype('str')

    nid0_map = IdMap(nid0_str)
    nid1_map = IdMap(nid1_str)

    os.makedirs(os.path.join(output_path, "id_mapping"), exist_ok=True)
    nid0_map.save(os.path.join(os.path.join(output_path, "id_mapping"), ntype0+"_id_remap.parquet"))
    nid1_map.save(os.path.join(os.path.join(output_path, "id_mapping"), ntype1+"_id_remap.parquet"))

    # generate faked edge results
    nid0_ = nid0[np.random.randint(1000, size=2000)]
    nid1_ = nid1[np.random.randint(1000, size=2000)]
    pred0 = np.stack((nid0_, nid0_), axis=1)
    pred1 = np.stack((nid1_, nid1_), axis=1)
    nid0_new, _ = nid0_map.map_id(nid0_.astype('str'))
    nid1_new, _ = nid1_map.map_id(nid1_.astype('str'))

    meta_info = {
        "format": "pytorch",
        "world_size": 2,
        "ntypes": [ntype0, ntype1]
    }

    pred_output = os.path.join(output_path, "pred")
    meta_fname = os.path.join(pred_output, "result_info.json")
    os.makedirs(pred_output)
    with open(meta_fname, 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, indent=4)

    pred_output_ntype0 = os.path.join(pred_output, ntype0)
    pred_output_ntype1 = os.path.join(pred_output, ntype1)
    os.makedirs(pred_output_ntype0)
    os.makedirs(pred_output_ntype1)
    th.save(th.tensor(pred0), os.path.join(pred_output_ntype0, f"predict-{pad_file_index(0)}.pt"))
    th.save(th.tensor(nid0_new), os.path.join(pred_output_ntype0, f"nids-{pad_file_index(0)}.pt"))

    th.save(th.tensor(pred1), os.path.join(pred_output_ntype1, f"predict-{pad_file_index(1)}.pt"))
    th.save(th.tensor(nid1_new), os.path.join(pred_output_ntype1, f"nids-{pad_file_index(1)}.pt"))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Check edge prediction remapping")
    argparser.add_argument("--output", type=str, required=True,
                           help="Path to save the generated data")

    args = argparser.parse_args()

    main(args)
