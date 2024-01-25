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
    etype0 = ("n0", "access", "n1")
    etype1 = ("n1", "access", "n0")

    # generate random node ids for nodes
    nid0 = np.random.randint(10000, size=1000) * 10000 + np.arange(1000)
    nid1 = np.random.randint(10000, size=1000) * 10000 + np.arange(1000)
    nid0_str = nid0.astype('str')
    nid1_str = nid1.astype('str')

    nid0_map = IdMap(nid0_str)
    nid1_map = IdMap(nid1_str)

    mapping_subpath = "id_mapping"
    os.makedirs(os.path.join(output_path, mapping_subpath), exist_ok=True)
    nid0_map.save(os.path.join(output_path, mapping_subpath, ntype0, "part-00000.parquet"))
    nid1_map.save(os.path.join(output_path, mapping_subpath, ntype1, "part-00000.parquet"))

    # generate faked edge results
    src0 = nid0[np.random.randint(1000, size=2000)]
    dst0 = nid1[np.random.randint(1000, size=2000)]
    pred0 = np.stack((src0, dst0), axis=1)
    src0_new, _ = nid0_map.map_id(src0.astype('str'))
    dst0_new, _ = nid1_map.map_id(dst0.astype('str'))


    src1 = nid0[np.random.randint(1000, size=2000)]
    dst1 = nid1[np.random.randint(1000, size=2000)]
    pred1 = np.stack((src1, dst1), axis=1)
    src1_new, _ = nid0_map.map_id(src1.astype('str'))
    dst1_new, _ = nid1_map.map_id(dst1.astype('str'))


    # generate faked edge results
    src2 = nid1[np.random.randint(1000, size=2000)]
    dst2 = nid0[np.random.randint(1000, size=2000)]
    pred2 = np.stack((src2, dst2), axis=1)
    src2_new, _ = nid1_map.map_id(src2.astype('str'))
    dst2_new, _ = nid0_map.map_id(dst2.astype('str'))


    src3 = nid1[np.random.randint(1000, size=2000)]
    dst3 = nid0[np.random.randint(1000, size=2000)]
    pred3 = np.stack((src3, dst3), axis=1)
    src3_new, _ = nid1_map.map_id(src3.astype('str'))
    dst3_new, _ = nid0_map.map_id(dst3.astype('str'))

    meta_info = {
        "format": "pytorch",
        "world_size": 2,
        "etypes": [etype0, etype1]
    }

    pred_output = os.path.join(output_path, "pred")
    meta_fname = os.path.join(pred_output, "result_info.json")
    os.makedirs(pred_output)
    with open(meta_fname, 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, indent=4)

    pred_output_etype0 = os.path.join(pred_output, "_".join(etype0))
    pred_output_etype1 = os.path.join(pred_output, "_".join(etype1))
    os.makedirs(pred_output_etype0)
    os.makedirs(pred_output_etype1)
    th.save(th.tensor(pred0), os.path.join(pred_output_etype0, f"predict-{pad_file_index(0)}.pt"))
    th.save(th.tensor(src0_new), os.path.join(pred_output_etype0, f"src_nids-{pad_file_index(0)}.pt"))
    th.save(th.tensor(dst0_new), os.path.join(pred_output_etype0, f"dst_nids-{pad_file_index(0)}.pt"))
    th.save(th.tensor(pred1), os.path.join(pred_output_etype0, f"predict-{pad_file_index(1)}.pt"))
    th.save(th.tensor(src1_new), os.path.join(pred_output_etype0, f"src_nids-{pad_file_index(1)}.pt"))
    th.save(th.tensor(dst1_new), os.path.join(pred_output_etype0, f"dst_nids-{pad_file_index(1)}.pt"))

    th.save(th.tensor(pred2), os.path.join(pred_output_etype1, f"predict-{pad_file_index(0)}.pt"))
    th.save(th.tensor(src2_new), os.path.join(pred_output_etype1, f"src_nids-{pad_file_index(0)}.pt"))
    th.save(th.tensor(dst2_new), os.path.join(pred_output_etype1, f"dst_nids-{pad_file_index(0)}.pt"))
    th.save(th.tensor(pred3), os.path.join(pred_output_etype1, f"predict-{pad_file_index(1)}.pt"))
    th.save(th.tensor(src3_new), os.path.join(pred_output_etype1, f"src_nids-{pad_file_index(1)}.pt"))
    th.save(th.tensor(dst3_new), os.path.join(pred_output_etype1, f"dst_nids-{pad_file_index(1)}.pt"))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Check edge prediction remapping")
    argparser.add_argument("--output", type=str, required=True,
                           help="Path to save the generated data")

    args = argparser.parse_args()

    main(args)
