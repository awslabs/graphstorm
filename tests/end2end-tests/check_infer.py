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

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("check_infer")
    argparser.add_argument("--train_embout", type=str, required=True,
                           help="Path to embedding saved by trainer")
    argparser.add_argument("--infer_embout", type=str, required=True,
                           help="Path to embedding saved by trainer")
    argparser.add_argument("--link_prediction", action='store_true',
                           help="Path to embedding saved by trainer")
    argparser.add_argument("--mini-batch-infer", action='store_true',
                           help="Inference use minibatch inference.")
    args = argparser.parse_args()
    with open(os.path.join(args.train_embout, "emb_info.json"), 'r', encoding='utf-8') as f:
        train_emb_info = json.load(f)

    with open(os.path.join(args.infer_embout, "emb_info.json"), 'r', encoding='utf-8') as f:
        info_emb_info = json.load(f)

    # meta info should be same
    if args.link_prediction:
        # For link prediction, both training and inference will save
        # node embeddings for each node type.
        assert len(train_emb_info["emb_name"]) == len(info_emb_info["emb_name"])
        assert sorted(train_emb_info["emb_name"]) == sorted(info_emb_info["emb_name"])
    else:
        # For other tasks, we only save node embeddings for the target type
        # in the inference.
        assert len(train_emb_info["emb_name"]) >= len(info_emb_info["emb_name"])

    # feats are same
    train_emb_files = os.listdir(args.train_embout)
    train_emb_files = sorted(train_emb_files)
    info_emb_files = os.listdir(args.infer_embout)
    info_emb_files = sorted(info_emb_files)
    for name in info_emb_info["emb_name"]:
        train_emb = []
        for f in train_emb_files:
            if f.startswith(f'{name}_emb.part'):
                # Only work with torch 1.13+
                train_emb.append(th.load(os.path.join(args.train_embout, f),weights_only=True))
        train_emb = th.cat(train_emb, dim=0)

        infer_emb = []
        for f in info_emb_files:
            if f.startswith(f'{name}_emb.part'):
                # Only work with torch 1.13+
                infer_emb.append(th.load(os.path.join(args.infer_embout, f), weights_only=True))
        infer_emb = th.cat(infer_emb, dim=0)

        assert train_emb.shape[0] == infer_emb.shape[0]
        assert train_emb.shape[1] == infer_emb.shape[1]

        if args.mini_batch_infer:
            # When inference is done with minibatch inference, only node
            # embeddings of the test set are computed.
            for i in range(len(train_emb)):
                if th.all(infer_emb[i] == 0.):
                    continue
                assert_almost_equal(train_emb[i].numpy(), infer_emb[i].numpy(), decimal=4)
        else:
            assert_almost_equal(train_emb.numpy(), infer_emb.numpy(), decimal=2)
