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
    argparser.add_argument("--edge_prediction", action='store_true',
                           help="Path to embedding saved by trainer")
    args = argparser.parse_args()
    with open(os.path.join(args.train_embout, "emb_info.json"), 'r', encoding='utf-8') as f:
        train_emb_info = json.load(f)

    with open(os.path.join(args.infer_embout, "emb_info.json"), 'r', encoding='utf-8') as f:
        info_emb_info = json.load(f)

    # meta info should be same
    if args.edge_prediction:
        # For edge classification, in inference we only save
        # node embeddings for the target type.
        assert len(train_emb_info["emb_name"]) >= len(info_emb_info["emb_name"])
    else:
        # For link prediction, both training and inference will save
        # node embeddings for each node type.
        assert len(train_emb_info["emb_name"]) == len(info_emb_info["emb_name"])
        assert sorted(train_emb_info["emb_name"]) == sorted(info_emb_info["emb_name"])

    # feats are same
    train_emb_files = os.listdir(args.train_embout)
    train_emb_files = sorted(train_emb_files)
    info_emb_files = os.listdir(args.infer_embout)
    info_emb_files = sorted(info_emb_files)
    for name in info_emb_info["emb_name"]:
        train_emb = []
        for f in train_emb_files:
            if f.startswith(f'{name}_emb.part'):
                train_emb.append(th.load(os.path.join(args.train_embout, f)))
        train_emb = th.cat(train_emb, dim=0)

        infer_emb = []
        for f in info_emb_files:
            if f.startswith(f'{name}_emb.part'):
                infer_emb.append(th.load(os.path.join(args.infer_embout, f)))
        infer_emb = th.cat(infer_emb, dim=0)

        assert train_emb.shape[0] == infer_emb.shape[0]
        assert train_emb.shape[1] == infer_emb.shape[1]
        assert_almost_equal(train_emb.numpy(), infer_emb.numpy(), decimal=4)
