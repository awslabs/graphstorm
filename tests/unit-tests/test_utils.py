import os
from pathlib import Path
from argparse import Namespace

import torch as th
import numpy as np
from graphstorm.model.utils import save_embeddings, LazyDistTensor

def test_save_embeddings():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        random_emb = th.rand((103, 12))
        type0_random_emb = LazyDistTensor(random_emb, th.arange(103))
        random_emb = th.rand((205, 12))
        type1_random_emb = LazyDistTensor(random_emb, th.arange(205))

        emb = {
            "type0": type0_random_emb,
            "type1": type1_random_emb
        }
        save_embeddings(tmpdirname, emb, 0, 4)
        emb = {
            "type0": type0_random_emb,
            "type1": type1_random_emb
        }
        save_embeddings(tmpdirname, emb, 1, 4)
        emb = {
            "type0": type0_random_emb,
            "type1": type1_random_emb
        }
        save_embeddings(tmpdirname, emb, 2, 4)
        emb = {
            "type0": type0_random_emb,
            "type1": type1_random_emb
        }
        save_embeddings(tmpdirname, emb, 3, 4)

        feats_type0 = [th.load(os.path.join(tmpdirname, "type0_emb.part{}.bin".format(i))) for i in range(4)]
        feats_type0 = th.cat(feats_type0, dim=0)
        feats_type1 = [th.load(os.path.join(tmpdirname, "type1_emb.part{}.bin".format(i))) for i in range(4)]
        feats_type1 = th.cat(feats_type1, dim=0)

        assert np.all(type0_random_emb.dist_tensor.numpy() == feats_type0.numpy())
        assert np.all(type1_random_emb.dist_tensor.numpy() == feats_type1.numpy())

if __name__ == '__main__':
    test_save_embeddings()
