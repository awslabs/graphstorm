from .movielens import MovieLens100kNCDataset
from .ogbn_arxiv import OGBArxivTextFeatDataset
from .ogbn_datasets import OGBTextFeatDataset
from .dataset import ConstructedGraphDataset
try:
    from .gs_dataset import StandardGSgnnDataset
except:
    print('Cannot load StandardGSgnnDataset suppport')
from .utils import generated_train_valid_test_splits, adjust_eval_mapping_for_partition
