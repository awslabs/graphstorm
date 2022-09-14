from .movielens import MovieLens100kNCDataset
from .ogbn_arxiv import OGBArxivTextFeatDataset
from .ogbn_datasets import OGBTextFeatDataset
from .dataset import ConstructedGraphDataset
try:
    # These two datasets require M5 support
    from .m5_dataset import StandardM5gnnDataset
except:
    print('No m5 suppport')
from .utils import generated_train_valid_test_splits, adjust_eval_mapping_for_partition
