"""Package initialization. Import all available dataset and functions
"""
from .movielens import MovieLens100kNCDataset
from .ogbn_arxiv import OGBArxivTextFeatDataset
from .ogbn_datasets import OGBTextFeatDataset
from .mag_lsc import MAGLSCDataset
from .dataset import ConstructedGraphDataset
from .utils import generated_train_valid_test_splits, adjust_eval_mapping_for_partition
