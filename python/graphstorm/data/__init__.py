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

    Package initialization. Import all available built-in dataset and functions
"""
from .movielens import MovieLens100kNCDataset
from .ogbn_datasets import OGBTextFeatDataset
from .mag_lsc import MAGLSCDataset
from .dataset import ConstructedGraphDataset
from .utils import generated_train_valid_test_splits, adjust_eval_mapping_for_partition
