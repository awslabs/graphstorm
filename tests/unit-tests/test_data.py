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

    tools to test the preprocess of movielens100k data
"""

import os
from graphstorm.data import MovieLens100kNCDataset


def test_moveliens100k_dataset_normal():
    # default and correct dataset configurations
    dataset_config = {
        'raw_dir': '/data',
        'max_sequence_length': 512,
        'use_text_feat': False,
        'use_age_as_label': False,
        'save_path': '/data/data/'
    }

    dataset = MovieLens100kNCDataset(raw_dir=dataset_config['raw_dir'],
                                     max_sequence_length=dataset_config['max_sequence_length'],
                                     use_text_feat=dataset_config['use_text_feat'],
                                     user_age_as_label=dataset_config['use_age_as_label'])

    # test dataset own property
    assert dataset.predict_category == 'movie', f"The predict category should be \"movie\" \
                                                  but got {dataset.predict_category}"
    assert dataset.num_classes == 19, f"The number of classes should be 19, \
                                       but got {dataset.num_classes}"

    # test graph data property
    assert dataset._g, f"Should have build a DGL graph!"
    assert len(dataset._g.ntypes) == 2, f"MovieLen100k should have 2 types of nodes, \
                                         but got {len(dataset._g.ntypes)}"
    assert len(dataset._g.canonical_etypes) == 2, f"MovieLen100k should have 10 types of edges, \
                                         but got {len(dataset._g.canonical_etypes)}"
    assert dataset._g.num_nodes('movie') == 1682, f"MovieLen100k should have 1682 \"movie\" nodes, \
                                                    but got {dataset._g.num_nodes('movie')}"
    assert dataset._g.num_nodes('user') == 943, f"MovieLen100k should have 943 \"user\" nodes, \
                                                  but got {dataset._g.num_nodes('user')}"

    # test save graph to local file
    dataset.save_graph(dataset_config['save_path'])

    assert os.path.exists(os.path.join(dataset_config['save_path'],
                                      'ml-100k.bin')), f"Should have a file named after \
                                      \"ml-100k.bin\" saved at {dataset_config['save_path']}, \
                                      but not."


if __name__ == "__main__":
    test_moveliens100k_dataset_normal()
