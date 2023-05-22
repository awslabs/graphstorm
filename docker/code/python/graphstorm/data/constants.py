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

    Define the constants for dataset and data processing.
"""
EDGE_SRC_IDX='src_id'
EDGE_DST_IDX='dst_id'
NODE_ID_IDX='id'

TOKEN_IDX = 'input_ids'
VALID_LEN = 'valid_len'
ATT_MASK_IDX = 'attention_mask'
TOKEN_TID_IDX = 'token_type_ids'
LABEL_IDX = 'labels'
TRAIN_IDX = 'train_mask'
VALID_IDX = 'val_mask'
TEST_IDX = 'test_mask'

REGRESSION_TASK = "regression"
CLASSIFICATION_TASK = "classification"
