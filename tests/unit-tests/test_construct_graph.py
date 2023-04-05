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
import os
import tempfile
import numpy as np
import graphstorm as gs

def test_parquet():
    from graphstorm.gconstruct import write_data_parquet, read_data_parquet
    handle, tmpfile = tempfile.mkstemp()
    os.close(handle)

    data = {}
    data["data1"] = np.random.rand(10, 3)
    data["data2"] = np.random.rand(10)
    write_data_parquet(data, tmpfile)
    data1 = read_data_parquet(tmpfile)
    assert len(data1) == 2
    assert "data1" in data1
    assert "data2" in data1
    assert np.all(data1['data1'] == data['data1'])
    assert np.all(data1['data2'] == data['data2'])

    data1 = read_data_parquet(tmpfile, data_fields=['data1'])
    assert len(data1) == 1
    assert "data1" in data1
    assert "data2" not in data1
    assert np.all(data1['data1'] == data['data1'])

def test_feat_ops():
    from graphstorm.gconstruct import parse_feat_ops, process_features

    feat_op1 = [{
        "feature_col": "test1",
        "feature_name": "test2",
    }]
    res1 = parse_feat_ops(feat_op1)
    assert len(res1) == 1
    assert len(res1[0]) == 4
    assert res1[0][0] == feat_op1[0]["feature_col"]
    assert res1[0][1] == feat_op1[0]["feature_name"]
    assert res1[0][2] is None   # dtype is always None for now.
    assert res1[0][3] is None   # There is not transformation.

    feat_op2 = [
        {
            "feature_col": "test1",
            "feature_name": "test2",
        },
        {
            "feature_col": "test3",
            "feature_name": "test4",
            "transform": {"name": 'tokenize_hf',
                'bert_model': 'bert-base-uncased',
                'max_seq_length': 16
            },
        },
    ]
    res2 = parse_feat_ops(feat_op2)
    assert len(res2) == 2
    assert len(res2[0]) == 4
    assert res2[1][0] == feat_op2[1]["feature_col"]
    assert res2[1][1] == feat_op2[1]["feature_name"]
    assert res2[1][2] is None   # dtype is always None for now.
    op = res2[1][3]
    tokens = op(["hello world", "hello world"])
    assert len(tokens) == 3
    assert tokens['token_ids'].shape == (2, 16)
    assert tokens['attention_mask'].shape == (2, 16)
    assert tokens['token_type_ids'].shape == (2, 16)
    assert np.all(tokens['token_ids'][0] == tokens['token_ids'][1])
    assert np.all(tokens['attention_mask'][0] == tokens['attention_mask'][1])
    assert np.all(tokens['token_type_ids'][0] == tokens['token_type_ids'][1])

    data = {
        "test1": np.random.rand(2, 4),
        "test3": ["hello world", "hello world"],
    }
    proc_res = process_features(data, res2)
    assert np.all(data['test1'] == proc_res['test2'])
    assert "token_ids" in proc_res
    assert "attention_mask" in proc_res
    assert "token_type_ids" in proc_res

def test_label():
    from graphstorm.gconstruct import process_labels
    data = {
        "label": np.random.randint(5, size=10),
    }
    label_conf = [
        {
            "label_col": "label",
            "task_type": "classification",
            "split_type": [0.8, 0.1, 0.1],
        },
    ]
    res = process_labels(data, label_conf)
    assert np.all(res['label'] == data['label'])
    assert res['train_mask'].shape == (len(data['label']),)
    assert res['val_mask'].shape == (len(data['label']),)
    assert res['test_mask'].shape == (len(data['label']),)
    assert np.sum(res['train_mask']) == 8
    assert np.sum(res['val_mask']) == 1
    assert np.sum(res['test_mask']) == 1

if __name__ == '__main__':
    test_parquet()
    test_feat_ops()
    test_label()
