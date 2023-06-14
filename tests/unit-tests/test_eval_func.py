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
import numpy as np
import torch as th

from numpy.testing import assert_almost_equal
from graphstorm.eval.eval_func import compute_mse, compute_rmse

def test_compute_mse():
    pred64 = th.rand((100,1), dtype=th.float64)
    label64 = pred64 + th.rand((100,1), dtype=th.float64) / 10

    pred32 = pred64.type(th.float32)
    label32 = label64.type(th.float32)
    mse32 = compute_mse(pred32, label32)

    mse_pred64 = compute_mse(pred64, label32)
    mse_label64 = compute_mse(pred32, label64)

    assert_almost_equal(mse32, mse_pred64)
    assert_almost_equal(mse32, mse_label64)

def test_compute_rmse():
    pred64 = th.rand((100,1), dtype=th.float64)
    label64 = pred64 + th.rand((100,1), dtype=th.float64) / 10

    pred32 = pred64.type(th.float32)
    label32 = label64.type(th.float32)
    rmse32 = compute_rmse(pred32, label32)

    rmse_pred64 = compute_rmse(pred64, label32)
    rmse_label64 = compute_rmse(pred32, label64)

    assert_almost_equal(rmse32, rmse_pred64)
    assert_almost_equal(rmse32, rmse_label64)

if __name__ == '__main__':
    test_compute_mse()
    test_compute_rmse()
