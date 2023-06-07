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

import numpy as np
from numpy.testing import assert_equal

from graphstorm.gconstruct.transform import FloatingPointTransform

def test_fp_transform():
    transform = FloatingPointTransform("test", "test")
    feats = np.random.randn(100)

    max_val, min_val = transform.pre_process(feats)
    max_v = np.amax(feats)
    min_v = np.amin(feats)
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert_equal(max_val[0], max_v)
    assert_equal(min_val[0], min_v)

    feats = np.random.randn(100, 10)
    max_val, min_val = transform.pre_process(feats)
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert len(max_val) == 10
    assert len(min_val) == 10
    for i in range(10):
        max_v = np.amax(feats[:i])
        min_v = np.amin(feats[:i])
        assert_equal(max_val[i], max_v)
        assert_equal(min_val[i], min_v)

    feats = np.r.random.randn(100)
    feats[0] = 10.
    feats[1] = -10.
    transform = FloatingPointTransform("test", "test", max_bound=5., min_bound=-5.)
    max_val, min_val = transform.pre_process(feats)
    max_v = np.amax(feats)
    min_v = np.amin(feats)
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert_equal(max_val[0], 5.)
    assert_equal(min_val[0], -5.)

    feats = np.r.random.randn(100, 10)
    feats[0] = 10.
    feats[1] = -10.
    transform = FloatingPointTransform("test", "test", max_bound=5., min_bound=-5.)
    max_val, min_val = transform.pre_process(feats)
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert len(max_val) == 10
    assert len(min_val) == 10
    for i in range(10):
        max_v = np.amax(feats[:i])
        min_v = np.amin(feats[:i])
        assert_equal(max_val[i], 5.)
        assert_equal(min_val[i], -5.)




if __name__ == '__main__':
    test_fp_transform()