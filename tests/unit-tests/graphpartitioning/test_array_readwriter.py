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
import pytest
import torch

from graphstorm.graphpartitioning import array_readwriter


@pytest.mark.parametrize(
    "shape", [[500, 1], [300, 10], [200, 5, 5], [100, 5, 5, 5]]
)
@pytest.mark.parametrize("format", ["numpy", "parquet"])
def test_array_readwriter(format, shape):
    """Unit test for read/write operations on numpy and parquet files."""
    original_array = np.random.rand(*shape)
    fmt_meta = {"name": format}

    with tempfile.TemporaryDirectory() as test_dir:
        path = os.path.join(test_dir, f"nodes.{format}")
        array_readwriter.get_array_parser(**fmt_meta).write(
            path, original_array
        )
        array = array_readwriter.get_array_parser(**fmt_meta).read(path)

        assert original_array.shape == array.shape
        assert np.array_equal(original_array, array)


@pytest.mark.parametrize("shape", [[500, 1], [300, 10]])
def test_csv_array_readwriter(shape):
    test_array_readwriter("csv", shape)


if __name__ == "__main__":
    test_array_readwriter()
    test_csv_array_readwriter()
