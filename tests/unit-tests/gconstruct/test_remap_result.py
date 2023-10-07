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
import pytest
import os
import tempfile
import pytest

from graphstorm.gconstruct.remap_result import _get_file_range

def test__get_file_range():
    start, end = _get_file_range(10, 0, 0)
    assert start == 0
    assert end == 10

    start, end = _get_file_range(10, 0, 1)
    assert start == 0
    assert end == 10

    start, end = _get_file_range(10, 0, 2)
    assert start == 0
    assert end == 5
    start, end = _get_file_range(10, 1, 2)
    assert start == 5
    assert end == 10

    start, end = _get_file_range(10, 0, 3)
    assert start == 0
    assert end == 3
    start, end = _get_file_range(10, 1, 3)
    assert start == 3
    assert end == 6
    start, end = _get_file_range(10, 2, 3)
    assert start == 6
    assert end == 10

if __name__ == '__main__':
    test__get_file_range()