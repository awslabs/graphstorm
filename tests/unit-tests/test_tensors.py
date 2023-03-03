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
import torch as th
import numpy as np
from graphstorm.model.utils import LazyDistTensor

def test_lazy_tensor():
    tensor = th.rand(100, 10)
    idx = th.tensor([1, 5, 10, 6, 7])
    t1 = LazyDistTensor(tensor, idx)
    t2 = tensor[idx]
    assert np.all(t1.shape == t2.shape)
    assert np.all(t2.numpy() == t1[0:len(t1)].numpy())

if __name__ == '__main__':
    test_lazy_tensor()
