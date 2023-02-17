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
