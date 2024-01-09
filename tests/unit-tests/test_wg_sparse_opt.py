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

    Unitest code for model save and restore
"""
import os
import dgl
import pytest

import torch as th
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn

from graphstorm.wholegraph import init_wholegraph
from graphstorm.utils import use_wholegraph_sparse_emb, is_wholegraph_sparse_emb


def initialize(use_wholegraph=True):

    from dgl.distributed import role
    role.init_role("default")
    os.environ["DGL_DIST_MODE"] = "standalone"

    backend = "nccl"
    assert th.cuda.is_available(), "NCCL backend requires CUDA device(s) to be available."
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    th.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group(backend=backend, rank=0, world_size=1)
    if use_wholegraph:
        init_wholegraph()

def create_wg_sparse_params(wgth, nnodes, embedding_dim, sparse_opt, location='cpu'): # location = ['cpu'|'cuda']
    global_comm = wgth.comm.get_global_communicator()
    embedding_wholememory_type = 'distributed'
    embedding_wholememory_location = location
    dist_embedding = wgth.create_embedding(global_comm,
                                           embedding_wholememory_type,
                                           embedding_wholememory_location,
                                           # due to here: https://github.com/dmlc/dgl/blob/master/python/dgl/distributed/nn/pytorch/sparse_emb.py#L79C12-L79C23
                                           th.float32,
                                           [nnodes, embedding_dim],
                                           optimizer=sparse_opt,
                                           cache_policy=None, # disable cache for now
                                           random_init=False,
                                           )
    th.manual_seed(0)
    local_tensor, local_offset = dist_embedding.get_embedding_tensor().get_local_tensor()
    # fill the value to match pytoch reference in the next
    xx=th.zeros(nnodes, embedding_dim, dtype=th.float32)
    th.manual_seed(0)
    nn.init.xavier_uniform_(xx)
    num_parts = dist.get_world_size()
    subpart_size = -(nnodes// -num_parts)
    for part in range(num_parts):
        l = part*subpart_size
        u = (part+1)*subpart_size if part != (num_parts - 1) else nnodes
        if part == dist.get_rank():
            local_tensor.copy_(xx[l:u])
    return dist_embedding

@pytest.mark.parametrize("embed_dim", [4, 15, 32])
@pytest.mark.parametrize("num_indices", [4, 32])
def test_wg_sparse_opt(embed_dim, num_indices):
     # initialize the torch and wholegraph distributed environment
    wgth = pytest.importorskip("pylibwholegraph.torch")
    import pylibwholegraph
    if pylibwholegraph.__version__ < "24.02":
        pytest.skip("Skipping this accuracy test due to a dependency on a fix from rapidsai/wholegraph:PR#108, which is targeted for release in version 24.02.")
    use_wholegraph_sparse_emb()
    initialize(use_wholegraph=is_wholegraph_sparse_emb())

    lr = 0.01
    num_nodes = 200
    wm_adam = wgth.create_wholememory_optimizer("adam", {})
    # create two embeddings (serial for torch; distributed for wholememory)
    th.manual_seed(0)
    wg_emb = create_wg_sparse_params(wgth, num_nodes, embed_dim, wm_adam)
    torch_emb = nn.Embedding(num_nodes, embed_dim, sparse=True)
    # initialize the embeddings to identical values
    th.manual_seed(0)
    nn.init.xavier_uniform_(torch_emb.weight)
    torch_adam = th.optim.SparseAdam(torch_emb.parameters(), lr=lr)

    # total selected indexes to be updated (avoid duplicates, and not sorted)
    idx_g = th.randperm(num_nodes, dtype=th.int64)[:num_indices]
    # local index for wholegraph
    idx = th.tensor_split(idx_g, dist.get_world_size())[dist.get_rank()]
    # total labels
    labels_g = th.ones((len(idx_g),)).long()
    # local labels for wholegraph
    labels = th.ones((len(idx),)).long()

    # 5 forward/backward passes of pytorch
    for i in range(5):
        torch_value = torch_emb(idx_g)
        torch_adam.zero_grad()
        torch_loss = nn.functional.cross_entropy(torch_value, labels_g)
        torch_loss.backward()
        torch_adam.step()


    # wrap wg emb into nn module
    MODEL = wgth.WholeMemoryEmbeddingModule(wg_emb)
    # 5 forward/backward passes of wholegraph
    for i in range(5):
        wg_value = MODEL(idx.cuda()).cpu()
        wg_loss = nn.functional.cross_entropy(wg_value, labels)
        wg_loss.backward()
        wg_emb.wmb_optimizer.step(lr)


    # collect updated parameters
    torch_weight=torch_emb.weight.detach()
    all_nodes = th.arange(num_nodes).cuda()
    wg_weight=wg_emb.gather(all_nodes).cpu()
    # check if they close after 5 passes
    assert th.allclose(torch_weight, wg_weight, atol=1e-05, rtol=1e-03)
    if is_wholegraph_sparse_emb():
        wgth.finalize()
        import pylibwholegraph.torch.comm as wgth_comm
        wgth_comm.global_communicators = {}
        wgth_comm.local_node_communicator = None
        wgth_comm.local_device_communicator = None
    th.distributed.destroy_process_group()


if __name__ == '__main__':
    test_wg_sparse_opt(4, 4)
    test_wg_sparse_opt(15, 12)
    test_wg_sparse_opt(40, 30)