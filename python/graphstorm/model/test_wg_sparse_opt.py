import torch
import os
import argparse
import torch.distributed as dist
import pylibwholegraph.torch as wgth
import pylibwholegraph.binding.wholememory_binding as wmb
torch.set_printoptions(precision=8)

class wholegraph_config:
    """Add/initialize default options required for distributed launch incorprating with wholegraph

    NOTE: This class might be deprecated soon once wholegraph's update its configuration API.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.launch_env_name_world_rank = "RANK"
        self.launch_env_name_world_size = "WORLD_SIZE"
        self.launch_env_name_master_addr = "MASTER_ADDR"
        self.launch_env_name_master_port = "MASTER_PORT"
        self.launch_env_name_local_size = "LOCAL_WORLD_SIZE"
        self.launch_env_name_local_rank = "LOCAL_RANK"
        if self.launch_agent == "mpi":
            self.master_addr = "" # pick from env var
            self.master_port = -1 # pick from env var
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
        if "LOCAL_WORLD_SIZE" in os.environ:
            self.local_size = int(os.environ["LOCAL_WORLD_SIZE"])

        # make sure the following arguments are avail for wholegraph
        assert self.local_rank is not None
        assert self.local_size is not None and self.local_size > 0

# initialize wholegraph and return its global communicator
def init_wholegraph():
    config = wholegraph_config(launch_agent="pytorch")
    wgth.distributed_launch(config, lambda: None)
    wmb.init(0)
    wgth.comm.set_world_info(wgth.get_rank(), wgth.get_world_size(), wgth.get_local_rank(), wgth.get_local_size(),)
    global_comm = wgth.comm.get_global_communicator('nccl')
    return global_comm

def create_wg_sparse_params(nnodes, embedding_dim, wm_adam, location='cpu'): # location = ['cpu'|'cuda']

    global_comm = wgth.comm.get_global_communicator()
    embedding_wholememory_type = 'distributed'
    embedding_wholememory_location = location
    dist_embedding = wgth.create_embedding(global_comm,
                                           embedding_wholememory_type,
                                           embedding_wholememory_location,
                                           # due to here: https://github.com/dmlc/dgl/blob/master/python/dgl/distributed/nn/pytorch/sparse_emb.py#L79C12-L79C23
                                           torch.float32,
                                           [nnodes, embedding_dim],
                                           optimizer=wm_adam,
                                           cache_policy=None, # disable cache for now
                                           random_init=False,
                                           )
    torch.manual_seed(0)
    local_tensor, local_offset = dist_embedding.get_embedding_tensor().get_local_tensor()
    # fill the value to match pytoch reference in the next
    xx=torch.zeros(nnodes, embedding_dim, dtype=torch.float32)
    torch.manual_seed(0)
    torch.nn.init.xavier_uniform_(xx)
    num_parts = dist.get_world_size()
    subpart_size = -(nnodes// -num_parts)

    for part in range(num_parts):
        l = part*subpart_size
        u = (part+1)*subpart_size if part != (num_parts - 1) else nnodes
        if part == dist.get_rank():
            local_tensor.copy_(xx[l:u])


    return dist_embedding

def run(num_nodes, args):
    # Find corresponding device for current process.
    emb_dim = args.embed_dim
    lr = 0.01
    wm_adam = wgth.create_wholememory_optimizer("adam", {})
    # create embedding (serial for torch; distributed for wholememory)
    torch.manual_seed(0)
    wg_emb = create_wg_sparse_params(num_nodes, emb_dim, wm_adam)
    torch_emb = torch.nn.Embedding(num_nodes, emb_dim, sparse=True)
    # initialize the embedding (make sure the two embs are identical)
    torch.manual_seed(0)
    torch.nn.init.xavier_uniform_(torch_emb.weight)
    torch_adam = torch.optim.SparseAdam(torch_emb.parameters(), lr=lr)

    # total index to be updated (no duplicate, not sorted)
    idx_g = torch.randperm(num_nodes, dtype=torch.int64)[:args.num_indices]
    # local index, partitioned from total index
    idx = torch.tensor_split(idx_g, dist.get_world_size())[dist.get_rank()]
    # total labels
    labels_g = torch.ones((len(idx_g),)).long()
    # local labels
    labels = torch.ones((len(idx),)).long()

    # forward/backward of pytorch flow
    torch_value = torch_emb(idx_g)
    torch_adam.zero_grad()
    torch_loss = torch.nn.functional.cross_entropy(torch_value, labels_g)
    torch_loss.backward()
    torch_adam.step()

    # wrap wg emb into nn module
    MODEL = wgth.WholeMemoryEmbeddingModule(wg_emb)
    MODEL.train()
    # forward/backward of wholegraph flow
    wg_value = MODEL(idx.cuda()).cpu()
    wg_loss = torch.nn.functional.cross_entropy(wg_value, labels)
    wg_loss.backward()
    wm_adam.step(lr)
    torch.cuda.synchronize()
    dist.barrier()

    # collect results (torch_emb are the same across all ranks)
    torch_weight=torch_emb.weight.detach()
    all_nodes = torch.arange(num_nodes).cuda()
    wg_weight=wg_emb.gather(all_nodes).cpu()

    # calculate diff
    x=torch_weight[idx_g] - wg_weight[idx_g]
    rel_error = torch.abs(x)/torch.abs(torch_weight[idx_g])
    # order the diff so that largest value print first
    ordered_error, order =rel_error.flatten().sort(descending=True)

    if (dist.get_rank() == 0):
        print('Ref err are:', ordered_error[:20])
        assert ordered_error[0] < 1e-3, "ref error should be always less than 1e-3..."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=4,
        help="Embedding dimension for each node.",
    )
    parser.add_argument(
        "--num-indices",
        type=int,
        default=4,
        help="Number of selected indexes (in total of all ranks) to update embedding table.",
    )
    args = parser.parse_args()
    assert (
        torch.cuda.is_available()
    ), f"Must have GPUs to enable multi-gpu training."
    num_nodes = 200
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.cuda.current_device()
    dist.init_process_group(backend="nccl")
    global_comm = init_wholegraph()

    run(num_nodes, args)
