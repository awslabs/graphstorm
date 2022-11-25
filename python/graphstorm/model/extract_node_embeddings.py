import torch as th
import dgl
from torch.nn.parallel import DistributedDataParallel

def prepare_batch_input(g, input_nodes,
                        dev='cpu', verbose=False, feat_field='feat'):
    """ Prepare minibatch input features

    Note: The output is stored in dev.

    Parameters
    ----------
    g: DGLGraph
        The graph.
    input_nodes: dict of tensor
        Input nodes.
    dev: th.device
        Device to put output in.
    verbose: bool
        Whether to print extra infor
    feat_field: str
        field to extract features

    Return:
    Dict of tensors.
        If a node type has features, it will get node features.
        If a node type is feature-less, the node id is returned as index to the trainable embeddings.

    """
    feat = {}
    for ntype, nid in input_nodes.items():
        feat_name = None if feat_field is None else \
            feat_field if isinstance(feat_field, str) \
            else feat_field[ntype] if ntype in feat_field else None

        if feat_name is not None:
            feat[ntype] = g.nodes[ntype].data[feat_name][nid].to(dev)
        if ntype not in feat: ## node has no features, then extract the sparse encoding
            feat[ntype] = nid.to(dev)
    return feat

def extract_all_embeddings_dist(g, batch_size, embed_layer, dev, task_tracker=None, feat_field='feat'):
    """
    This function extracts the embeddings for all the nodes in a distributed graph either from the node features
    or from the embedding layer.
    Parameters
    ----------
    g
    batch_size
    embed_layer
    dev

    Returns
    -------

    """
    if embed_layer is not None:
        embed_layer.eval()

    n_embs = {}
    th.cuda.empty_cache()
    with th.no_grad():
        for ntype in g.ntypes:
            assert embed_layer is not None, "The input embedding layer is needed"
            if 'input_emb' not in g.nodes[ntype].data:
                embed_size = embed_layer.module.embed_size \
                        if isinstance(embed_layer, DistributedDataParallel) else embed_layer.embed_size
                g.nodes[ntype].data['input_emb'] = dgl.distributed.DistTensor(
                        (g.number_of_nodes(ntype), embed_size),
                        dtype=th.float32, name='{}_input_emb'.format(ntype),
                        part_policy=g.get_node_partition_policy(ntype),
                        persistent=True)
            input_emb = g.nodes[ntype].data['input_emb']
            infer_nodes = dgl.distributed.node_split(th.ones((g.number_of_nodes(ntype),), dtype=th.bool),
                                                     partition_book=g.get_partition_book(),
                                                     ntype=ntype, force_even=False)
            node_list = th.split(infer_nodes, batch_size)
            for iter_l, input_nodes in enumerate(node_list):
                if iter_l % 10000 == 0:
                    print ("extract_all_embeddings_dist on {}: {} of {}".format(ntype, iter_l, len(node_list)))
                if task_tracker is not None:
                    task_tracker.keep_alive(iter_l)

                feat = prepare_batch_input(g, {ntype: input_nodes}, dev=dev, feat_field=feat_field)
                assert ntype in feat
                emb = embed_layer(feat, ntype, {ntype: input_nodes})
                input_emb[input_nodes] = emb[ntype].to('cpu')
            n_embs[ntype] = input_emb
        if g.rank() == 0:
            print("Extract node embeddings")
    if embed_layer is not None:
        embed_layer.train()
    g.barrier()
    return n_embs
