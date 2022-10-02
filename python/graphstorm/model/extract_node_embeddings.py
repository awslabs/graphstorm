import torch as th
import dgl
from torch.nn.parallel import DistributedDataParallel

from .hbert import extract_bert_embed
from ..data.constants import TOKEN_IDX

def generate_pretrained_bert_embeddings(g, batch_size, bert_train, bert_static, bert_hidden_size, dev, verbose=False):
    """This function is to infer the bert embeddings for a large number of data points.

    Note: The output is stored in CPU

    Parameters
    ----------
    g: DGLGraph
        The graph.
    batch_size: int
        The batch size
    bert_trains: Wrapper for trainable bert
        The bert model wrapper used to generate embeddings for trainable text nodes.
    bert_static: Wrapper for static bert
        The bert model wrapper used to generate embeddings for static text bides.
    bert_hidden_size: int
        The bert embedding hidden size.
    dev: th.device
        Device to collect tmp embeddings.
    """
    for ntype in g.ntypes:
        if ntype in bert_train.keys():
            bert_train[ntype].eval()
            bert_static[ntype].eval()

    out_embs = {}
    with th.no_grad():
        for ntype in g.ntypes:
            if TOKEN_IDX in g.nodes[ntype].data:
                print("generating BERT embedding for %s" % ntype)
                bsize = batch_size
                n_nodes = g.num_nodes(ntype)
                nodes = th.arange(n_nodes)
                embs = []
                for i in range(int((n_nodes+bsize-1)/bsize)):
                    nid = nodes[i*bsize:(i+1)*bsize \
                        if (i+1)*bsize < n_nodes else n_nodes]
                    mask = th.full((nid.shape[0],), False, dtype=th.bool)
                    text_embs, _ = extract_bert_embed(nid=nid,
                                                    mask=mask,
                                                    bert_train=bert_train[ntype],
                                                    bert_static=bert_static[ntype],
                                                    bert_hidden_size=bert_hidden_size[ntype] \
                                                        if isinstance(bert_hidden_size, dict) \
                                                        else bert_hidden_size,
                                                    dev=dev,
                                                    verbose=verbose)
                    if verbose and i%100==0:
                        print( "Processed "+ str((i+1)*bsize)+" nodes")
                    embs.append(text_embs.to('cpu'))
                embs = th.cat(embs, dim=0)
                out_embs[ntype] = embs

    for ntype in g.ntypes:
        if ntype in bert_train.keys():
            bert_train[ntype].train()
            bert_static[ntype].train()

    return out_embs

def prepare_batch_input(g, bert_trains, bert_statics, bert_hidden_size, input_nodes,
                        bert_infer_bs=128, train_mask=None, emb_cache=None,
                        dev='cpu', verbose=False, feat_field='feat'):
    """ Prepare minibatch input features

    Note: The output is stored in dev.

    Parameters
    ----------
    g: DGLGraph
        The graph.
    bert_trains: Wrapper for trainable bert
        The bert model wrapper used to generate embeddings for trainable text nodes.
    bert_statics: Wrapper for static bert
        The bert model wrapper used to generate embeddings for static text bides.
    bert_hidden_size: int
        The bert embedding hidden size.
    input_nodes: dict of tensor
        Input nodes.
    bert_infer_bs: int
        Batch size when doing bert inference
    train_mask: dict of tensor
        A dict of boolean tensors indicating whether the corresponding nodes will be involved in back-propagation.
    emb_cache: dict of tensor
        A bert embedding cache
    dev: th.device
        Device to put output in.
    verbose: bool
        Whether to print extra infor
    feat_field: str
        field to extract features

    Return:
    Dict of tensors.
        If a node type has text feature, the feature will be encoded using the corresponding bert model.
        If a node type has non-text feature, it will be concatenated with the text feature if there is any.
        If a node type is feature-less, the node id is returned as index to the trainable embeddings.

    """
    emb = {}
    losses = {}
    for ntype, nid in input_nodes.items():
        mask = train_mask[ntype] if train_mask is not None else None
        if TOKEN_IDX in g.nodes[ntype].data:
            text_embs, loss = extract_bert_embed(nid=nid,
                                                 mask=mask,
                                                 bert_train=bert_trains[ntype],
                                                 bert_static=bert_statics[ntype],
                                                 emb_cache=emb_cache[ntype] if emb_cache is not None else None,
                                                 bert_hidden_size=bert_hidden_size[ntype] \
                                                    if isinstance(bert_hidden_size, dict) \
                                                    else bert_hidden_size,
                                                 dev=dev,
                                                 verbose=verbose)
            emb[ntype] = text_embs.type(th.float32)
            losses[ntype] = loss

        feat_name = None if feat_field is None else \
            feat_field if isinstance(feat_field, str) \
            else feat_field[ntype] if ntype in feat_field else None

        if feat_name is not None:
            if ntype in emb:
                emb[ntype] = th.cat((g.nodes[ntype].data[feat_name][nid].to(dev), emb[ntype]), dim=1)
            else:
                emb[ntype] = g.nodes[ntype].data[feat_name][nid].to(dev)
        if ntype not in emb: ## node has no text, no feature, then extract the sparse encoding
            emb[ntype] = nid.to(dev)
    return emb, losses

def extract_bert_embeddings_dist(g, batch_size, bert_train, bert_static, bert_hidden_size, dev, verbose=False,
                                 task_tracker=None):
    """
    This function extracts the bert embeddings for all the text nodes in a distributed graph.

    Parameters
    ----------
    g
    batch_size
    embed_layer
    bert_train
    bert_static
    bert_hidden_size
    dev

    Returns
    -------

    """
    for ntype in g.ntypes:
        if ntype in bert_train.keys():
            bert_train[ntype].eval()
            bert_static[ntype].eval()

    out_embs = {}
    with th.no_grad():
        for ntype in g.ntypes:
            if TOKEN_IDX in g.nodes[ntype].data:
                if g.rank() == 0:
                    print('compute bert embedding on node {}'.format(ntype))
                hidden_size = bert_hidden_size[ntype] if isinstance(bert_hidden_size, dict) else bert_hidden_size
                if 'bert_emb' not in g.nodes[ntype].data:
                    g.nodes[ntype].data['bert_emb'] = dgl.distributed.DistTensor((g.number_of_nodes(ntype), hidden_size),
                                                                                  name="bert_emb",
                                                                                  dtype=th.float32,
                                                                                  part_policy=g.get_node_partition_policy(ntype),
                                                                                  persistent=True)
                input_emb = g.nodes[ntype].data['bert_emb']
                infer_nodes = dgl.distributed.node_split(th.ones((g.number_of_nodes(ntype),), dtype=th.bool),
                                                        partition_book=g.get_partition_book(),
                                                        ntype=ntype, force_even=False)
                node_list = th.split(infer_nodes, batch_size)
                for iter_l, input_nodes in enumerate(node_list):
                    if task_tracker is not None:
                        task_tracker.keep_alive(iter_l)

                    mask = th.full((input_nodes.shape[0],), False, dtype=th.bool)
                    text_embs, _ = extract_bert_embed(nid=input_nodes,
                                                      mask=mask,
                                                      bert_train=bert_train[ntype],
                                                      bert_static=bert_static[ntype],
                                                      bert_hidden_size=hidden_size,
                                                      dev=dev,
                                                      verbose=verbose)
                    input_emb[input_nodes] = text_embs.to('cpu')
                out_embs[ntype] = input_emb

    for ntype in g.ntypes:
        if ntype in bert_train.keys():
            bert_train[ntype].train()
            bert_static[ntype].train()
    g.barrier()
    return out_embs

def extract_all_embeddings_dist(g, batch_size, embed_layer, bert_train, bert_static,
    bert_hidden_size, dev, emb_cache=None, task_tracker=None, feat_field='feat'):
    """
    This function extracts the embeddings for all the nodes in a distributed graph either from the BERT model
    or from the embedding layer.
    Parameters
    ----------
    g
    batch_size
    embed_layer
    bert_train
    bert_static
    bert_hidden_size
    dev

    Returns
    -------

    """
    if embed_layer is not None:
        embed_layer.eval()
    for ntype in g.ntypes:
        if len(bert_train) > 0 and ntype in bert_train.keys():
            bert_train[ntype].eval()
            bert_static[ntype].eval()

    n_embs = {}
    th.cuda.empty_cache()
    with th.no_grad():
        for ntype in g.ntypes:
            if 'input_emb' not in g.nodes[ntype].data:
                if embed_layer is not None:
                    embed_size = embed_layer.module.embed_size \
                            if isinstance(embed_layer, DistributedDataParallel) else embed_layer.embed_size
                else:
                    embed_size = bert_hidden_size[ntype]
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

                train_mask = {ntype: th.full((input_nodes.shape[0],), False, dtype=th.bool)}
                emb, _ = prepare_batch_input(g, bert_train, bert_static, bert_hidden_size, {ntype: input_nodes},
                                                  train_mask=train_mask, emb_cache=emb_cache, dev=dev, feat_field=feat_field)
                if ntype not in emb:
                    emb[ntype]=None
                    assert embed_layer is not None, "In this case the embedding layer is needed"
                if embed_layer is not None:
                    emb = embed_layer(emb, ntype, {ntype: input_nodes})
                input_emb[input_nodes] = emb[ntype].to('cpu')
            n_embs[ntype] = input_emb
        if g.rank() == 0:
            print("Extract node embeddings")
    if embed_layer is not None:
        embed_layer.train()
    for ntype in g.ntypes:
        if len(bert_train) > 0 and ntype in bert_train.keys():
            bert_train[ntype].train()
            bert_static[ntype].train()
    g.barrier()
    return n_embs
