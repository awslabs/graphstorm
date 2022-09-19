"""
Test functions and classes in the dataloading.py
"""

import torch as th
from data_utils import generate_dummy_dist_graph

from graphstorm.model.dataloading import GSgnnLinkPredictionTrainData


def test_GSgnnLinkPredictionTrainData():

    # get the test dummy distributed graph
    dist_graph = generate_dummy_dist_graph()
    pb = dist_graph.get_partition_book()

    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='nccl',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    tr_etypes = [("n0", "r1", "n1")]
    va_etypes = [("n0", "r1", "n1")]
    ts_etypes = [("n0", "r1", "n1")]
    lp_data = GSgnnLinkPredictionTrainData(dist_graph, pb=pb,
                                          train_etypes=tr_etypes, eval_etypes=va_etypes,
                                          full_graph_training=False)
    # successful initialization with default setting
    assert lp_data.train_etypes == tr_etypes
    assert lp_data.eval_etypes == va_etypes
    assert lp_data.full_graph_training == False

    # sucessfully split train/val/test idxs
    assert len(lp_data.train_idxs) == len(tr_etypes)
    for etype in tr_etypes:
        assert th.all(lp_data.train_idxs[etype[1]] == th.tensor([0,1]))

    assert len(lp_data.val_idxs) == len(va_etypes)
    for etype in va_etypes:
        assert th.all(lp_data.val_idxs[etype[1]] == th.tensor([2,3]))

    assert len(lp_data.test_idxs) == len(ts_etypes)
    for etype in ts_etypes:
        assert th.all(lp_data.test_idxs[etype[1]] == th.tensor([4,5]))

    # successful set configuration
    assert lp_data.do_validation == True

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()


if __name__ == '__main__':
    test_GSgnnLinkPredictionTrainData()