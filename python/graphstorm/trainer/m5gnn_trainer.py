import dgl
import torch as th

class M5gnnTrainer():
    """ Generic M5GNN trainer.


    Parameters
    ----------
    config: M5GNNConfig
        Task configuration
    bert_model: dict
        A dict of BERT models in the format of node-type -> M5 BERT model
    """
    def __init__(self):
        super(M5gnnTrainer, self).__init__()

    def init_dist_context(self, ip_config, graph_name, part_config, backend):
        """ Initialize distributed inference context

        Parameters
        ----------
        ip_config: str
            File path of ip_config file
        graph_name: str
            Name of the graph
        part_config: str
            File path of partition config
        backend: str
            Torch distributed backend
        """

        # We need to use socket for communication in DGL 0.8. The tensorpipe backend has a bug.
        # This problem will be fixed in the future.
        dgl.distributed.initialize(ip_config, net_type='socket')
        self._g = dgl.distributed.DistGraph(graph_name, part_config=part_config)
        print("Start init distributed group ...")
        th.distributed.init_process_group(backend=backend)
