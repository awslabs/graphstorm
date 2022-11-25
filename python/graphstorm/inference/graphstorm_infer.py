""" Infererence framework
"""
import dgl
import torch as th

class GSInfer():
    """ Generic GSgnn infer.


    Parameters
    ----------
    config: GSConfig
        Task configuration
    """
    def __init__(self):
        """empty
        """

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
