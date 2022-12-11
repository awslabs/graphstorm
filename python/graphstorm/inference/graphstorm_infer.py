""" Infererence framework
"""
import dgl
import torch as th
from ..tracker import get_task_tracker_class

class GSInfer():
    """ Generic GSgnn infer.


    Parameters
    ----------
    config: GSConfig
        Task configuration
    """
    def __init__(self, config):
        self.config = config

        self.device = f'cuda:{int(config.local_rank)}'
        self.init_dist_context(config.ip_config,
                               config.graph_name,
                               config.part_config,
                               config.backend)
        self.evaluator = None
        tracker_class = get_task_tracker_class(config.task_tracker)
        self.task_tracker = tracker_class(config, self._g.rank())

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

    def log_print_metrics(self, val_score, test_score, dur_eval, total_steps, train_score=None):
        """
        This function prints and logs all the metrics for evaluation

        Parameters
        ----------
        train_score: dict
            Training score
        val_score: dict
            Validation score
        test_score: dict
            Test score
        dur_eval:
            Total evaluation time
        total_steps: int
            The corresponding step/iteration
        """
        if self.task_tracker is None:
            return

        best_val_score = self.evaluator.best_val_score
        best_test_score = self.evaluator.best_test_score
        best_iter_num = self.evaluator.best_iter_num
        self.task_tracker.log_iter_metrics(self.evaluator.metric,
                train_score=train_score, val_score=val_score,
                test_score=test_score, best_val_score=best_val_score,
                best_test_score=best_test_score, best_iter_num=best_iter_num,
                eval_time=dur_eval, total_steps=total_steps)
