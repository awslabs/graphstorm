"""
Task tracker

Avaliable task tracker:
StdoutTracker
FileTracker
TODO(MLFlowTracker)
"""

def init_tracker(config):
    # TODO(xiangsx): Add more trackers
    return BasicTracker(config)

class BasicTracker():
    """ BasicTracker

    Use stdout to print the log

    Parameters
    ----------
    config:
        Configurations
    """
    def __init__(self, config):
        self._task_name = config.task_name if hasattr(config, "task_name") else "Task"
        self._report_frequency = config.report_frequency if hasattr(config, "report_frequency") else 100 # report every 100 steps

    def log_param(self, param_name, param_value, step=None):
        if step % self.report_frequency == 0:
            print("[INFO: {}][{}={}]".format(self.task_name, param_name, param_value))

    def log_params(self, params):
        for param_name, param_value in params.items():
            self.log_param(param_name, param_value)

    def log_metric(self, metric_name, metric_value, step):
        if step % self.report_frequency == 0:
            print("[METRIC INFO: {}][Step {}][{}={}]".format(
                self.task_name, step, metric_name, metric_value))

    @property
    def task_name(self):
        return self._task_name

    @property
    def report_frequency(self):
        return self._report_frequency
