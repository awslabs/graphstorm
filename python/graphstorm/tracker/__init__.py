""" GraphStorm task tracker

    Builtin training tracker supports:
     - GSSageMakerTaskTracker: GraphStorm SageMaker Task Tracker
"""
from .graphstorm_tracker import GSTaskTrackerAbc
from .sagemaker_tracker import GSSageMakerTaskTracker

def get_task_tracker_class(tracker_name):
    """ Get builtin task tracker

    Parameters
    ----------
    tracker_name: str
        task tracker name. 'SageMaker' for GSSageMakerTaskTracker
    """
    if tracker_name == 'SageMaker':
        # SageMaker tracker also works as normal print tracker
        return GSSageMakerTaskTracker
    # TODO: Support mlflow, etc.
    else:
        # by default use GSSageMakerTaskTracker
        return GSSageMakerTaskTracker
