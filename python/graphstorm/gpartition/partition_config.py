from dataclasses import dataclass

@dataclass
class ParMETISConfig:
    """
    Dataclass for holding the configuration for a ParMETIS partitioning algorithm.

    Parameters
    ----------

    """
    ip_list: str
    input_path: str
    dgl_tool_path: str
    metadata_filename: str