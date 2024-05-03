from dataclasses import dataclass

@dataclass
class ParMETISConfig:
    """
    Dataclass for holding the configuration for a ParMETIS partitioning algorithm.

    Parameters
    ----------
    ip_list: str
        ip list
    input_path: str
        Path to the input graph data
    dgl_tool_path: str
        Path to the dgl tool added in the PYTHONPATH
    metadata_filename: str
        schema file name defined in the parmetis step
    """
    ip_list: str
    input_path: str
    dgl_tool_path: str
    metadata_filename: str