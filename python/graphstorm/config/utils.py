""" config related utils """
import json

def get_graph_name(part_cnfig):
    """ Get graph name from graph partition config file

    Parameter
    ---------
    part_config: str
        Path to graph partition config file

    Return
    ------
        graph_name
    """
    with open(part_cnfig, "r", encoding='utf-8') as f:
        config = json.load(f)
    return config["graph_name"]
