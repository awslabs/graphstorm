REGISTRY = {}


def register_array_parser(name):
    """Decorator function."""
    def _deco(cls):
        REGISTRY[name] = cls
        return cls

    return _deco


def get_array_parser(**fmt_meta):
    """User interfacing function to retrieve appropriate class
    for a read/write task.

    Argument
    --------
    fmt_meta: dict
        Dictionary of parameters.
    """
    cls = REGISTRY[fmt_meta.pop("name")]
    return cls(**fmt_meta)
