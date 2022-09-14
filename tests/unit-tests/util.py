class Dummy: # dummy object used to create config objects
    # constructor
    def __init__(self, arg_dict):
        self.__dict__.update(arg_dict)
