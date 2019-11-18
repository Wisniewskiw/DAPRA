class Wrapper(object):
    """
    A more convenient dictionary object
    Has auto completion in notebooks and ipython
    """

    def __init__(self, dico):
        self.__dict__.update(dico)

    def getvars(self, dico):
        self.__dict__.update(dico)

    def keys(self):
        return self.__dict__.keys()
