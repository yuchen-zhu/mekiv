__all__ = ["DotDict"]


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def from_dict(dct: dict) -> "DotDict":
        dotdict = DotDict(dct)
        for key in dotdict.keys():
            if type(dotdict[key]) is dict:
                dotdict[key] = DotDict.from_dict(dotdict[key])
        return dotdict
