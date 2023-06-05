import numpy as np

def my_func2():
    try:
        my_func1()
    except:
        return np.nan
    # raise ValueError

def my_func1():
    raise ValueError


if __name__ == "__main__":
    my_func2()