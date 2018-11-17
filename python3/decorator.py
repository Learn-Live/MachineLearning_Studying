# https://stackoverflow.com/questions/35758323/hook-python-module-function

import functools

import whatever


def prefix_function(function, prefunction):
    @functools.wraps(function)
    def run(*args, **kwargs):
        prefunction(*args, **kwargs)
        return function(*args, **kwargs)

    return run


def this_is_a_function(parameter):
    pass  # Your own code here that will be run before


whatever.this_is_a_function = prefix_function(
    whatever.this_is_a_function, this_is_a_function)
