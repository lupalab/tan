import tensorflow as tf
import copy
import os
import errno


def add_noise(inputs, scale, noise=tf.random_uniform):
    """ Helper function to add noise in tensorflow. """
    return inputs + scale*noise(tf.shape(inputs))


def get_default(dictionary, index, default=None):
    """Simple helper function to get values from dictionary, or returning a
    default value if the index is not in the dictionary.
    """
    try:
        return dictionary[index]
    except KeyError:
        return default


def addkeyval(dictionary, key, value):
    """ Return a new dictionary with key, value added."""
    new_dict = copy.copy(dictionary)
    new_dict[key] = value
    return new_dict


def make_arguments(arg_choices, args={}):
    # if there are no choices left, then return only the current arguments.
    if len(arg_choices) == 0:
        return [args]
    # Pop a choice, populate the resulting argument list with each argument
    # value.
    arg_choices = copy.copy(arg_choices)
    key = arg_choices.keys()[0]
    vals = arg_choices.pop(key)
    args_list = []
    for v in vals:
        args_list += make_arguments(arg_choices, addkeyval(args, key, v))
    return args_list


def make_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
