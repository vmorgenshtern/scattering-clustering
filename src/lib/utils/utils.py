"""
Useful methods for several purposes
Patched-Imagenet/lib/utils
"""

import os
import json
import datetime

import numpy as np

import torch


def timestamp():
    """
    Computes and returns current timestamp

    Args:
    -----
    timestamp: String
        Current timestamp in formate: hh-mm-ss
    """

    timestamp = str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')
    return timestamp


def for_all_methods(decorator):
    """
    Decorator that applies a decorator to all methods inside a class
    """
    def decorate(cls):
        for attr in cls.__dict__: # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


def create_directory(path):
    """
    Method that creates a directory if it does not already exist

    Args
    ----
    path: String
        path to the directory to be created
    dir_existed: boolean
        Flag that captures of the directory existed or was created
    """
    dir_existed = True
    if not os.path.exists(path):
        os.makedirs(path)
        dir_existed = False
    return dir_existed

#
