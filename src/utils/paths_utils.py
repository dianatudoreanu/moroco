from os.path import dirname
import os


PROJECT_PATH = dirname(dirname(dirname(os.path.abspath(__file__))))


def get_absolute_path(path):
    if path[0] == '@':
        return os.path.join(PROJECT_PATH, path[1:])
    else:
        return path
