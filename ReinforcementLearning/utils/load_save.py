import yaml
import glob
import os
import pandas as pd
import random


def get_file_path(file_path):
    project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(project_path, file_path)
    return file_path


def load_yaml(file_path):
    file_path = get_file_path(file_path)
    files = glob.glob(file_path, recursive=True)
    assert(len(files) == 1)
    cfg = yaml.safe_load(open(files[0], 'r'))
    return cfg


def load_pickle(file_path):
    file_path = get_file_path(file_path)
    files = glob.glob(file_path, recursive=True)
    assert(len(files) == 1)
    data = pd.read_pickle(open(files[0], 'rb'))
    return data

class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a tuple."""
        self.memory.append([*args])

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.memory
        else:
            random_batch = random.sample(self.memory, batch_size)
            return random_batch

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)

