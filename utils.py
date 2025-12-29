import lzma  # compress files, higher comression ratio than gzip
import os
from pathlib import Path 
import dill as pickle
import pandas as pd 
import numpy as np
import random
from datetime import timedelta  # for run sim purposes, check again
from copy import deepcopy
from collections import defaultdict
from timeme import timeme
os.chdir(r"C:/Users/Jiawe.JIAWEI/OneDrive/Desktop/Coding/Python/cryptostatarb")



def load_pickle(path): #fast cache reload for large datasets and intermediate results.
    with lzma.open(path, 'rb') as fp:
        file = pickle.load(fp)
    return file 


def save_pickle(path, obj):
    with lzma.open(path, 'wb') as fp:
        pickle.dump(obj, fp)