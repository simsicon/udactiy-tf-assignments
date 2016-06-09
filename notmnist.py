from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

pickle_file = "notMNIST.pickle"

try:
  with open(pickle_file, "rb") as f:
    letter_set = pickle.load(f)
    np.random.shuffle(letter_set)

  print(letter_set[0])

except Exception as e:
  raise
