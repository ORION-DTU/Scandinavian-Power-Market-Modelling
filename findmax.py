# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import xlrd
import gurobipy as gb
from gurobipy import *
from collections import defaultdict
from numpy import *

# STORE ISSUES
# OPEN the store
store = pd.HDFStore('store.h5')
# LOAD the data
winddata = store['winddata']
sundata = store['sundata']
demanddata = store['demanddata']
impexpdata = store['impexpdata']
gendata = store['gendata']
linecap = store['linecap']
# CLOSE the store
store.close()

# a = ['DK', 'DK', 'DK', 'DK', 'DK', 'NO', 'FI', 'SE']


def findmax(x, t):
    return demanddata[x][t] - min(impexpdata[x][t], 0)
