# Import libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as np
import sys
import xlrd


# Import of data
# WIND
'''
winddata = pd.read_excel('dataset.xlsx', 'Wind')
winddata['time'] = pd.date_range('2012-01-01 00:00', '2013-12-31 23:00',
                                 periods=None, freq='1H')

# DEMAND
demanddata = pd.read_excel('dataset.xlsx', 'Demand')
demanddata['time'] = pd.date_range('2012-01-01 00:00', '2013-12-31 23:00',
                                   periods=None, freq='1H')

# SUN
sundata = pd.read_excel('dataset.xlsx', 'Sun')
sundata['time'] = pd.date_range('2012-01-01 00:00', '2013-12-31 23:00',
                                periods=None, freq='1H')

# IMPEXP
impexpdata = pd.read_excel('dataset.xlsx', 'Imp_Exp')
impexpdata['time'] = pd.date_range('2012-01-01 00:00', '2013-12-31 23:00',
                                   periods=None, freq='1H')
'''
# GENERATORS
gendata = pd.read_excel('dataset.xlsx', 'Disaggregated install capacity')
'''
# LINE CAPACITY
linecap = pd.read_excel('dataset.xlsx', 'Line capacity')

# DAY AHEAD MARKET PRICES 2012-2013
daprices = pd.read_excel('dataset.xlsx', 'Day ahead market prices')
daprices['time'] = pd.date_range('2012-01-01 00:00', '2013-12-31 23:00',
                                 periods=None, freq='1H')

# Hydro power data

# Capacity and head
datahydro = pd.read_excel('dataset.xlsx', 'hydro')


# Sensitivity analysis on the Glacial inflow
# 0,005
newglacialinflow0_005 = pd.read_excel('dataset.xlsx', 'newglacialinflow0_005')
newglacialinflow0_005['time'] = pd.date_range('2012-01-01 00:00', '2013-12-31 23:00',
                                              periods=None, freq='1H')

# 0,01
newglacialinflow0_01 = pd.read_excel('dataset.xlsx', 'newglacialinflow0_01')
newglacialinflow0_01['time'] = pd.date_range('2012-01-01 00:00', '2013-12-31 23:00',
                                             periods=None, freq='1H')

# 0,015
newglacialinflow0_015 = pd.read_excel('dataset.xlsx', 'newglacialinflow0_015')
newglacialinflow0_015['time'] = pd.date_range('2012-01-01 00:00', '2013-12-31 23:00',
                                              periods=None, freq='1H')

# 0,02
newglacialinflow0_02 = pd.read_excel('dataset.xlsx', 'newglacialinflow0_02')
newglacialinflow0_02['time'] = pd.date_range('2012-01-01 00:00', '2013-12-31 23:00',
                                             periods=None, freq='1H')

# NEWFylling
newfylling = pd.read_excel('dataset.xlsx', 'newfilling')
newfylling['time'] = pd.date_range('2012-01-01 00:00', '2013-12-31 23:00',
                                   periods=None, freq='1H')

# NEWRiverinflow
newriverinflow = pd.read_excel('dataset.xlsx', 'newriverinflow')
newriverinflow['time'] = pd.date_range('2012-01-01 00:00', '2013-12-31 23:00',
                                       periods=None, freq='1H')

# NEW Max Volume
newmaxvolume = pd.read_excel('dataset.xlsx', 'newmaxvolume')
newmaxvolume['time'] = pd.date_range('2012-01-01 00:00', '2013-12-31 23:00',
                                     periods=None, freq='1H')

# Weekly damfylling
weekdamfill = pd.read_excel('dataset.xlsx', 'weeklydamfill')

'''
# Open the store
store = pd.HDFStore('store.h5')

# Save the data in the store
'''
store['winddata'] = winddata
store['demanddata'] = demanddata
store['sundata'] = sundata
store['impexpdata'] = impexpdata
'''
store['gendata'] = gendata
'''
store['linecap'] = linecap
store['daprices'] = daprices
store['datahydro'] = datahydro
store['newglacialinflow0_005'] = newglacialinflow0_005
store['newglacialinflow0_01'] = newglacialinflow0_01
store['newglacialinflow0_015'] = newglacialinflow0_015
store['newglacialinflow0_02'] = newglacialinflow0_02
store['newfylling'] = newfylling
store['newriverinflow'] = newriverinflow
store['newmaxvolume'] = newmaxvolume
store['weekdamfill'] = weekdamfill
'''
# Close the store
store.close()
