#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Import libraries
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import numpy as np
import sys
import xlrd
import gurobipy as gb
from gurobipy import *
from collections import defaultdict
from numpy import *
from findmax import findmax
import os
import time
import pylab
import xlwt
import time as pytime
import seaborn as sns
import Expando_class
from Expando_class import Expando
sns.set_style('ticks')
from sklearn.metrics import mean_squared_error
from math import sqrt
import miao
import scipy as scipy


# STORE ISSUES
# OPEN the store
store = pd.HDFStore('store.h5')
# LOAD the data
mystore = Expando()
mystore.winddata = store['winddata']
mystore.sundata = store['sundata']
mystore.demanddata = store['demanddata']
mystore.impexpdata = store['impexpdata']
gendata = store['gendata']
linecap = store['linecap']
daprices = store['daprices']
datahydro = store['datahydro']
# CLOSE the store
store.close()


# Display completely in the screen
pd.set_option('display.width', None)


# MODEL DATA
# Time
time = range(12961, 13127)

# Regions
nodes = ['DK', 'SE', 'NO', 'FI']

# print 'Step1:{}'.format(pytime.time()-starttime)

# Index by units
unitind = gendata.index


# Parameters
VOLL = 2500  # ValueOfLostLoad
hydrocoeff = 0.75  # Capacity factor of the hydro power plants


# LINES
lines = [('DK', 'NO'), ('DK', 'SE'), ('SE', 'FI'), ('SE', 'NO'), ('NO', 'FI')]

lineinfo = {}

lineinfo[('DK', 'NO')] = {'linecapacity': 1000, 'x': 1, 'otherinfo': []}
lineinfo[('DK', 'SE')] = {'linecapacity': 2050, 'x': 1, 'otherinfo': []}
lineinfo[('SE', 'FI')] = {'linecapacity': 2050, 'x': 1, 'otherinfo': []}
lineinfo[('SE', 'NO')] = {'linecapacity': 3450, 'x': 1, 'otherinfo': []}
lineinfo[('NO', 'FI')] = {'linecapacity': 100, 'x': 1, 'otherinfo': []}


# CREATE OPTIMIZATION MODEL
m = gb.Model('orion')


# CREATE THE VARIABLES
# Power production from each unit
genprod = {}
for t in time:
    for g in unitind:
        genprod[t, g] = m.addVar(ub=gendata['capacity'][g],
                                 name='Power_production_at_time_{0}_of_unit_{1}_in_country_{2}'.format(t, g, gendata['region'][g]))

# Load Shedding in each country
loadshed = {}
for t in time:
    for n in nodes:
        loadshed[t, n] = m.addVar(ub=findmax(n, t),
                                  name='Loadshed_at_time_{0}_in_country_{1}'.format(t, n))

# Flow over the lines (power actually flowing over a specific line, gives an
# information about where the power is going to)
flow = {}
for t in time:
    for l in lines:
        flow[t, l] = m.addVar(lb=-lineinfo[l]['linecapacity'],
                              ub=lineinfo[l]['linecapacity'],
                              name='flow_at_time_{0}_in_line_{1}'.format(t, l))

# Sun Production
solarprod = {}
for t in time:
    for n in nodes:
        solarprod[t, n] = m.addVar(ub=mystore.sundata.ix[t, n],
                                   name='Solar_production_at_time_{0}_in_country_{1}'.format(t, n))

# Wind production
windprod = {}
for t in time:
    for n in nodes:
        windprod[t, n] = m.addVar(ub=mystore.winddata.ix[t, n],
                                  name='Wind_production_at_time_{0}_in_country_{1}'.format(t, n))

# Power exchanged (This is the power "left" within a zone j which is send/taken to/from the network)
p_exch = {}
for t in time:
    for n in nodes:
        p_exch[t, n] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY,
                                name='Power_surplus_at_time_{0}_in_country_{1}'.format(t, n))

# Update the variables
m.update()


# OBJECTIVE FUNCTION

m.setObjective(
    quicksum(gendata['marg_cost'][g]*genprod[t, g]
             for t in time for g in unitind) +
    quicksum(VOLL*loadshed[t, n]
             for t in time for n in nodes),
    gb.GRB.MINIMIZE)


# SET THE CONSTRAINTS

# Electricity balance
powerbalance = {}
for t in time:
    for n in nodes:
        powerbalance[t, n] = m.addConstr(
            quicksum(genprod[t, g] for g in unitind if gendata.region[g] == n) +
            windprod[t, n] +
            solarprod[t, n] +
            loadshed[t, n] -
            mystore.impexpdata.ix[t, n], gb.GRB.EQUAL,
            mystore.demanddata.ix[t, n] +
            p_exch[t, n])

'''
# No power flow case
noflowconstrain = {}
for t in time:
    for n in nodes:
        noflowconstrain[t, n] = m.addConstr(
            p_exch[t, n], gb.GRB.EQUAL, 0,
            "no flow")
'''

'''
# Simple flow
simpleflowconstrain = {}
for t in time:
    simpleflowconstrain[t] = m.addConstr(
        quicksum(p_exch[t, n] for n in nodes), gb.GRB.EQUAL, 0,
        "simple flow")

'''

# Constrained flow
constrainedflow = {}
for t in time:
    for n in nodes:
        constrainedflow[t, n] = m.addConstr(
            p_exch[t, n] + quicksum(flow[t, l] for l in lines if l[1] == n), gb.GRB.EQUAL,
            quicksum(flow[t, l] for l in lines if l[0] == n)
            )

# Hydropower limitation
hydroprod = {}
for g in unitind:
    if gendata['fuel_type'][g] == 'Water':
        hydroprod[g] = m.addConstr(
            quicksum(genprod[t, g] for t in time), gb.GRB.LESS_EQUAL,
            hydrocoeff * gendata['capacity'][g] * len(time),
            )


m.update()

# Compute optimal solution
m.optimize()


# Fix hydro profile
hydroconstr = {}
for t in time:
    for g in unitind:
        if gendata['fuel_type'][g] == 'Water':
                genprod[t, g].ub = genprod[t, g].x
                genprod[t, g].lb = genprod[t, g].x

m.update()

m.reset()

# Optimize second time to get prices right
m.optimize()


# Create the dataframes
df_oldprice = pd.DataFrame({n: {t: powerbalance[t, n].pi for t in time} for n in nodes})
df_windprod = pd.DataFrame({n: {t: windprod[t, n].x for t in time} for n in nodes})
df_solarprod = pd.DataFrame({n: {t: solarprod[t, n].x for t in time} for n in nodes})
df_genprod = pd.DataFrame({g: {t: genprod[t, g].x for t in time} for g in unitind})
df_loadshed = pd.DataFrame({n: {t: loadshed[t, n].x for t in time} for n in nodes})
df_flow = pd.DataFrame({l: {t: flow[t, l].x for t in time} for l in lines})
df_p_exch = pd.DataFrame({n: {t: p_exch[t, n].x for t in time} for n in nodes})
df_hydro = pd.DataFrame({g: {t: genprod[t, g].x for t in time} for g in unitind if gendata['fuel_type'][g] == 'Water'})

# OPEN the store to export outputs

store = pd.HDFStore('outputstore.h5')

# Save the data in the store

# store['lowmc_price'] = df_oldprice
# store['lowmc_windprod'] = df_windprod
# store['lowmc_solarprod'] = df_solarprod
# store['lowmc_genprod'] = df_genprod
# store['lowmc_loadshed'] = df_loadshed
# store['lowmc_flow'] = df_flow
# store['lowmc_p_exch'] = df_p_exch
# store['lowmc_hydroprod'] = df_hydro

store['priceold13'] = df_oldprice
store['windprodold13'] = df_windprod
store['solarprodold13'] = df_solarprod
store['genprodold13'] = df_genprod
store['loadshedold13'] = df_loadshed
store['flowold13'] = df_flow
store['p_exchold13'] = df_p_exch
store['hydroprodold13'] = df_hydro


# store['price'] = df_oldprice
# store['windprod'] = df_windprod
# store['solarprod'] = df_solarprod
# store['genprod'] = df_genprod
# store['loadshed'] = df_loadshed
# store['flow'] = df_flow
# store['p_exch'] = df_p_exch
# store['hydroprod'] = df_hydro

# store['price2012'] = df_oldprice
# store['windprod2012'] = df_windprod
# store['solarprod2012'] = df_solarprod
# store['genprod2012'] = df_genprod
# store['loadshed2012'] = df_loadshed
# store['flow2012'] = df_flow
# store['p_exch2012'] = df_p_exch
# store['hydroprod2012'] = df_hydro

# store['price2013'] = df_oldprice
# store['windprod2013'] = df_windprod
# store['solarprod2013'] = df_solarprod
# store['genprod2013'] = df_genprod
# store['loadshed2013'] = df_loadshed
# store['flow2013'] = df_flow
# store['p_exch2013'] = df_p_exch
# store['hydroprod2013'] = df_hydro

# Sensitivity analysis

# store['0.1_price2012'] = df_oldprice
# store['0.1_windprod2012'] = df_windprod
# store['0.1_solarprod2012'] = df_solarprod
# store['0.1_genprod2012'] = df_genprod
# store['0.1_loadshed2012'] = df_loadshed
# store['0.1_flow2012'] = df_flow
# store['0.1_p_exch2012'] = df_p_exch
# store['0.1_hydroprod2012'] = df_hydro

# store['0.35_price2012'] = df_oldprice
# store['0.35_windprod2012'] = df_windprod
# store['0.35_solarprod2012'] = df_solarprod
# store['0.35_genprod2012'] = df_genprod
# store['0.35_loadshed2012'] = df_loadshed
# store['0.35_flow2012'] = df_flow
# store['0.35_p_exch2012'] = df_p_exch
# store['0.35_hydroprod2012'] = df_hydro

# store['0_55_price2012'] = df_oldprice
# store['0_55_windprod2012'] = df_windprod
# store['0_55_solarprod2012'] = df_solarprod
# store['0_55_genprod2012'] = df_genprod
# store['0_55_loadshed2012'] = df_loadshed
# store['0_55_flow2012'] = df_flow
# store['0_55_p_exch2012'] = df_p_exch
# store['0_55_hydroprod2012'] = df_hydro

# store['0.7_price2012'] = df_oldprice
# store['0.7_windprod2012'] = df_windprod
# store['0.7_solarprod2012'] = df_solarprod
# store['0.7_genprod2012'] = df_genprod
# store['0.7_loadshed2012'] = df_loadshed
# store['0.7_flow2012'] = df_flow
# store['0.7_p_exch2012'] = df_p_exch
# store['0.7_hydroprod2012'] = df_hydro

# store['0.9_price2012'] = df_oldprice
# store['0.9_windprod2012'] = df_windprod
# store['0.9_solarprod2012'] = df_solarprod
# store['0.9_genprod2012'] = df_genprod
# store['0.9_loadshed2012'] = df_loadshed
# store['0.9_flow2012'] = df_flow
# store['0.9_p_exch2012'] = df_p_exch
# store['0.9_hydroprod2012'] = df_hydro

# store['1_price2012'] = df_oldprice
# store['1_windprod2012'] = df_windprod
# store['1_solarprod2012'] = df_solarprod
# store['1_genprod2012'] = df_genprod
# store['1_loadshed2012'] = df_loadshed
# store['1_flow2012'] = df_flow
# store['1_p_exch2012'] = df_p_exch
# store['1_hydroprod2012'] = df_hydro

# CLOSE the store
store.close()
