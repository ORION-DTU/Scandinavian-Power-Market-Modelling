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
from findmax import findmax
from hydromax import hydromax
import os
import time
import time as pytime
import seaborn as sns
import Expando_class
from Expando_class import Expando
# starttime = pytime.time()


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
mystore.newglacialinflow0_005 = store['newglacialinflow0_005']
mystore.newglacialinflow0_01 = store['newglacialinflow0_01']
mystore.newglacialinflow0_015 = store['newglacialinflow0_015']
mystore.newglacialinflow0_02 = store['newglacialinflow0_02']
mystore.newfylling = store['newfylling']
mystore.newriverinflow = store['newriverinflow']
mystore.newmaxvolume = store['newmaxvolume']

# CLOSE the store
store.close()


# Display completely in the screen
pd.set_option('display.width', None)


# MODEL DATA

# Time
# offset = 24*30*12
# time = range(0+offset, 8760+offset)
time = range(8760, 17544)


# Regions
nodes = ['DK', 'SE', 'NO', 'FI']

# print 'Step1:{}'.format(pytime.time()-starttime)

# Index by units
unitind = gendata.index
hydroind = gendata.index[214:254]


# Group the hydro
hydrocountry = {'FI': gendata.index[214:223],
                'NO': gendata.index[224:243],
                'SE': gendata.index[244:254]}


# Parameters
VOLL = 2500  # ValueOfLostLoad
hydrocoeff = 0.455  # Capacity factor of the hydro power plants


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

# Hydro level reservoir
res_lev = {}
for t in time:
    for g in hydroind:
        res_lev[t, g] = m.addVar(ub=gb.GRB.INFINITY,  # 1.1*max(mystore.newfylling[g]),
                                 name='Reservoir_level_at_time_{0}_of_unit_{1}'.format(t, g))

# Hydro spill = possibility to spill water
hydrospill = {}
for t in time:
    for g in hydroind:
        hydrospill[t, g] = m.addVar(name="Hydro spill at gen. {0} at time {1}".format(g, t))

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

# Demand satisfaction
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
hydrolimit = {}
for t in time[1:]:
    for g in hydroind:
            hydrolimit[t, g] = m.addConstr(
                res_lev[t, g], gb.GRB.EQUAL,
                res_lev[t-1, g] +
                mystore.newriverinflow.ix[t, g] -
                genprod[t, g]
                )


m.update()

# Compute optimal solution
m.optimize()

# Fix hydro profile
hydroconstr = {}
for t in time:
    for g in hydroind:
                genprod[t, g].ub = genprod[t, g].x
                genprod[t, g].lb = genprod[t, g].x


m.update()

m.reset()

# Optimize second time to get prices right
m.optimize()


# Create the dataframe

df_windprod = pd.DataFrame({n: {t: windprod[t, n].x for t in time} for n in nodes})
df_solarprod = pd.DataFrame({n: {t: solarprod[t, n].x for t in time} for n in nodes})
df_genprod = pd.DataFrame({g: {t: genprod[t, g].x for t in time} for g in unitind})
df_loadshed = pd.DataFrame({n: {t: loadshed[t, n].x for t in time} for n in nodes})
df_flow = pd.DataFrame({l: {t: flow[t, l].x for t in time} for l in lines})
df_p_exch = pd.DataFrame({n: {t: p_exch[t, n].x for t in time} for n in nodes})
df_hydro = pd.DataFrame({g: {t: genprod[t, g].x for t in time} for g in hydroind})
df_oldprice = pd.DataFrame({n: {t: powerbalance[t, n].pi for t in time} for n in nodes})
df_res_lev = pd.DataFrame({g: {t: res_lev[t, g].x for t in time} for g in hydroind})
df_hydroprod = pd.DataFrame({f: df_hydro[idxs].sum(axis=1) for f, idxs in hydrocountry.iteritems()})
df_country_hydro = pd.DataFrame({c: df_res_lev[idxs].sum(axis=1) for c, idxs in hydrocountry.iteritems()})
# df_hydrospill = pd.DataFrame({g: {t: hydrospill[t, g].x for t in time} for g in hydroind})
# df_country_hydrospill = pd.DataFrame({l: df_hydrospill[idxs].sum(axis=1) for l, idxs in hydrocountry.iteritems()})
df_inflow = pd.DataFrame({l: mystore.newriverinflow[idxs].sum(axis=1) for l, idxs in hydrocountry.iteritems()})
df_realfyll = pd.DataFrame({l: mystore.newfylling[idxs].sum(axis=1) for l, idxs in hydrocountry.iteritems()})

# OPEN the store to export outputs
store = pd.HDFStore('hydrobasestore.h5')

# Save the data in the store

# year 2012
# store just for fun
# store['windprod'] = df_windprod
# store['solarprod'] = df_solarprod
# store['genprod'] = df_genprod
# store['loadshed'] = df_loadshed
# store['flow'] = df_flow
# store['p_exch'] = df_p_exch
# store['hydroprod'] = df_hydro
# store['price'] = df_oldprice
# store['reservoirlevel'] = df_res_lev
# store['reslevpercountry'] = df_hydroprod
# store['hydropercountry'] = df_country_hydro


# year 2013
store['windprod13'] = df_windprod
store['solarprod13'] = df_solarprod
store['genprod13'] = df_genprod
store['loadshed13'] = df_loadshed
store['flow13'] = df_flow
store['p_exch13'] = df_p_exch
store['hydroprod13'] = df_hydro
store['price13'] = df_oldprice
store['reservoirlevel13'] = df_res_lev
store['reslevpercountry13'] = df_hydroprod
store['hydropercountry13'] = df_country_hydro

# CLOSE the store
store.close()
