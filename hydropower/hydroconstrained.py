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
sns.set_style('ticks')
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
time = range(1, 8761)


# Regions
nodes = ['DK', 'SE', 'NO', 'FI']


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
DD = 0.1
UD = 0.1

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
gen_prod_dict = gendata.capacity.to_dict()
genprod = {}
for t in time:
    for g in unitind:
        genprod[t, g] = m.addVar(ub=gen_prod_dict[g],
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

# Up deviation
up_dev = {}
for t in time:
    for g in hydroind:
        up_dev[t, g] = m.addVar(ub=gb.GRB.INFINITY, lb=0,
                                name='Up_deviation_at_time_{0}_of_unit_{1}_from_the_hystorical_profile'.format(t, g))

# Down deviation
down_dev = {}
for t in time:
    for g in hydroind:
        down_dev[t, g] = m.addVar(ub=gb.GRB.INFINITY, lb=0,
                                  name='Down_deviation_at_time_{0}_of_unit_{1}_from_the_hystorical_profile'.format(t, g))

# Spill of water in case of copious inflow
hydrospill = {}
for t in time:
    for g in hydroind:
        hydrospill[t, g] = m.addVar(name="Hydro spill at gen. {0} at time {1}".format(g, t))

m.update()

# the beginning value of the reservoirs has be the same of the hystorical data
for g in hydroind:
    res_lev[time[0], g].lb = mystore.newfylling[g].ix[0]

m.update()

# OBJECTIVE FUNCTION

m.setObjective(
    quicksum(gendata.marg_cost[g]*(1+0.1*np.random.random())*genprod[t, g]
             for t in time for g in unitind) +
    quicksum(VOLL*loadshed[t, n]
             for t in time for n in nodes),
    gb.GRB.MINIMIZE)


# SET THE CONSTRAINTS

# Electricity balance
powerbalance = {}
for t in time[1:]:
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
                genprod[t, g] -
                hydrospill[t, g]
                )

# level of the resevoir at the beginning(time 0)has to be equal at the end (time[-1])
reslevmaintain = {}
for g in hydroind:
    reslevmaintain[g] = m.addConstr(
        res_lev[time[-1], g],
        gb.GRB.EQUAL,
        res_lev[time[0], g])


m.update()

# Compute optimal solution
m.optimize()


# Fix hydro profile
hydroconstr = {}
for t in time[1:]:
    for g in hydroind:
                genprod[t, g].ub = genprod[t, g].x
                genprod[t, g].lb = genprod[t, g].x

m.update()

m.reset()

# Optimize second time to get prices right
m.optimize()


# Create the dataframes

df_windprod = pd.DataFrame({n: {t: windprod[t, n].x for t in time} for n in nodes})
df_solarprod = pd.DataFrame({n: {t: solarprod[t, n].x for t in time} for n in nodes})
df_genprod = pd.DataFrame({g: {t: genprod[t, g].x for t in time} for g in unitind})
df_loadshed = pd.DataFrame({n: {t: loadshed[t, n].x for t in time} for n in nodes})
df_flow = pd.DataFrame({l: {t: flow[t, l].x for t in time} for l in lines})
df_p_exch = pd.DataFrame({n: {t: p_exch[t, n].x for t in time} for n in nodes})
df_hydro = pd.DataFrame({g: {t: genprod[t, g].x for t in time} for g in hydroind})
df_down_dev = pd.DataFrame({g: {t: down_dev[t, g].x for t in time} for g in hydroind})
df_up_dev = pd.DataFrame({g: {t: up_dev[t, g].x for t in time} for g in hydroind})
df_oldprice = pd.DataFrame({n: {t: powerbalance[t, n].pi for t in time[1:]} for n in nodes})
df_res_lev = pd.DataFrame({g: {t: res_lev[t, g].x for t in time} for g in hydroind})
df_hydroprod = pd.DataFrame({f: df_hydro[idxs].sum(axis=1) for f, idxs in hydrocountry.iteritems()})
df_country_hydro = pd.DataFrame({c: df_res_lev[idxs].sum(axis=1) for c, idxs in hydrocountry.iteritems()})
df_hydrospill = pd.DataFrame({g: {t: hydrospill[t, g].x for t in time} for g in hydroind})
df_country_hydrospill = pd.DataFrame({l: df_hydrospill[idxs].sum(axis=1) for l, idxs in hydrocountry.iteritems()})

# OPEN the store to export outputs

store = pd.HDFStore('hydroconsstore.h5')

# Save the data in the store

# 0,005
# store['windprodconstr12'] = df_windprod
# store['solarprodconstr12'] = df_solarprod
# store['genprodconstr12'] = df_genprod
# store['loadshedconstr12'] = df_loadshed
# store['flowconstr12'] = df_flow
# store['p_exchconstr12'] = df_p_exch
# store['hydroprodconstr12'] = df_hydro
# store['downdeviationconstr12'] = df_down_dev
# store['updeviationconstr12'] = df_up_dev
# store['priceconstr12'] = df_oldprice
# store['reservoirlevelconstr12'] = df_res_lev
# store['reslevpercountryconstr12'] = df_hydroprod
# store['hydropercountryconstr12'] = df_country_hydro
# store['hydrospillingconstr12'] = df_hydrospill
# store['hydrospillpercountryconstr12'] = df_country_hydrospill

# 0,015
# store['windprodconstr2'] = df_windprod
# store['solarprodconstr2'] = df_solarprod
# store['genprodconstr2'] = df_genprod
# store['loadshedconstr2'] = df_loadshed
# store['flowconstr2'] = df_flow
# store['p_exchconstr2'] = df_p_exch
# store['hydroprodconstr2'] = df_hydro
# store['downdeviationconstr2'] = df_down_dev
# store['updeviationconstr2'] = df_up_dev
# store['priceconstr2'] = df_oldprice
# store['reservoirlevelconstr2'] = df_res_lev
# store['reslevpercountryconstr2'] = df_hydroprod
# store['hydropercountryconstr2'] = df_country_hydro
# store['hydrospillingconstr2'] = df_hydrospill
# store['hydrospillpercountryconstr2'] = df_country_hydrospill

# 0,02
# store['windprodconstr3'] = df_windprod
# store['solarprodconstr3'] = df_solarprod
# store['genprodconstr3'] = df_genprod
# store['loadshedconstr3'] = df_loadshed
# store['flowconstr3'] = df_flow
# store['p_exchconstr3'] = df_p_exch
# store['hydroprodconstr3'] = df_hydro
# store['downdeviationconstr3'] = df_down_dev
# store['updeviationconstr3'] = df_up_dev
# store['priceconstr3'] = df_oldprice
# store['reservoirlevelconstr3'] = df_res_lev
# store['reslevpercountryconstr3'] = df_hydroprod
# store['hydropercountryconstr3'] = df_country_hydro
# store['hydrospillingconstr3'] = df_hydrospill
# store['hydrospillpercountryconstr3'] = df_country_hydrospill


store['windprodconstr12'] = df_windprod
store['solarprodconstr12'] = df_solarprod
store['genprodconstr12'] = df_genprod
store['loadshedconstr12'] = df_loadshed
store['flowconstr12'] = df_flow
store['p_exchconstr12'] = df_p_exch
store['hydroprodconstr12'] = df_hydro
store['downdeviationconstr12'] = df_down_dev
store['updeviationconstr12'] = df_up_dev
store['priceconstr12'] = df_oldprice
store['reservoirlevelconstr12'] = df_res_lev
store['reslevpercountryconstr12'] = df_hydroprod
store['hydropercountryconstr12'] = df_country_hydro
store['hydrospillingconstr12'] = df_hydrospill
store['hydrospillpercountryconstr12'] = df_country_hydrospill

# store['windprodconstr13'] = df_windprod
# store['solarprodconstr13'] = df_solarprod
# store['genprodconstr13'] = df_genprod
# store['loadshedconstr13'] = df_loadshed
# store['flowconstr13'] = df_flow
# store['p_exchconstr13'] = df_p_exch
# store['hydroprodconstr13'] = df_hydro
# store['downdeviationconstr13'] = df_down_dev
# store['updeviationconstr13'] = df_up_dev
# store['priceconstr13'] = df_oldprice
# store['reservoirlevelconstr13'] = df_res_lev
# store['reslevpercountryconstr13'] = df_hydroprod
# store['hydropercountryconstr13'] = df_country_hydro
# store['hydrospillingconstr13'] = df_hydrospill
# store['hydrospillpercountryconstr13'] = df_country_hydrospill

# CLOSE the store
store.close()
