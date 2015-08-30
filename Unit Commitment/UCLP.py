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
import os
import time
import xlwt
import time as pytime  # starttime = pytime.time()
import seaborn as sns
import Expando_class
from Expando_class import Expando

##################################################################################################################################################

# DATA FROM THE DIFFERENT STORES

# OPEN the general store
store = pd.HDFStore('store.h5')

# LOAD the "common" data
mystore = Expando()
mystore.winddata = store['winddata']
mystore.sundata = store['sundata']
mystore.demanddata = store['demanddata']
mystore.impexpdata = store['impexpdata']
gendata = store['gendata']
linecap = store['linecap']
daprices = store['daprices']

# CLOSE the store
store.close()


# OPEN the store to import outputs from the UCMILP
store = pd.HDFStore('UCMILPstore.h5')

# Import UCMILP data from the store

# TEST

df_windprod = store['windprodweekTEST']
df_solarprod = store['solarprodweekTEST']
df_genprod = store['genprodweekTEST']
df_loadshed = store['loadshedweekTEST']
df_flow = store['flowweekTEST']
df_p_exch = store['p_exchweekTEST']
df_hydro = store['hydroprodweekTEST']
df_u_onoff = store['onoffweekTEST']
df_v_startup = store['startupweekTEST']
df_v_shutdown = store['shutdownweekTEST']

# WEEK 3 Low wind High demand
# Cold start up costs and 50/50 fuel mix

# df_windprod = store['windprodweek3']
# df_solarprod = store['solarprodweek3']
# df_genprod = store['genprodweek3']
# df_loadshed = store['loadshedweek3']
# df_flow = store['flowweek3']
# df_p_exch = store['p_exchweek3']
# df_hydro = store['hydroprodweek3']
# df_u_onoff = store['onoffweek3']
# df_v_startup = store['startupweek3']
# df_v_shutdown = store['shutdownweek3']

# WEEK 4 High wind High demand
# Cold start up costs and 50/50 fuel mix

# df_windprod = store['windprodweek4']
# df_solarprod = store['solarprodweek4']
# df_genprod = store['genprodweek4']
# df_loadshed = store['loadshedweek4']
# df_flow = store['flowweek4']
# df_p_exch = store['p_exchweek4']
# df_hydro = store['hydroprodweek4']
# df_u_onoff = store['onoffweek4']
# df_v_startup = store['startupweek4']
# df_v_shutdown = store['shutdownweek4']


# WEEK 26 Low wind Low demand
# Cold start up costs and 50/50 fuel mix

# df_windprod = store['windprodweek26']
# df_solarprod = store['solarprodweek26']
# df_genprod = store['genprodweek26']
# df_loadshed = store['loadshedweek26']
# df_flow = store['flowweek26']
# df_p_exch = store['p_exchweek26']
# df_hydro = store['hydroprodweek26']
# df_u_onoff = store['onoffweek26']
# df_v_startup = store['startupweek26']
# df_v_shutdown = store['shutdownweek26']


# WEEK 27 High wind Low demand
# Cold start up costs and 50/50 fuel mix

# df_windprod = store['windprodweek27']
# df_solarprod = store['solarprodweek27']
# df_genprod = store['genprodweek27']
# df_loadshed = store['loadshedweek27']
# df_flow = store['flowweek27']
# df_p_exch = store['p_exchweek27']
# df_hydro = store['hydroprodweek27']
# df_u_onoff = store['onoffweek27']
# df_v_startup = store['startupweek27']
# df_v_shutdown = store['shutdownweek27']


# # Cold start up costs and 20/80 fuel mix

# df_windprod = store['windprodweek20/80cs']
# df_solarprod = store['solarprodweek20/80cs']
# df_genprod = store['genprodweek20/80cs']
# df_loadshed = store['loadshedweek20/80cs']
# df_flow = store['flowweek20/80cs']
# df_p_exch = store['p_exchweek20/80cs']
# df_hydro = store['hydroprodweek20/80cs']
# df_u_onoff = store['onoffweek20/80cs']
# df_v_startup = store['startupweek20/80cs']
# df_v_shutdown = store['shutdownweek20/80cs']


# # Cold start up costs and 80/20 fuel mix

# df_windprod = store['windprodweek80/20cs']
# df_solarprod = store['solarprodweek80/20cs']
# df_genprod = store['genprodweek80/20cs']
# df_loadshed = store['loadshedweek80/20cs']
# df_flow = store['flowweek80/20cs']
# df_p_exch = store['p_exchweek80/20cs']
# df_hydro = store['hydroprodweek80/20cs']
# df_u_onoff = store['onoffweek80/20cs']
# df_v_startup = store['startupweek80/20cs']
# df_v_shutdown = store['shutdownweek80/20cs']


# # Warm start up costs
# df_windprod = store['windprodweekws']
# df_solarprod = store['solarprodweekws']
# df_genprod = store['genprodweekws']
# df_loadshed = store['loadshedweekws']
# df_flow = store['flowweekws']
# df_p_exch = store['p_exchweekws']
# df_hydro = store['hydroprodweekws']
# df_u_onoff = store['onoffweekws']
# df_v_startup = store['startupweekws']
# df_v_shutdown = store['shutdownweekws']


# # Hot start up costs

# df_windprod = store['windprodweekhs']
# df_solarprod = store['solarprodweekhs']
# df_genprod = store['genprodweekhs']
# df_loadshed = store['loadshedweekhs']
# df_flow = store['flowweekhs']
# df_p_exch = store['p_exchweekhs']
# df_hydro = store['hydroprodweekhs']
# df_u_onoff = store['onoffweekhs']
# df_v_startup = store['startupweekhs']
# df_v_shutdown = store['shutdownweekhs']


# CLOSE the store
store.close()

# Display completely in the screen
pd.set_option('display.width', None)

##################################################################################################################################################

# MODEL DATA

# Time

# TEST
# time = range(8760, 17521)

# Week 3 2013
# time = range(9096, 9263)

# Week 4 2013
# time = range(9264, 9431)

# Week 26 2013
# time = range(12792, 12959)

# Week 27 2013
# time = range(12960, 13127)

# Regions
nodes = ['DK', 'SE', 'NO', 'FI']


# Index by units
unitind = gendata.index

hydrocountry = {'FI': gendata.index[214:223],
                'NO': gendata.index[224:243],
                'SE': gendata.index[244:254]}

##################################################################################################################################################

# PARAMETERS

VOLL = 2500  # ValueOfLostLoad

# Hydro coefficient

# hydrocoeff = 0.75  # Capacity factor of the hydro power plants WINTER
# hydrocoeff = 0.50  # Capacity factor of the hydro power plants SPRING/FALL
# hydrocoeff = 0.30  # Capacity factor of the hydro power plants SUMMER
hydrocoeff = 0.455

# Lines
lines = [('DK', 'NO'), ('DK', 'SE'), ('SE', 'FI'), ('SE', 'NO'), ('NO', 'FI')]

lineinfo = {}

lineinfo[('DK', 'NO')] = {'linecapacity': 1000, 'x': 1, 'otherinfo': []}
lineinfo[('DK', 'SE')] = {'linecapacity': 2050, 'x': 1, 'otherinfo': []}
lineinfo[('SE', 'FI')] = {'linecapacity': 2050, 'x': 1, 'otherinfo': []}
lineinfo[('SE', 'NO')] = {'linecapacity': 3450, 'x': 1, 'otherinfo': []}
lineinfo[('NO', 'FI')] = {'linecapacity': 100, 'x': 1, 'otherinfo': []}


# Range created for the minimum up time
minup = {}
for t in time:
    for g in unitind:
        minup = np.arange(t+1, t+gendata['min_up'][g]-1, 1)

# Range created for the minimum down time
mindown = {}
for t in time:
    for g in unitind:
        mindown = np.arange(t+1, t+gendata['min_down'][g]-1, 1)

##################################################################################################################################################

# UPDATE THE VARIABLES FROM THE MILP

# OnOff
u_UC = {}
for t in time:
    for g in unitind:
        u_UC.update({(t, g): df_u_onoff.ix[t, g]})

# Startup
v_up_UC = {}
for t in time:
    for g in unitind:
        v_up_UC.update({(t, g): df_v_startup.ix[t, g]})

# Shutdown
v_down_UC = {}
for t in time:
    for g in unitind:
        v_down_UC.update({(t, g): df_v_shutdown.ix[t, g]})

##################################################################################################################################################

# CREATE OPTIMIZATION MODEL

m = gb.Model("milp")


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


# Unit commitment


# Binary variable for ON/OFF status of generator "g" at time "t"
u = {}
for t in time:
    for g in unitind:
        u[t, g] = m.addVar(ub=u_UC[t, g],
                           lb=u_UC[t, g],
                           name='ON/OFF_status_at_time_{0}_of_unit_{1}_in_country_{2}'.format(t, g, gendata['region'][g]))

# Binary variable for START-UP of unit g at time t
v_up = {}
for t in time:
    for g in unitind:
        v_up[t, g] = m.addVar(ub=v_up_UC[t, g],
                              lb=v_up_UC[t, g],
                              name='Start_-_up_at_time_{0}_of_unit_{1}_in_country_{2}'.format(t, g, gendata['region'][g]))

# Binary variable for SHUT-DOWN of unit g at time t
v_down = {}
for t in time:
    for g in unitind:
        v_down[t, g] = m.addVar(ub=v_down_UC[t, g],
                                lb=v_down_UC[t, g],
                                name='Shut_-_down_at_time_{0}_of_unit_{1}_in_country_{2}'.format(t, g, gendata['region'][g]))

# Variable Unit always on
vonl = {}
for g in unitind:
    vonl[g] = 1

m.update()

##################################################################################################################################################

# OBJECTIVE FUNCTION

m.setObjective(

    quicksum(gendata['marg_cost'][g] * genprod[t, g]
             for t in time for g in unitind) +

    quicksum(v_up[t, g] * gendata['start_up_cost'][g]
             for t in time for g in unitind) +

    quicksum(v_down[t, g] * gendata['shut_down_cost'][g]
             for t in time for g in unitind) +

    quicksum(VOLL*loadshed[t, n]
             for t in time for n in nodes),

    gb.GRB.MINIMIZE)

##################################################################################################################################################

# SET THE CONSTRAINTS

# Demand satisfaction - Balance equation
powerbalance = {}
for t in time[1:]:
    for n in nodes:
        powerbalance[t, n] = m.addConstr(

            quicksum(genprod[t, g] for g in unitind if gendata.region[g] == n) +
            windprod[t, n] +
            solarprod[t, n] +
            loadshed[t, n] -
            mystore.impexpdata.ix[t, n],

            gb.GRB.EQUAL,

            mystore.demanddata.ix[t, n] +
            p_exch[t, n])


# Constrained flow
constrainedflow = {}
for t in time[1:]:
    for n in nodes:
        constrainedflow[t, n] = m.addConstr(

            p_exch[t, n] + quicksum(flow[t, l] for l in lines if l[1] == n), 

            gb.GRB.EQUAL,

            quicksum(flow[t, l] for l in lines if l[0] == n)
            )

# Hydropower limitation
hydroprod = {}
for g in unitind:
    if gendata['fuel_type'][g] == 'Water':
        hydroprod[g] = m.addConstr(

            quicksum(genprod[t, g] for t in time), 
            gb.GRB.LESS_EQUAL,

            hydrocoeff * gendata['capacity'][g] * len(time),
            )

# ON/OFF status of unit 'g' at time "t" including the previously introduced binary variable
on_off_low = {}
for t in time[1:]:
    for g in unitind:
        on_off_low[t, g] = m.addConstr(

            gendata['p_min'][g] * u[t, g],

            gb.GRB.LESS_EQUAL,

            genprod[t, g],
            )


on_off_high = {}
for t in time[1:]:
    for g in unitind:
        on_off_high[t, g] = m.addConstr(

            genprod[t, g],

            gb.GRB.LESS_EQUAL,

            gendata['p_max'][g] * u[t, g],
            )

# Restriction including Upward and Downward power
# ramping limit for unit "g" at time "t"
rampdown = {}
for t in time[1:]:
    for g in unitind:
        rampdown[t, g] = m.addConstr(

            genprod[t-1, g] -
            gendata.ramp_down[g] *
            gendata.p_max[g],

            gb.GRB.LESS_EQUAL,

            genprod[t, g],
            )

rampup = {}
for t in time[1:]:
    for g in unitind:
        rampup[t, g] = m.addConstr(

            genprod[t-1, g] +
            gendata.ramp_up[g] *
            gendata.p_max[g],

            gb.GRB.GREATER_EQUAL,

            genprod[t, g],
            )

# Start-up of a unit associated when the unit is turned ON
start_up = {}
for t in time[1:]:
    for g in unitind:
        start_up[t, g] = m.addConstr(

            v_up[t, g],

            gb.GRB.GREATER_EQUAL,

            u[t, g] -
            u[t-1, g] +
            (u[t, g] - vonl[g])*(t == time[1])
            )

# Shut-down of a unit associated when the unit is turned OFF
shut_down = {}
for t in time[1:]:
    for g in unitind:
        shut_down[t, g] = m.addConstr(

            v_down[t, g],

            gb.GRB.LESS_EQUAL,

            u[t, g] -
            u[t-1, g]
            )


# Minimum time offline
min_down_time = {}
for t in time[1:]:
    for g in unitind:
        for x in mindown:
            min_down_time[t, g] = addConstr(

                u[t, g] - u[t-1, g],

                gb.GRB.LESS_EQUAL,

                u[g, x],
                )

# Minimum time online
min_up_time = {}
for t in time[1:]:
    for g in unitind:
        for x in minup:
            min_up_time[t, g] = addConstr(

                u[t-1, g] - u[t, g],

                gb.GRB.LESS_EQUAL,

                1 - u[g, x],
                )
m.update()

##################################################################################################################################################

# Compute optimal solution
m.optimize()


# Fix hydro power profile
hydroconstr = {}
for t in time:
    for g in unitind:
        if gendata['fuel_type'][g] == 'Water':
                genprod[t, g].ub = genprod[t, g].x
                genprod[t, g].lb = genprod[t, g].x


m.update()

m.reset()


# Optimize second time to get prices right after fixing the hydro profile
m.optimize()

##################################################################################################################################################

# CREATE DE DATAFRAME

df_UCwindprod = pd.DataFrame({n: {t: windprod[t, n].x for t in time} for n in nodes})
df_UCsolarprod = pd.DataFrame({n: {t: solarprod[t, n].x for t in time} for n in nodes})
df_UCgenprod = pd.DataFrame({g: {t: genprod[t, g].x for t in time} for g in unitind})
df_UCloadshed = pd.DataFrame({n: {t: loadshed[t, n].x for t in time} for n in nodes})
df_UCflow = pd.DataFrame({l: {t: flow[t, l].x for t in time} for l in lines})
df_UCp_exch = pd.DataFrame({n: {t: p_exch[t, n].x for t in time} for n in nodes})
df_UChydro = pd.DataFrame({g: {t: genprod[t, g].x for t in time} for g in unitind if gendata['fuel_type'][g] == 'Water'})
# df_startup = pd.DataFrame({g: {t: start_up[t, g].x for t in time} for g in unitind})
df_UCprice = pd.DataFrame({n: {t: powerbalance[t, n].pi for t in time[1:]} for n in nodes})
df_hydroprod = pd.DataFrame({f: df_UChydro[idxs].sum(axis=1) for f, idxs in hydrocountry.iteritems()})

# OPEN the store to export outputs
store = pd.HDFStore('UCLPstore.h5')

# SAVE the data in the store


# TEST 

store['windprodweekTEST'] = df_UCwindprod
store['solarprodweekTEST'] = df_UCsolarprod
store['genprodweekTEST'] = df_UCgenprod
store['loadshedweekTEST'] = df_UCloadshed
store['flowweekTEST'] = df_UCflow
store['p_exchweekTEST'] = df_UCp_exch
store['hydroprodweekTEST'] = df_hydroprod
store['marg_costweekTEST'] = df_UCprice


# WEEK 3 Low wind High demand
# Cold start up costs and 50/50 fuel mix

# store['windprodweek3'] = df_UCwindprod
# store['solarprodweek3'] = df_UCsolarprod
# store['genprodweek3'] = df_UCgenprod
# store['loadshedweek3'] = df_UCloadshed
# store['flowweek3'] = df_UCflow
# store['p_exchweek3'] = df_UCp_exch
# store['hydroprodweek3'] = df_hydroprod
# store['marg_costweek3'] = df_UCprice


# WEEK 4 High wind High demand
# Cold start up costs and 50/50 fuel mix

# store['windprodweek4'] = df_UCwindprod
# store['solarprodweek4'] = df_UCsolarprod
# store['genprodweek4'] = df_UCgenprod
# store['loadshedweek4'] = df_UCloadshed
# store['flowweek4'] = df_UCflow
# store['p_exchweek4'] = df_UCp_exch
# store['hydroprodweek4'] = df_hydroprod
# store['marg_costweek4'] = df_UCprice


# WEEK 26 Low wind Low demand
# Cold start up costs and 50/50 fuel mix

# store['windprodweek26'] = df_UCwindprod
# store['solarprodweek26'] = df_UCsolarprod
# store['genprodweek26'] = df_UCgenprod
# store['loadshedweek26'] = df_UCloadshed
# store['flowweek26'] = df_UCflow
# store['p_exchweek26'] = df_UCp_exch
# store['hydroprodweek26'] = df_hydroprod
# store['marg_costweek26'] = df_UCprice

# WEEK 27 High wind Low demand
# Cold start up costs and 50/50 fuel mix

# store['windprodweek27'] = df_UCwindprod
# store['solarprodweek27'] = df_UCsolarprod
# store['genprodweek27'] = df_UCgenprod
# store['loadshedweek27'] = df_UCloadshed
# store['flowweek27'] = df_UCflow
# store['p_exchweek27'] = df_UCp_exch
# store['hydroprodweek27'] = df_hydroprod
# store['marg_costweek27'] = df_UCprice

# # Cold start up costs and 20/80 fuel mix
# store['windprod20/80cs'] = df_UCwindprod
# store['solarprod20/80cs'] = df_UCsolarprod
# store['genprod20/80cs'] = df_UCgenprod
# store['loadshed20/80cs'] = df_UCloadshed
# store['flow20/80cs'] = df_UCflow
# store['p_exch20/80cs'] = df_UCp_exch
# store['hydroprod20/80cs'] = df_hydroprod
# store['marg_cost20/80cs'] = df_UCprice

# # Cold start up costs and 80/20 fuel mix
# store['windprod80/20cs'] = df_UCwindprod
# store['solarprod80/20cs'] = df_UCsolarprod
# store['genprod80/20cs'] = df_UCgenprod
# store['loadshed80/20cs'] = df_UCloadshed
# store['flow80/20cs'] = df_UCflow
# store['p_exch80/20cs'] = df_UCp_exch
# store['hydroprod80/20cs'] = df_hydroprod
# store['marg_cost80/20cs'] = df_UCprice

# # Warm start up costs
# store['windprodws'] = df_UCwindprod
# store['solarprodws'] = df_UCsolarprod
# store['genprodws'] = df_UCgenprod
# store['loadshedws'] = df_UCloadshed
# store['flowws'] = df_UCflow
# store['p_exchws'] = df_UCp_exch
# store['hydroprodws'] = df_hydroprod
# store['marg_costws'] = df_UCprice

# # Hot start up costs
# store['windprodhs'] = df_UCwindprod
# store['solarprodhs'] = df_UCsolarprod
# store['genprodhs'] = df_UCgenprod
# store['loadshedhs'] = df_UCloadshed
# store['flowhs'] = df_UCflow
# store['p_exchhs'] = df_UCp_exch
# store['hydroprodhs'] = df_hydroprod
# store['marg_cosths'] = df_UCprice


# CLOSE the store
store.close()

