# Import libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
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

# INPUT DATA STORE where all the input parameters are saved (previously imported from Excel)

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

# CLOSE the store
store.close()


# Display completely in the screen
pd.set_option('display.width', None)

##################################################################################################################################################

# MODEL DATA

# Time

# TEST
time = range(8760, 17521)

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

# Index the data by units
unitind = gendata.index

##################################################################################################################################################

# PARAMETERS

VOLL = 2500  # ValueOfLostLoad

# Hydro coefficient

# hydrocoeff = 0.75  # Capacity factor of the hydro power plants WINTER
# hydrocoeff = 0.50  # Capacity factor of the hydro power plants SPRING/FALL
# hydrocoeff = 0.30  # Capacity factor of the hydro power plants SUMMER
hydrocoeff = 0.455

high = 0.03  # Increase of the demand to test the UC model


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

# CREATE OPTIMIZATION MODE

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
        u[t, g] = m.addVar(vtype=GRB.BINARY,
                           name='ON/OFF_status_at_time_{0}_of_unit_{1}_in_country_{2}'.format(t, g, gendata['region'][g]))

# Binary variable for START UP of unit g at time t
v_up = {}
for t in time:
    for g in unitind:
        v_up[t, g] = m.addVar(vtype=GRB.BINARY,
                              name='Start_-_up_at_time_{0}_of_unit_{1}_in_country_{2}'.format(t, g, gendata['region'][g]))

# Binary variable for SHUT DOWN of unit g at time t
v_down = {}
for t in time:
    for g in unitind:
        v_down[t, g] = m.addVar(vtype=GRB.BINARY,
                                name='Shut_-_down_at_time_{0}_of_unit_{1}_in_country_{2}'.format(t, g, gendata['region'][g]))

########################################################################################################################

# Variable Unit always on
# vonl = {}
# for g in unitind:
#     vonl[g] = 1

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

# Demand satisfaction
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

            mystore.demanddata.ix[t, n]*(1+high) +
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

# ON/OFF status of unit 'g' at time "t" including the previously introduced binary variable u
on_off_low = {}
for t in time:
    for g in unitind:
        on_off_low[t, g] = m.addConstr(

            gendata['p_min'][g] * u[t, g],

            gb.GRB.LESS_EQUAL,

            genprod[t, g],
            )


on_off_high = {}
for t in time:
    for g in unitind:
        on_off_high[t, g] = m.addConstr(

            genprod[t, g],

            gb.GRB.LESS_EQUAL,

            gendata['p_max'][g] * u[t, g],
            )

# Restriction including Upward and Downward power
# ramping limit for unit 'g' at time "t"
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
            genprod[t, g],
            gb.GRB.LESS_EQUAL,
            genprod[t-1, g] +
            gendata.ramp_up[g] *
            gendata.p_max[g],
            )

# Start-up of a unit associated when the unit is turned ON
start_up = {}
for t in time[1:]:
    for g in unitind:
        start_up[t, g] = m.addConstr(

            v_up[t, g],

            gb.GRB.GREATER_EQUAL,

            u[t, g] -
            u[t-1, g]
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

# Tune GUROBI's parameter to improve the computation

m.params.SimplexPricing = 3
m.params.Heuristics = 0.20
m.params.MIPFocus = 3
m.params.NodeMethod = 2
m.params.Cuts = 1
m.params.MIPGap = 0.1

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

# Create the dataframe
df_windprod = pd.DataFrame({n: {t: windprod[t, n].x for t in time} for n in nodes})
df_solarprod = pd.DataFrame({n: {t: solarprod[t, n].x for t in time} for n in nodes})
df_genprod = pd.DataFrame({g: {t: genprod[t, g].x for t in time} for g in unitind})
df_loadshed = pd.DataFrame({n: {t: loadshed[t, n].x for t in time} for n in nodes})
df_flow = pd.DataFrame({l: {t: flow[t, l].x for t in time} for l in lines})
df_p_exch = pd.DataFrame({n: {t: p_exch[t, n].x for t in time} for n in nodes})
df_hydro = pd.DataFrame({g: {t: genprod[t, g].x for t in time} for g in unitind if gendata['fuel_type'][g] == 'Water'})
df_u_onoff = pd.DataFrame({g: {t: u[t, g].x for t in time} for g in unitind})
df_v_startup = pd.DataFrame({g: {t: v_up[t, g].x for t in time} for g in unitind})
df_v_shutdown = pd.DataFrame({g: {t: v_down[t, g].x for t in time} for g in unitind})

# OPEN the store to export outputs
store = pd.HDFStore('UCMILPstore.h5')

# TEST

store['windprodweekTEST'] = df_windprod
store['solarprodweekTEST'] = df_solarprod
store['genprodweekTEST'] = df_genprod
store['loadshedweekTEST'] = df_loadshed
store['flowweekTEST'] = df_flow
store['p_exchweekTEST'] = df_p_exch
store['hydroprodweekTEST'] = df_hydro
store['onoffweekTEST'] = df_u_onoff
store['startupweekTEST'] = df_v_startup
store['shutdownweekTEST'] = df_v_shutdown


# WEEK 3 Low wind High demand
# Cold start up costs and 50/50 fuel mix

# store['windprodweek3'] = df_windprod
# store['solarprodweek3'] = df_solarprod
# store['genprodweek3'] = df_genprod
# store['loadshedweek3'] = df_loadshed
# store['flowweek3'] = df_flow
# store['p_exchweek3'] = df_p_exch
# store['hydroprodweek3'] = df_hydro
# store['onoffweek3'] = df_u_onoff
# store['startupweek3'] = df_v_startup
# store['shutdownweek3'] = df_v_shutdown

# WEEK 4 High wind High demand
# Cold start up costs and 50/50 fuel mix

# store['windprodweek4'] = df_windprod
# store['solarprodweek4'] = df_solarprod
# store['genprodweek4'] = df_genprod
# store['loadshedweek4'] = df_loadshed
# store['flowweek4'] = df_flow
# store['p_exchweek4'] = df_p_exch
# store['hydroprodweek4'] = df_hydro
# store['onoffweek4'] = df_u_onoff
# store['startupweek4'] = df_v_startup
# store['shutdownweek4'] = df_v_shutdown

# WEEK 26 Low wind Low demand
# Cold start up costs and 50/50 fuel mix

# store['windprodweek26'] = df_windprod
# store['solarprodweek26'] = df_solarprod
# store['genprodweek26'] = df_genprod
# store['loadshedweek26'] = df_loadshed
# store['flowweek26'] = df_flow
# store['p_exchweek26'] = df_p_exch
# store['hydroprodweek26'] = df_hydro
# store['onoffweek26'] = df_u_onoff
# store['startupweek26'] = df_v_startup
# store['shutdownweek26'] = df_v_shutdown

# WEEK 27 High wind Low demand
# Cold start up costs and 50/50 fuel mix

# store['windprodweek27'] = df_windprod
# store['solarprodweek27'] = df_solarprod
# store['genprodweek27'] = df_genprod
# store['loadshedweek27'] = df_loadshed
# store['flowweek27'] = df_flow
# store['p_exchweek27'] = df_p_exch
# store['hydroprodweek27'] = df_hydro
# store['onoffweek27'] = df_u_onoff
# store['startupweek27'] = df_v_startup
# store['shutdownweek27'] = df_v_shutdown

# # Cold start up costs and 20/80 fuel mix
# store['windprod20/80cs'] = df_windprod
# store['solarprod20/80cs'] = df_solarprod
# store['genprod20/80cs'] = df_genprod
# store['loadshed20/80cs'] = df_loadshed
# store['flow20/80cs'] = df_flow
# store['p_exch20/80cs'] = df_p_exch
# store['hydroprod20/80cs'] = df_hydro
# store['onoff20/80cs'] = df_u_onoff
# store['startup20/80cs'] = df_v_startup
# store['shutdown20/80cs'] = df_v_shutdown

# # Cold start up costs and 80/20 fuel mix
# store['windprod80/20cs'] = df_windprod
# store['solarprod80/20cs'] = df_solarprod
# store['genprod80/20cs'] = df_genprod
# store['loadshed80/20cs'] = df_loadshed
# store['flow80/20cs'] = df_flow
# store['p_exch80/20cs'] = df_p_exch
# store['hydroprod80/20cs'] = df_hydro
# store['onoff80/20cs'] = df_u_onoff
# store['startup80/20cs'] = df_v_startup
# store['shutdown80/20cs'] = df_v_shutdown

# # Warm start up costs
# store['windprodws'] = df_windprod
# store['solarprodws'] = df_solarprod
# store['genprodws'] = df_genprod
# store['loadshedws'] = df_loadshed
# store['flowws'] = df_flow
# store['p_exchws'] = df_p_exch
# store['hydroprodws'] = df_hydro
# store['onoffws'] = df_u_onoff
# store['startupws'] = df_v_startup
# store['shutdownws'] = df_v_shutdown

# # Hot start up costs
# store['windprodhs'] = df_windprod
# store['solarprodhs'] = df_solarprod
# store['genprodhs'] = df_genprod
# store['loadshedhs'] = df_loadshed
# store['flowhs'] = df_flow
# store['p_exchhs'] = df_p_exch
# store['hydroprodhs'] = df_hydro
# store['onoffhs'] = df_u_onoff
# store['startuphs'] = df_v_startup
# store['shutdownhs'] = df_v_shutdown


# CLOSE the store
store.close()

