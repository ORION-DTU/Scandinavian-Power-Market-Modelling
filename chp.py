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
mystore.heatdemand = store['heatdemand']
gendata = store['gendata']
daprices = store['daprices']
# CLOSE the store
store.close()


# Display completely in the screen
pd.set_option('display.width', None)

# MODEL DATA
# Time
time = range(8761, 17544)  # 8906

# Regions
nodes = ['DK', 'SE', 'NO', 'FI']

# Indicies by units

# Index for the generatin units
unitind = gendata.index

# Index for the hydropower plants
hydroind = gendata.index[214:254]

# Index for the hydrounits in each country
hydrocountry = {'FI': gendata.index[214:223],
                'NO': gendata.index[224:243],
                'SE': gendata.index[244:254]}

# Index for all the power plants
powplants = {f: [g for g in gendata.index if gendata.techn_type[g] == f] for f in np.unique(gendata.techn_type)}
country = {f: [g for g in gendata.index if gendata.region[g] == f] for f in np.unique(gendata.region)}

# Index for power plants NON CHP
nonchp = set(g for g in gendata.index if gendata.q_max[g] < 0.00001)

# Index for the different power plants
powerplants = {'flex': list(set(powplants['IGCONDENSING'] + powplants['IGEXTRACTION'] + powplants['IGELECTOHEAT']).difference(nonchp)),
               'fixed': list(set(powplants['IGBACKPRESSURE']).difference(nonchp)),
               'nonchp': list(nonchp)}

heatplants = {'flex': list(set(powplants['IGCONDENSING'] + powplants['IGEXTRACTION'] + powplants['IGELECTOHEAT']).difference(nonchp)),
              'fixed': list(set(powplants['IGBACKPRESSURE']).difference(nonchp))}

# Index for all the power plants MINUS the NON-CHP
powerindex = list(set(gendata.index).difference(nonchp))

# Index for the kind of power plants
kindpowpl = {'Condensing': powplants['IGCONDENSING'],
             'Extraction': powplants['IGEXTRACTION'],
             'Electroheat': powplants['IGELECTOHEAT'],
             'Backpressure': powplants['IGBACKPRESSURE']}


countries = {'DK': list(set([g for g in gendata.index if gendata.region[g] == 'DK']).difference(nonchp)),
             'SE': list(set([g for g in gendata.index if gendata.region[g] == 'SE']).difference(nonchp)),
             'NO': list(set([g for g in gendata.index if gendata.region[g] == 'NO']).difference(nonchp)),
             'FI': list(set([g for g in gendata.index if gendata.region[g] == 'FI']).difference(nonchp))}

# PARAMETERS
# ValueOfLostLoad
VOLL = 2500

# Capacity factor of the hydro power plants
hydrocoeff = 0.455

# Efficiency of the charge of the heat storage
effIN = 0.9

# Efficiency of the discharge of the heat storage
effOUT = 0.9

# Losses of the store during the time
loss = 0.90

# Charge and Discharge costs
CH = 0
DIS = 0
SPILL = 0

# Correction factor to consider that just 66% of the heat production is done with chp
# corrfact = 0.66, 0.5, 0.4. 0.74 (ORIGINAL VALUES, CHANGED TO CONSIDER LOSSES)
corrfact = {'DK': 0.7,
            'SE': 0.6,
            'NO': 0.5,
            'FI': 0.8}

maxheatprod = {'DK': 103867,  # 103867
               'SE': 112725,  # 112725
               'NO': 11030,   # 11030
               'FI': 166590}  # 166590
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
gen_dict = gendata.capacity.to_dict()
reg_dict = gendata.region.to_dict()
genprod = {}
for t in time:
    for g in unitind:
        genprod[t, g] = m.addVar(ub=gen_dict[g],
                                 name='Power_production_at_time_{0}_of_unit_{1}_in_country_{2}'.format(t, g, reg_dict[g]))

# Heat production from each unit
q_max_dict = gendata.q_max.to_dict()
reg_dict = gendata.region.to_dict()
heatprod = {}
for t in time:
    for g in powerindex:
        heatprod[t, g] = m.addVar(ub=q_max_dict[g],
                                  name='Heat_production_at_time_{0}_of_unit_{1}_in_country_{2}'.format(t, g, reg_dict[g]))


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

# Storage level
stor = {}
for t in time:
    for n in nodes:
        stor[t, n] = m.addVar(lb=0, ub=maxheatprod[n],  # 1.1*max(mystore.newfylling[g]),
                              name='Storare_level_at_time_{0}_in_country_{1}'.format(t, n))

# Charge and Discharge for the heat storage
charge = {}
for t in time:
    for n in nodes:
        charge[t, n] = m.addVar(lb=0, ub=gb.GRB.INFINITY,
                                name='Charge_at_time_{0}_in_country_{1}'.format(t, n))

discharge = {}
for t in time:
    for n in nodes:
        discharge[t, n] = m.addVar(lb=0, ub=gb.GRB.INFINITY,
                                   name='Discharge_at_time_{0}_in_country_{1}'.format(t, n))

# Spill of heat in case of excessive production
heatspill = {}
for t in time:
    for n in nodes:
        heatspill[t, n] = m.addVar(lb=0, ub=gb.GRB.INFINITY,
                                   name="Heat_spill_in_country_{0}_at_time_{1}".format(n, t))

# Update the variables
m.update()


# OBJECTIVE FUNCTION

m.setObjective(
    quicksum(gendata.marg_cost[g]*(((gendata.p_max[g]-gendata.p_lim[g])/gendata.q_max[g])*heatprod[t, g])
             for t in time for g in powerplants['flex']) +
    quicksum(gendata.marg_cost[g]*genprod[t, g]
             for t in time for g in unitind) +
    quicksum(VOLL*loadshed[t, n]
             for t in time for n in nodes) +
    quicksum(CH*charge[t, n]
             for t in time for n in nodes) +
    quicksum(DIS*discharge[t, n]
             for t in time for n in nodes) +
    quicksum(SPILL*heatspill[t, n]
             for t in time for n in nodes),
    gb.GRB.MINIMIZE)


# SET THE CONSTRAINTS
# Electricity balance
reg_dict = gendata.region.to_dict()
powerbalance = {}
for t in time[1:]:
    for n in nodes:
        powerbalance[t, n] = m.addConstr(
            quicksum(genprod[t, g] for g in unitind if reg_dict[g] == n) +
            windprod[t, n] +
            solarprod[t, n] +
            loadshed[t, n] -
            mystore.impexpdata.ix[t, n], gb.GRB.EQUAL,
            mystore.demanddata.ix[t, n] +
            p_exch[t, n])

# Heat balance
reg_dict = gendata.region.to_dict()
heatbalance = {}
for t in time[1:]:
    for n in nodes:
        heatbalance[t, n] = m.addConstr(
            quicksum(heatprod[t, g] for g in powerindex if reg_dict[g] == n) -
            charge[t, n] +
            discharge[t, n] -
            heatspill[t, n], gb.GRB.EQUAL,
            mystore.heatdemand.ix[t, n] * corrfact[n]
            )


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
for t in time[1:]:
    for n in nodes:
        constrainedflow[t, n] = m.addConstr(
            p_exch[t, n] + quicksum(flow[t, l] for l in lines if l[1] == n), gb.GRB.EQUAL,
            quicksum(flow[t, l] for l in lines if l[0] == n)
            )

# Hydropower limitation
gen_dict = gendata.capacity.to_dict()
hydroprod = {}
for g in hydroind:
    hydroprod[g] = m.addConstr(
        quicksum(genprod[t, g] for t in time), gb.GRB.LESS_EQUAL,
        hydrocoeff * gen_dict[g] * len(time),
        )

# Flexible plants constrains
plim_dict = gendata.p_lim.to_dict()
qmax_dict = gendata.q_max.to_dict()
lowboundpq = {}
for t in time[1:]:
    for g in powerplants['flex']:
        lowboundpq[t, g] = m.addConstr(
            genprod[t, g], gb.GRB.GREATER_EQUAL,
            (plim_dict[g]/qmax_dict[g])*heatprod[t, g]
            )

plim_dict = gendata.p_lim.to_dict()
qmax_dict = gendata.q_max.to_dict()
pmax_dict = gendata.p_max.to_dict()
upboundpq = {}
for t in time[1:]:
    for g in powerplants['flex']:
        upboundpq[t, g] = m.addConstr(
            genprod[t, g], gb.GRB.LESS_EQUAL,
            pmax_dict[g]-((pmax_dict[g]-plim_dict[g])/qmax_dict[g])*heatprod[t, g]
            )

# NON Flexible plants constrains
qmax_dict = gendata.q_max.to_dict()
pmax_dict = gendata.p_max.to_dict()
nonflexpq = {}
for t in time[1:]:
    for g in powerplants['fixed']:
        lowboundpq[t, g] = m.addConstr(
            genprod[t, g], gb.GRB.EQUAL,
            (pmax_dict[g]/qmax_dict[g])*heatprod[t, g]
            )


# the beginning value of the heat storage has be the same of the hystorical data
# for g in powerindex:
#     stor[time[0], g].lb = 10


# Heat storage limitation
heatlimit = {}
for t in time[1:]:
    for n in nodes:
            heatlimit[t, n] = m.addConstr(
                stor[t, n], gb.GRB.EQUAL,
                loss*stor[t-1, n] +
                charge[t, n] * effIN -
                discharge[t, n]/effOUT
                )


# # Heat storage level maintain
# storlevelmaintain = {}
# for g in powerindex:
#     for t in time[168::168]:
#         storlevelmaintain[t, g] = m.addConstr(
#             stor[t-168, g],
#             gb.GRB.EQUAL,
#             stor[t, g])

m.update()

# Compute optimal solution
m.optimize()


# Fix hydro profile
hydroconstr = {}
for t in time[1:]:
    for g in hydroind:
            genprod[t, g].ub = genprod[t, g].x
            genprod[t, g].lb = genprod[t, g].x


# Fix Heat profile
heatconstr = {}
for t in time[1:]:
    for g in powerindex:
        if reg_dict[g] == n:
            heatprod[t, g].ub = heatprod[t, g].x
            heatprod[t, g].lb = heatprod[t, g].x

# # Charge
# chargeconstr = {}
# for t in time[1:]:
#     for n in nodes:
#         charge[t, n].ub = charge[t, n].x
#         charge[t, n].lb = charge[t, n].x

# # # Discharge
# chargeconstr = {}
# for t in time[1:]:
#     for n in nodes:
#         discharge[t, n].ub = discharge[t, n].x
#         discharge[t, n].lb = discharge[t, n].x

m.update()

m.reset()

# Optimize second time to get prices right
m.optimize()


# Create the dataframe
df_oldprice = pd.DataFrame({n: {t: powerbalance[t, n].pi for t in time[1:]} for n in nodes})
df_windprod = pd.DataFrame({n: {t: windprod[t, n].x for t in time[1:]} for n in nodes})
df_solarprod = pd.DataFrame({n: {t: solarprod[t, n].x for t in time[1:]} for n in nodes})
df_genprod = pd.DataFrame({g: {t: genprod[t, g].x for t in time[1:]} for g in unitind})
df_loadshed = pd.DataFrame({n: {t: loadshed[t, n].x for t in time[1:]} for n in nodes})
df_flow = pd.DataFrame({l: {t: flow[t, l].x for t in time[1:]} for l in lines})
df_p_exch = pd.DataFrame({n: {t: p_exch[t, n].x for t in time[1:]} for n in nodes})
df_hydro = pd.DataFrame({g: {t: genprod[t, g].x for t in time[1:]} for g in hydroind})
df_heatprice = pd.DataFrame({n: {t: heatbalance[t, n].pi for t in time[1:]} for n in nodes})
df_heatstore = pd.DataFrame({n: {t: stor[t, n].x for t in time[1:]} for n in nodes})
df_charge = pd.DataFrame({n: {t: charge[t, n].x for t in time[1:]} for n in nodes})
df_discharge = pd.DataFrame({n: {t: discharge[t, n].x for t in time[1:]} for n in nodes})
df_heatprod = pd.DataFrame({g: {t: heatprod[t, g].x for t in time[1:]} for g in powerindex})
df_heatspill = pd.DataFrame({n: {t: heatspill[t, n].x for t in time[1:]} for n in nodes})
df_heatprodbypowerplants = pd.DataFrame({f: df_heatprod[idxs].sum(axis=1) for f, idxs in heatplants.iteritems()})
df_hydroprodbycountry = pd.DataFrame({f: df_hydro[idxs].sum(axis=1) for f, idxs in hydrocountry.iteritems()})
df_genprodbypowerplants = pd.DataFrame({f: df_genprod[idxs].sum(axis=1) for f, idxs in powerplants.iteritems()})
df_heatprodbycountry = pd.DataFrame({f: df_heatprod[idxs].sum(axis=1) for f, idxs in countries.iteritems()})

# # OPEN the store to export outputs
store = pd.HDFStore('outputstore.h5')

# 24hrs storage & different charge and discharge costs
store['priceNEW2013'] = df_oldprice
store['windprodNEW2013'] = df_windprod
store['solarprodNEW2013'] = df_solarprod
store['genprodNEW2013'] = df_genprod
store['loadshedNEW2013'] = df_loadshed
store['flowNEW2013'] = df_flow
store['p_exchNEW2013'] = df_p_exch
store['hydroprodNEW2013'] = df_hydro
store['heatpriceNEW2013'] = df_heatprice
store['heatstoreNEW2013'] = df_heatstore
store['chargeNEW2013'] = df_charge
store['dischargeNEW2013'] = df_discharge
store['heatprodNEW2013'] = df_heatprod
store['heatspillNEW2013'] = df_heatspill
store['storebypowerplantsNEW2013'] = df_storebypowerplants
store['chargebypowerplantsNEW2013'] = df_chargebypowerplants
store['dischargebypowerplantsNEW2013'] = df_dischargebypowerplants
store['heatprodbypowerplantsNEW2013'] = df_heatprodbypowerplants
store['hydroprodbycountryNEW2013'] = df_hydroprodbycountry
store['genprodbypowerplantsNEW2013'] = df_genprodbypowerplants
store['heatspillbypowerplantsNEW2013'] = df_heatspillbypowerplants

# Playing with the model
# store['price168'] = df_oldprice
# store['windprod168'] = df_windprod
# store['solarprod168'] = df_solarprod
# store['genprod168'] = df_genprod
# store['loadshed168'] = df_loadshed
# store['flow168'] = df_flow
# store['p_exch168'] = df_p_exch
# store['hydroprod168'] = df_hydro
# store['heatprice168'] = df_heatprice
# store['heatstore168'] = df_heatstore
# store['charge168'] = df_charge
# store['discharge168'] = df_discharge
# store['heatprod168'] = df_heatprod
# store['heatspill168'] = df_heatspill
# store['heatprodbypowerplants168'] = df_heatprodbypowerplants
# store['hydroprodbycountry168'] = df_hydroprodbycountry
# store['genprodbypowerplants168'] = df_genprodbypowerplants
# store['heatprodbycountry168'] = df_heatprodbycountry
# CLOSE the store
store.close()
