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
mystore.newglacialinflow0_005 = store['newglacialinflow0_005']
mystore.newglacialinflow0_01 = store['newglacialinflow0_01']
mystore.newglacialinflow0_015 = store['newglacialinflow0_015']
mystore.newglacialinflow0_02 = store['newglacialinflow0_02']
mystore.newfylling = store['newfylling']
mystore.newriverinflow = store['newriverinflow']
mystore.newmaxvolume = store['newmaxvolume']
mystore.weekdamfill = store['weekdamfill']
# CLOSE the store
store.close()


# Display completely in the screen
pd.set_option('display.width', None)

# MODEL DATA
# Time
time = range(0, 8788)  # 2012: 1, 8789  # 2013: 8762, 17544
weekhours = time[1::169]
weekhours[:len(mystore.weekdamfill)]

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
loss = 0.99

# Charge and Discharge costs
CH = 0
DIS = 0
SPILL = 0
WASTECOST = 10

# Down deviation and updeviaton costs associated
DD = 0.9
UD = 0.9

# Correction factor to consider that just 66% of the heat production is done with chp
# corrfact = 0.66, 0.5, 0.4. 0.74 (ORIGINAL VALUES, CHANGED TO CONSIDER LOSSES) 0.7 0.6 0.5 0.8 are the original!!
corrfact = {'DK': 0.7,
            'SE': 0.6,
            'NO': 0.5,
            'FI': 0.8}

maxheatprod = {'DK': 103867,
               'SE': 112725,
               'NO': 11030,
               'FI': 166590}
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

# HEAT VARIABLES

# Heat production
q_max_dict = gendata.q_max.to_dict()
reg_dict = gendata.region.to_dict()
heatprod = {}
for t in time:
    for g in powerindex:
        heatprod[t, g] = m.addVar(ub=q_max_dict[g],
                                  name='Heat_production_at_time_{0}_of_unit_{1}_in_country_{2}'.format(t, g, reg_dict[g]))

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

# Waste of heat in case of excessive production
waste = {}
for t in time:
    for n in nodes:
        waste[t, n] = m.addVar(lb=0, ub=gb.GRB.INFINITY,
                               name="Waste_in_country_{0}_at_time_{1}".format(n, t))

# HYDROPOWER VARIABLES

# Hydro reservoir level
res_lev = {}
for t in time:
    for g in hydroind:
        res_lev[t, g] = m.addVar(ub=gb.GRB.INFINITY,  # 1.1*max(mystore.newfylling[g]),
                                 name='Reservoir_level_at_time_{0}_of_unit_{1}'.format(t, g))


# Up deviation from historical data
up_dev = {}
for t in time:
    for g in hydroind:
        up_dev[t, g] = m.addVar(ub=gb.GRB.INFINITY, lb=0,
                                name='Up_deviation_at_time_{0}_of_unit_{1}_from_the_hystorical_profile'.format(t, g))

# Down deviation from historical data
down_dev = {}
for t in time:
    for g in hydroind:
        down_dev[t, g] = m.addVar(ub=gb.GRB.INFINITY, lb=0,
                                  name='Down_deviation_at_time_{0}_of_unit_{1}_from_the_hystorical_profile'.format(t, g))

# Possibility to spill water
hydrospill = {}
for t in time:
    for g in hydroind:
        hydrospill[t, g] = m.addVar(name="Hydro spill at gen. {0} at time {1}".format(g, t))

m.update()

# The beginning value of the reservoirs has be the same of the hystorical data
for g in hydroind:
    res_lev[time[0], g].lb = mystore.newfylling[g].ix[0]


# Update the variables
m.update()


# OBJECTIVE FUNCTION
margi_cost_dict = gendata.marg_cost.to_dict()
p_maxi_dict = gendata.p_max.to_dict()
p_limi_dict = gendata.p_lim.to_dict()
q_maxi_dict = gendata.q_max.to_dict()

m.setObjective(
    quicksum(margi_cost_dict[g]*(genprod[t, g]+((p_maxi_dict[g]-p_limi_dict[g])/q_maxi_dict[g])*heatprod[t, g])
             for t in time for g in powerplants['flex']) +
    quicksum(margi_cost_dict[g]*genprod[t, g]
             for t in time for g in unitind) +
    quicksum(VOLL*loadshed[t, n]
             for t in time for n in nodes) +
    quicksum(CH*charge[t, n]
             for t in time for n in nodes) +
    quicksum(DIS*discharge[t, n]
             for t in time for n in nodes) +
    quicksum(WASTECOST*waste[t, n]
             for t in time for n in nodes) +
    quicksum(DD*down_dev[t, g]
             for t in time[1:] for g in hydroind) +
    quicksum(UD*up_dev[t, g]
             for t in time[1:] for g in hydroind),
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
            quicksum(heatprod[t, g] for g in powerindex if reg_dict[g] == n) +
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

# Level of the reservoir at the beginning (time 0) has to be equal at the end (time[-1])
reslevmaintain = {}
for g in hydroind:
    reslevmaintain[g] = m.addConstr(
        res_lev[time[0], g],
        gb.GRB.EQUAL,
        res_lev[time[-1], g])

# Deviations from the historical data
reslevdeviations = {}
for g in hydroind:
    for t, rt in zip(weekhours, mystore.weekdamfill[g]):
        reslevdeviations[t, g] = m.addConstr(
            res_lev[t, g],
            gb.GRB.EQUAL,
            rt +
            up_dev[t, g] -
            down_dev[t, g]
            )

# Flexible power plants constrains
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

# NON Flexible power plants constrains
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

# Heat balance maintain
balance = {}
for t in time[1:]:
    for n in nodes:
        balance[t, n] = m.addConstr(
            heatspill[t, n] -
            charge[t, n] -
            waste[t, n], gb.GRB.EQUAL,
            0)


# Heat storage limitation
heatlimit = {}
for t in time[1:]:
    for n in nodes:
            heatlimit[t, n] = m.addConstr(
                stor[t, n], gb.GRB.EQUAL,
                loss*stor[t-1, n] +
                charge[t, n]*effIN -
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


# Fix hydro
hydroconstr = {}
for t in time[1:]:
    for g in hydroind:
            genprod[t, g].ub = genprod[t, g].x
            genprod[t, g].lb = genprod[t, g].x


# Fix heat storage profile production
heatconstr = {}
for t in time[1:]:
    for g in powerindex:
        if reg_dict[g] == n:
            heatprod[t, g].ub = heatprod[t, g].x
            heatprod[t, g].lb = heatprod[t, g].x

# # Fix the Charge
# chargeconstr = {}
# for t in time[1:]:
#     for n in nodes:
#         charge[t, n].ub = charge[t, n].x
#         charge[t, n].lb = charge[t, n].x

# # # Fix the Discharge
# dischargeconstr = {}
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
df_waste = pd.DataFrame({n: {t: waste[t, n].x for t in time[1:]} for n in nodes})
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
df_heatprodflex = pd.DataFrame({f: df_heatprod[idxs].sum(axis=1) for f, idxs in heatplants.iteritems()})


df_hydro = pd.DataFrame({g: {t: genprod[t, g].x for t in time} for g in hydroind})
df_down_dev = pd.DataFrame({g: {t: down_dev[t, g].x for t in time} for g in hydroind})
df_up_dev = pd.DataFrame({g: {t: up_dev[t, g].x for t in time} for g in hydroind})
df_res_lev = pd.DataFrame({g: {t: res_lev[t, g].x for t in time} for g in hydroind})
df_hydroprod = pd.DataFrame({f: df_hydro[idxs].sum(axis=1) for f, idxs in hydrocountry.iteritems()})
df_country_hydro = pd.DataFrame({c: df_res_lev[idxs].sum(axis=1) for c, idxs in hydrocountry.iteritems()})
df_hydrospill = pd.DataFrame({g: {t: hydrospill[t, g].x for t in time} for g in hydroind})
df_country_hydrospill = pd.DataFrame({l: df_hydrospill[idxs].sum(axis=1) for l, idxs in hydrocountry.iteritems()})
df_country_reslev = pd.DataFrame({l: df_res_lev[idxs].sum(axis=1) for l, idxs in hydrocountry.iteritems()})
df_country_updev = pd.DataFrame({l: df_up_dev[idxs].sum(axis=1) for l, idxs in hydrocountry.iteritems()})
df_country_downdev = pd.DataFrame({l: df_down_dev[idxs].sum(axis=1) for l, idxs in hydrocountry.iteritems()})


df_country_reslev = pd.DataFrame({l: mystore.weekdamfill[idxs].sum(axis=1) for l, idxs in hydrocountry.iteritems()})


# # OPEN the store to export outputs
store = pd.HDFStore('jointstore.h5')

# 2012
# store['priceJOINT12'] = df_oldprice
# store['windprodJOINT12'] = df_windprod
# store['solarprodJOINT12'] = df_solarprod
# store['genprodJOINT12'] = df_genprod
# store['loadshedJOINT12'] = df_loadshed
# store['flowJOINT12'] = df_flow
# store['p_exchJOINT12'] = df_p_exch
# store['hydroprodJOINT12'] = df_hydro
# store['heatpriceJOINT12'] = df_heatprice
# store['heatstoreJOINT12'] = df_heatstore
# store['chargeJOINT12'] = df_charge
# store['dischargeJOINT12'] = df_discharge
# store['heatprodJOINT12'] = df_heatprod
# store['heatspillJOINT12'] = df_heatspill
# store['heatprodbypowerplantsJOINT12'] = df_heatprodbypowerplants
# store['hydroprodbycountryJOINT12'] = df_hydroprodbycountry
# store['genprodbypowerplantsJOINT12'] = df_genprodbypowerplants
# store['heatprodbycountryJOINT12'] = df_heatprodbycountry
# store['downdeviationJOINT12'] = df_down_dev
# store['updeviationJOINT12'] = df_up_dev
# store['reservlevelJOINT12'] = df_res_lev
# store['hydroproductionJOINT12'] = df_hydroprod
# store['countryhydroJOINT12'] = df_country_hydro
# store['hydrospillJOINT12'] = df_hydrospill
# store['countryhydrospillJOINT12'] = df_country_hydrospill


# 2013
# store['priceJOINT13'] = df_oldprice
# store['windprodJOINT13'] = df_windprod
# store['solarprodJOINT13'] = df_solarprod
# store['genprodJOINT13'] = df_genprod
# store['loadshedJOINT13'] = df_loadshed
# store['flowJOINT13'] = df_flow
# store['p_exchJOINT13'] = df_p_exch
# store['hydroprodJOINT13'] = df_hydro
# store['heatpriceJOINT13'] = df_heatprice
# store['heatstoreJOINT13'] = df_heatstore
# store['chargeJOINT13'] = df_charge
# store['dischargeJOINT13'] = df_discharge
# store['heatprodJOINT13'] = df_heatprod
# store['heatspillJOINT13'] = df_heatspill
# store['heatprodbypowerplantsJOINT13'] = df_heatprodbypowerplants
# store['hydroprodbycountryJOINT13'] = df_hydroprodbycountry
# store['genprodbypowerplantsJOINT13'] = df_genprodbypowerplants
# store['heatprodbycountryJOINT13'] = df_heatprodbycountry
# store['downdeviationJOINT13'] = df_down_dev
# store['updeviationJOINT13'] = df_up_dev
# store['reservlevelJOINT13'] = df_res_lev
# store['hydroproductionJOINT13'] = df_hydroprod
# store['countryhydroJOINT13'] = df_country_hydro
# store['hydrospillJOINT13'] = df_hydrospill
# store['countryhydrospillJOINT13'] = df_country_hydrospill


# CLOSE the store
store.close()
