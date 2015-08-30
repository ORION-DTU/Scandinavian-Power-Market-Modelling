####################
# IMPORT LIBRARIES #
####################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import xlrd
import gurobipy as gb
from gurobipy import *
from collections import defaultdict
from numpy import *
import os
import time
import xlwt
import time as pytime  # starttime = pytime.time()
import seaborn as sns
import Expando_class
from Expando_class import Expando

##################################################################################################################################################

####################
# INPUT DATA STORE # 
####################

###
# OPEN the store
###

store = pd.HDFStore('store_demand_response.h5')

###
# LOAD the data
###

mystore = Expando()
mystore.winddata = store['winddata']
mystore.sundata = store['sundata']
mystore.demanddata = store['demanddata']
mystore.impexpdata = store['impexpdata']
gendata = store['gendata']
linecap = store['linecap']
daprices = store['daprices']

###
# CLOSE the store
###

store.close()

###
# Load the new calculated demand profile from the first solve
###

store = pd.HDFStore('outputstore_DR.h5')


########
# 2013 #
########

# based on 2012 average prices (around 70)

# df_demand_new = store['demand_new_base_2013_calculated']
# df_demand_new = store['demand_new_DR_0_1_2013_calculated']
# df_demand_new = store['demand_new_DR_0_2_2013_calculated']
# df_demand_new = store['demand_new_DR_0_3_2013_calculated']

# based on 2013 average prices (around 55)

# df_demand_new = store['demand_new_base_2013_calculated_2']
# df_demand_new = store['demand_new_DR_0_1_2013_calculated_2']
# df_demand_new = store['demand_new_DR_0_2_2013_calculated_2']
df_demand_new = store['demand_new_DR_0_3_2013_calculated_2']

###
# Display completely in the screen
###

pd.set_option('display.width', None)

##################################################################################################################################################

##############
# MODEL DATA #
##############

###
# Time
###

time = range(8760,17521)

###
# Regions
###

nodes = ['DK', 'SE', 'NO', 'FI']

###
# Index by units
###

unitind = gendata.index

###
# Position of hydro generator in index per country
###

hydrocountry = {'FI': gendata.index[214:223],
                'NO': gendata.index[224:243],
                'SE': gendata.index[244:254]}

################################################################################################################################################################

##############
# PARAMETERS #
##############

###
# Hydro coefficient
###

hydrocoeff = 0.455  # Yearly capacity factor of the hydro power plants 

###
# Lines
###

lines = [('DK', 'NO'), ('DK', 'SE'), ('SE', 'FI'), ('SE', 'NO'), ('NO', 'FI')]

lineinfo = {}

lineinfo[('DK', 'NO')] = {'linecapacity': 1000, 'x': 1, 'otherinfo': []}
lineinfo[('DK', 'SE')] = {'linecapacity': 2050, 'x': 1, 'otherinfo': []}
lineinfo[('SE', 'FI')] = {'linecapacity': 2050, 'x': 1, 'otherinfo': []}
lineinfo[('SE', 'NO')] = {'linecapacity': 3450, 'x': 1, 'otherinfo': []}
lineinfo[('NO', 'FI')] = {'linecapacity': 100, 'x': 1, 'otherinfo': []}


##################################################################################################################################################

#############################
# CREATE OPTIMIZATION MODEL #
#############################

m = gb.Model("Demand_response")


########################
# CREATE THE VARIABLES #
########################

###
# Power production from each unit
###

# Temporary dictionnary to gain in processing time
gen_dic_cap = gendata.capacity.to_dict()
gen_dic_reg = gendata.region.to_dict()

genprod = {}
for t in time:
    for g in unitind:
        genprod[t, g] = m.addVar(ub=gen_dic_cap[g],
                                 name='Power_production_at_time_{0}_of_unit_{1}_in_country_{2}'.format(t, g, gen_dic_reg[g]))
###
# Flow over the lines (power actually flowing over a specific line, gives an
# information about where the power is going to)
###

flow = {}
for t in time:
    for l in lines:
        flow[t, l] = m.addVar(lb=-lineinfo[l]['linecapacity'],
                              ub=lineinfo[l]['linecapacity'],
                              name='flow_at_time_{0}_in_line_{1}'.format(t, l))
###
# Sun Production
###

solarprod = {}
for t in time:
    for n in nodes:
        solarprod[t, n] = m.addVar(ub=mystore.sundata.ix[t, n],
                                   name='Solar_production_at_time_{0}_in_country_{1}'.format(t, n))
###
# Wind production
###

windprod = {}
for t in time:
    for n in nodes:
        windprod[t, n] = m.addVar(ub=mystore.winddata.ix[t, n],
                                  name='Wind_production_at_time_{0}_in_country_{1}'.format(t, n))
###
# Power exchanged (This is the power "left" within a zone j which is send/taken to/from the network)
###

p_exch = {}
for t in time:
    for n in nodes:
        p_exch[t, n] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY,
                                name='Power_surplus_at_time_{0}_in_country_{1}'.format(t, n))

m.update()

##################################################################################################################################################

#################################################
# OBJECTIVE FUNCTION - SYSTEM COST MINIMIZATION #
#################################################

m.setObjective(

    quicksum(gendata.marg_cost[g] * genprod[t, g]
             for t in time for g in unitind), 

    gb.GRB.MINIMIZE)

##################################################################################################################################################

#######################
# SET THE CONSTRAINTS #
#######################

###
# Demand satisfaction - Balance equation
###

powerbalance = {}
for t in time:
    for n in nodes:
        powerbalance[t, n] = m.addConstr(

            quicksum(genprod[t, g] for g in unitind if gendata.region[g] == n) +
            windprod[t, n] +
            solarprod[t, n] -
            mystore.impexpdata.ix[t, n], 

            gb.GRB.EQUAL,

            df_demand_new.ix[t,n] +
            p_exch[t, n])

###
# Constrained flow
###

constrainedflow = {}
for t in time:
    for n in nodes:
        constrainedflow[t, n] = m.addConstr(

            p_exch[t, n] + quicksum(flow[t, l] for l in lines if l[1] == n),

             gb.GRB.EQUAL,

            quicksum(flow[t, l] for l in lines if l[0] == n)
            )

###
# Hydropower limitation
###

hydroprod = {}
for g in unitind:
    if gendata.fuel_type[g] == 'Water':
        hydroprod[g] = m.addConstr(

            quicksum(genprod[t, g] for t in time), 

            gb.GRB.LESS_EQUAL,

            hydrocoeff * gendata.capacity[g] * len(time),
            )

m.update()

################################################################################################################################################################

###
# Compute optimal solution
###

m.optimize()

###
# Fix hydro power profile
###

hydroconstr = {}
for t in time:
    for g in unitind:
        if gendata.fuel_type[g] == 'Water':
                genprod[t, g].ub = genprod[t, g].x
                genprod[t, g].lb = genprod[t, g].x


m.update()

m.reset()

###
# Optimize second time to get prices right after fixing the hydro profile
###

m.optimize()

##################################################################################################################################################

#########################
# CREATE THE DATAFRAMES #
#########################

df_windprod = pd.DataFrame({n: {t: windprod[t, n].x for t in time} for n in nodes})
df_solarprod = pd.DataFrame({n: {t: solarprod[t, n].x for t in time} for n in nodes})
df_genprod = pd.DataFrame({g: {t: genprod[t, g].x for t in time} for g in unitind})
df_flow = pd.DataFrame({l: {t: flow[t, l].x for t in time} for l in lines})
df_p_exch = pd.DataFrame({n: {t: p_exch[t, n].x for t in time} for n in nodes})
df_hydro = pd.DataFrame({g: {t: genprod[t, g].x for t in time} for g in unitind if gendata['fuel_type'][g] == 'Water'})
df_price = pd.DataFrame({n: {t: powerbalance[t, n].pi for t in time[1:]} for n in nodes})
df_hydroprod = pd.DataFrame({f: df_hydro[idxs].sum(axis=1) for f, idxs in hydrocountry.iteritems()})

###
# OPEN the store to export outputs
###

store = pd.HDFStore('outputstore_DR.h5')


###
# Save the data in the store
###


########
# 2013 #
########

# Based on 2012 mean price (around 70)

# Own price elasticity = 0.00

# store['windprod_base_final_2013_calculated'] = df_windprod
# store['solarprod_base_final_2013_calculated'] = df_solarprod
# store['genprod_base_final_2013_calculated'] = df_genprod
# store['flow_base_final_2013_calculated'] = df_flow
# store['p_exch_base_final_2013_calculated'] = df_p_exch
# store['hydroprod_base_final_2013_calculated'] = df_hydro
# store['marg_cost_base_final_2013_calculated'] = df_price
# store['hydroprod_total_base_final_2013_calculated'] = df_hydroprod

###
# Perturbations included and limited by U_positive and U_negative
###

# Own price elasticity = -0.1

# store['windprod_DR_0_1_final_2013_calculated'] = df_windprod
# store['solarprod_DR_0_1_final_2013_calculated'] = df_solarprod
# store['genprod_DR_0_1_final_2013_calculated'] = df_genprod
# store['flow_DR_0_1_final_2013_calculated'] = df_flow
# store['p_exch_DR_0_1_final_2013_calculated'] = df_p_exch
# store['hydroprod_DR_0_1_final_2013_calculated'] = df_hydro
# store['marg_cost_DR_0_1_final_2013_calculated'] = df_price
# store['hydroprod_total_DR_0_1_final_2013_calculated'] = df_hydroprod


# Own price elasticity = -0.2

# store['windprod_DR_0_2_final_2013_calculated'] = df_windprod
# store['solarprod_DR_0_2_final_2013_calculated'] = df_solarprod
# store['genprod_DR_0_2_final_2013_calculated'] = df_genprod
# store['flow_DR_0_2_final_2013_calculated'] = df_flow
# store['p_exch_DR_0_2_final_2013_calculated'] = df_p_exch
# store['hydroprod_DR_0_2_final_2013_calculated'] = df_hydro
# store['marg_cost_DR_0_2_final_2013_calculated'] = df_price
# store['hydroprod_total_DR_0_2_final_2013_calculated'] = df_hydroprod


# Own price elasticity = -0.3

# store['windprod_DR_0_3_final_2013_calculated'] = df_windprod
# store['solarprod_DR_0_3_final_2013_calculated'] = df_solarprod
# store['genprod_DR_0_3_final_2013_calculated'] = df_genprod
# store['flow_DR_0_3_final_2013_calculated'] = df_flow
# store['p_exch_DR_0_3_final_2013_calculated'] = df_p_exch
# store['hydroprod_DR_0_3_final_2013_calculated'] = df_hydro
# store['marg_cost_DR_0_3_final_2013_calculated'] = df_price
# store['hydroprod_total_DR_0_3_final_2013_calculated'] = df_hydroprod

##################################################################################################################################################

########
# 2013 #
########

# Based on 2013 mean price (around 55)

# Own price elasticity = 0.00

# store['windprod_base_final_2013_calculated_2'] = df_windprod
# store['solarprod_base_final_2013_calculated_2'] = df_solarprod
# store['genprod_base_final_2013_calculated_2'] = df_genprod
# store['flow_base_final_2013_calculated_2'] = df_flow
# store['p_exch_base_final_2013_calculated_2'] = df_p_exch
# store['hydroprod_base_final_2013_calculated_2'] = df_hydro
# store['marg_cost_base_final_2013_calculated_2'] = df_price
# store['hydroprod_total_base_final_2013_calculated_2'] = df_hydroprod

###
# Perturbations included and limited by U_positive and U_negative
###

# Own price elasticity = -0.1

# store['windprod_DR_0_1_final_2013_calculated_2'] = df_windprod
# store['solarprod_DR_0_1_final_2013_calculated_2'] = df_solarprod
# store['genprod_DR_0_1_final_2013_calculated_2'] = df_genprod
# store['flow_DR_0_1_final_2013_calculated_2'] = df_flow
# store['p_exch_DR_0_1_final_2013_calculated_2'] = df_p_exch
# store['hydroprod_DR_0_1_final_2013_calculated_2'] = df_hydro
# store['marg_cost_DR_0_1_final_2013_calculated_2'] = df_price
# store['hydroprod_total_DR_0_1_final_2013_calculated_2'] = df_hydroprod


# Own price elasticity = -0.2

# store['windprod_DR_0_2_final_2013_calculated_2'] = df_windprod
# store['solarprod_DR_0_2_final_2013_calculated_2'] = df_solarprod
# store['genprod_DR_0_2_final_2013_calculated_2'] = df_genprod
# store['flow_DR_0_2_final_2013_calculated_2'] = df_flow
# store['p_exch_DR_0_2_final_2013_calculated_2'] = df_p_exch
# store['hydroprod_DR_0_2_final_2013_calculated_2'] = df_hydro
# store['marg_cost_DR_0_2_final_2013_calculated_2'] = df_price
# store['hydroprod_total_DR_0_2_final_2013_calculated_2'] = df_hydroprod


# Own price elasticity = -0.3

store['windprod_DR_0_3_final_2013_calculated_2'] = df_windprod
store['solarprod_DR_0_3_final_2013_calculated_2'] = df_solarprod
store['genprod_DR_0_3_final_2013_calculated_2'] = df_genprod
store['flow_DR_0_3_final_2013_calculated_2'] = df_flow
store['p_exch_DR_0_3_final_2013_calculated_2'] = df_p_exch
store['hydroprod_DR_0_3_final_2013_calculated_2'] = df_hydro
store['marg_cost_DR_0_3_final_2013_calculated_2'] = df_price
store['hydroprod_total_DR_0_3_final_2013_calculated_2'] = df_hydroprod

###
# CLOSE the store
###

store.close()


################################################################################################################################################################

