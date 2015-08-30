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
import time as pytime # starttime = pytime.time()
import seaborn as sns
import Expando_class
from Expando_class import Expando

################################################################################################################################################################

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
mystore.winddata= store['winddata']
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
# Display completely in the screen
###

pd.set_option('display.width', None)

################################################################################################################################################################

##############
# MODEL DATA #
##############

###
# Time
###

time = range(8760, 17521)

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

hydrocoeff = 0.455  # Capacity factor of the hydro power plants

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

###
# Own-price elasticity - Epsilon
###

epsilon_dict = {}

epsilon_dict['DK'] = -0.3
epsilon_dict['NO'] = -0.3
epsilon_dict['SE'] = -0.3
epsilon_dict['FI'] = -0.3


###
# Fixed-price calculation - Weigthed average over the time horizon 
###

price_0 = {}

price_0['DK'] = 55.88
price_0['NO'] = 56.62
price_0['SE'] = 55.73
price_0['FI'] = 55.73


###
# PARTITION PROPERTIES
###

# Left hand side partition of the anchor: decrease in demand level/increae in price
part_minus = np.arange(1, 101, 1)
# Right hand side partition of the anchor: increase in demand level/decrease in price
part_plus = np.arange(1, 11, 1)

# Size of the left hand side partition implemented later as % of the demand level 
part_size_minus = 0.01
# Size of the right hand side partition implemented later as % of the demand level 
part_size_plus = 0.01  
# Offset coefficient for more accuracy
offset = 0.005

###
# Partition sizes left/right side of the anchor (Fixed_tariff, Iitial demand)
###

U_negative = {}
for n in nodes:
    for t in time:
        U_negative[t,n] = mystore.demanddata.ix[t, n] * part_size_minus


U_positive = {}
for n in nodes:
    for t in time:
        U_positive[t,n] = mystore.demanddata.ix[t, n] * part_size_plus

###
# Partition/steps on the left/right side of the anchor
###

D_minus = {}
for t in time:
    for n in nodes:
        for i in part_minus:
            D_minus[t,n,i] = mystore.demanddata.ix[t, n] - \
                i * part_size_minus * mystore.demanddata.ix[t, n] + \
                offset * part_size_minus

D_plus = {}
for t in time:
    for n in nodes:
        for i in part_plus:
            D_plus[t,n,i] = mystore.demanddata.ix[t, n] + \
                i * part_size_plus * mystore.demanddata.ix[t, n] + \
                offset * part_size_plus

################################################################################################################################################################

########################################################
# FUNCTION THAT RETURN THE PRICE OF A NEW DEMAND LEVEL #
########################################################

def price_step(d_new, L_t, epsilon, pi_0):
    pi_it = (pi_0 * (d_new - L_t + epsilon * L_t)) / (epsilon * L_t)
    return pi_it 

pi_plus = {}
pi_minus = {}
for n in nodes:
    epsilon = epsilon_dict[n]
    pi_0 = price_0[n]
    for t in time:
        L_t = mystore.demanddata.ix[t, n]
        
        for i in part_minus:
            pi_minus[t,n,i] = price_step(D_minus[t,n,i], L_t, epsilon, pi_0)
        for i in part_plus:
            pi_plus[t,n,i] = price_step(D_plus[t,n,i], L_t, epsilon, pi_0)
        
################################################################################################################################################################

# Associate a demand level to a price from the elastic demand function previously created

###
# Prices on the negative partition
###

Price_negative_perturbation = {}
for t in time:
    for n in nodes:
        for i in part_minus:
            Price_negative_perturbation[t,n,i]= price_step(D_minus[t,n,i], mystore.demanddata.ix[t, n], epsilon_dict[n], price_0[n])
###
# Prices on the positive partition
###

Price_positive_perturbation = {}
for t in time:
    for n in nodes:
        for i in part_plus:
            Price_positive_perturbation[t,n,i]= price_step(D_plus[t,n,i], mystore.demanddata.ix[t, n], epsilon_dict[n], price_0[n])

################################################################################################################################################################

Net_load = {}
for t in time:
    for n in nodes:
        Net_load[t,n] = mystore.demanddata.ix[t,n] + mystore.impexpdata.ix[t,n]

#############################
# CREATE OPTIMIZATION MODEL #
#############################

m = gb.Model('orion')


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
# Load Shedding in each country
###

loadshed = {}
for t in time:
    for n in nodes:
        loadshed[t, n] = m.addVar(ub=Net_load[t,n],
                                  name='Loadshed_at_time_{0}_in_country_{1}'.format(t, n))

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
# Solar Production
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
###
# Negative demand perturbation on the left hand side of the anchor
###

perturbation_negative = {}
for t in time:
    for n in nodes:
        for i in part_minus:
            perturbation_negative[t,n,i] =  m.addVar(lb=0, ub=U_negative[t,n],
                                                     name='Negative_demand_perturbation_at_time_{0}_in_country_{1}_of_{2}_intervals'.format(t, n, i))
###
# Positive demand perturbation on the right hand side of the anchor
###

perturbation_positive = {}
for t in time:
    for n in nodes:
        for i in part_plus:
            perturbation_positive[t,n,i] =  m.addVar(lb=0, ub=U_positive[t,n],
                                                     name='Positive_demand_perturbation_at_time_{0}_in_country_{1}_of_{2}_intervals'.format(t, n, i))

###
# New demand level after flexibility has been applied
###

demand_new = {}
for t in time:
    for n in nodes:
        demand_new[t,n] = m.addVar(lb=0, ub=gb.GRB.INFINITY)

m.update()

################################################################################################################################################################

####################################################
# OBJECTIVE FUNCTION - SOCIAL WELFARE MAXIMIZATION #
####################################################

m.setObjective  ((quicksum(

                pi_minus[t,n,i] * U_negative[t,n] 
                    for t in time for n in nodes for i in part_minus ) +

                quicksum(  
                Price_positive_perturbation[t,n,i] * perturbation_positive[t,n,i] 
                    for t in time for n in nodes for i in part_plus) - 

                quicksum(
                Price_negative_perturbation[t,n,i] * perturbation_negative[t,n,i]
                    for t in time for n in nodes for i in part_minus)) -

                quicksum(
                gendata.marg_cost[g]*genprod[t, g]
                    for t in time for g in unitind),

                gb.GRB.MAXIMIZE)

################################################################################################################################################################

#######################
# SET THE CONSTRAINTS #
#######################

###
# Demand satisfaction
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

            mystore.demanddata.ix[t, n] +
            p_exch[t, n] +

            quicksum(
            perturbation_positive[t,n,i] for i in part_plus) -

            quicksum(
            perturbation_negative[t,n,i] for i in part_minus)
            )

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

###
# New demand levels
###

fixed_demand = {}
for t in time:
    for n in nodes:
        fixed_demand[t,n] = m.addConstr(

            demand_new[t,n],

            gb.GRB.EQUAL,

            mystore.demanddata.ix[t, n] + 
            sum(perturbation_positive[t,n,i] for i in part_plus) - 
            sum(perturbation_negative[t,n,i] for i in part_minus),

            )


################################################################################################################################################################

m.update()

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

################################################################################################################################################################

#########################
# CREATE THE DATAFRAMES #
#########################

df_windprod = pd.DataFrame({n: {t: windprod[t, n].x for t in time} for n in nodes})
df_solarprod = pd.DataFrame({n: {t: solarprod[t, n].x for t in time} for n in nodes})
df_genprod = pd.DataFrame({g: {t: genprod[t, g].x for t in time} for g in unitind})
# df_loadshed = pd.DataFrame({n: {t: loadshed[t, n].x for t in time} for n in nodes})
df_flow = pd.DataFrame({l: {t: flow[t, l].x for t in time} for l in lines})
df_p_exch = pd.DataFrame({n: {t: p_exch[t, n].x for t in time} for n in nodes})
df_hydro = pd.DataFrame({g: {t: genprod[t, g].x for t in time} for g in unitind if gendata['fuel_type'][g] == 'Water'})
df_price = pd.DataFrame({n: {t: powerbalance[t, n].pi for t in time} for n in nodes})
df_hydroprod = pd.DataFrame({f: df_hydro[idxs].sum(axis=1) for f, idxs in hydrocountry.iteritems()})
df_demand_new = pd.DataFrame({n: {t: demand_new[t, n].x for t in time} for n in nodes})

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

# 2013 with mean price of 2012 (around 70)

# Own price elasticity = 0.00

# store['windprod_base_2013_calculated'] = df_windprod
# store['solarprod_base_2013_calculated'] = df_solarprod
# store['genprod_base_2013_calculated'] = df_genprod
# store['flow_base_2013_calculated'] = df_flow
# store['p_exch_base_2013_calculated'] = df_p_exch
# store['hydroprod_base_2013_calculated'] = df_hydro
# store['marg_cost_base_2013_calculated'] = df_price
# store['hydroprod_total_base_2013_calculated'] = df_hydroprod
# store['demand_new_base_2013_calculated'] = df_demand_new

###
# Perturbations included and limited by U_positive and U_negative 
###


# Own price elasticity = -0.1

# store['windprod_DR_0_1_2013_calculated'] = df_windprod
# store['solarprod_DR_0_1_2013_calculated'] = df_solarprod
# store['genprod_DR_0_1_2013_calculated'] = df_genprod
# store['flow_DR_0_1_2013_calculated'] = df_flow
# store['p_exch_DR_0_1_2013_calculated'] = df_p_exch
# store['hydroprod_DR_0_1_2013_calculated'] = df_hydro
# store['marg_cost_DR_0_1_2013_calculated'] = df_price
# store['hydroprod_total_DR_0_1_2013_calculated'] = df_hydroprod
# store['demand_new_DR_0_1_2013_calculated'] = df_demand_new


# Own price elasticity = -0.2

# store['windprod_DR_0_2_2013_calculated'] = df_windprod
# store['solarprod_DR_0_2_2013_calculated'] = df_solarprod
# store['genprod_DR_0_2_2013_calculated'] = df_genprod
# store['flow_DR_0_2_2013_calculated'] = df_flow
# store['p_exch_DR_0_2_2013_calculated'] = df_p_exch
# store['hydroprod_DR_0_2_2013_calculated'] = df_hydro
# store['marg_cost_DR_0_2_2013_calculated'] = df_price
# store['hydroprod_total_DR_0_2_2013_calculated'] = df_hydroprod
# store['demand_new_DR_0_2_2013_calculated'] = df_demand_new


# Own price elasticity = -0.3

# store['windprod_DR_0_3_2013_calculated'] = df_windprod
# store['solarprod_DR_0_3_2013_calculated'] = df_solarprod
# store['genprod_DR_0_3_2013_calculated'] = df_genprod
# store['flow_DR_0_3_2013_calculated'] = df_flow
# store['p_exch_DR_0_3_2013_calculated'] = df_p_exch
# store['hydroprod_DR_0_3_2013_calculated'] = df_hydro
# store['marg_cost_DR_0_3_2013_calculated'] = df_price
# store['hydroprod_total_DR_0_3_2013_calculated'] = df_hydroprod
# store['demand_new_DR_0_3_2013_calculated'] = df_demand_new


################################################################################################################################################################


# 2013 with mean price of 2013 (around 55)


# Own price elasticity = 0.00

# store['windprod_base_2013_calculated_2'] = df_windprod
# store['solarprod_base_2013_calculated_2'] = df_solarprod
# store['genprod_base_2013_calculated_2'] = df_genprod
# store['flow_base_2013_calculated_2'] = df_flow
# store['p_exch_base_2013_calculated_2'] = df_p_exch
# store['hydroprod_base_2013_calculated_2'] = df_hydro
# store['marg_cost_base_2013_calculated_2'] = df_price
# store['hydroprod_total_base_2013_calculated_2'] = df_hydroprod
# store['demand_new_base_2013_calculated_2'] = df_demand_new


###
# Perturbations included and limited by U_positive and U_negative 
###

# Own price elasticity = -0.1

# store['windprod_DR_0_1_2013_calculated_2'] = df_windprod
# store['solarprod_DR_0_1_2013_calculated_2'] = df_solarprod
# store['genprod_DR_0_1_2013_calculated_2'] = df_genprod
# store['flow_DR_0_1_2013_calculated_2'] = df_flow
# store['p_exch_DR_0_1_2013_calculated_2'] = df_p_exch
# store['hydroprod_DR_0_1_2013_calculated_2'] = df_hydro
# store['marg_cost_DR_0_1_2013_calculated_2'] = df_price
# store['hydroprod_total_DR_0_1_2013_calculated_2'] = df_hydroprod
# store['demand_new_DR_0_1_2013_calculated_2'] = df_demand_new


# Own price elasticity = -0.2

# store['windprod_DR_0_2_2013_calculated_2'] = df_windprod
# store['solarprod_DR_0_2_2013_calculated_2'] = df_solarprod
# store['genprod_DR_0_2_2013_calculated_2'] = df_genprod
# store['flow_DR_0_2_2013_calculated_2'] = df_flow
# store['p_exch_DR_0_2_2013_calculated_2'] = df_p_exch
# store['hydroprod_DR_0_2_2013_calculated_2'] = df_hydro
# store['marg_cost_DR_0_2_2013_calculated_2'] = df_price
# store['hydroprod_total_DR_0_2_2013_calculated_2'] = df_hydroprod
# store['demand_new_DR_0_2_2013_calculated_2'] = df_demand_new


# Own price elasticity = -0.3

store['windprod_DR_0_3_2013_calculated_2'] = df_windprod
store['solarprod_DR_0_3_2013_calculated_2'] = df_solarprod
store['genprod_DR_0_3_2013_calculated_2'] = df_genprod
store['flow_DR_0_3_2013_calculated_2'] = df_flow
store['p_exch_DR_0_3_2013_calculated_2'] = df_p_exch
store['hydroprod_DR_0_3_2013_calculated_2'] = df_hydro
store['marg_cost_DR_0_3_2013_calculated_2'] = df_price
store['hydroprod_total_DR_0_3_2013_calculated_2'] = df_hydroprod
store['demand_new_DR_0_3_2013_calculated_2'] = df_demand_new



###
# CLOSE the store
###

store.close()


################################################################################################################################################################


