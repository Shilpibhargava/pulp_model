# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 09:52:05 2018
Simon.Guo Edited on April 18, 2019 to make the code only for RDC_Store Alignment

@author: z002n9p Simon Guo
"""

import pandas as pd
import numpy as np
#from gurobipy import *
import os
import datetime

import pulp as plp



"""
######################################
Create the input/output file directory
######################################
"""
#os.chdir('\\\\corp.target.com\\dfsroot\\Collaboration\\Merchandising-SupplyChain\\93139000-NetworkPlanning\\2018 Projects\\Work Space\\Re_Alignment\\Assignment_Code')
os.chdir('/Users/z003cnf/Desktop/Fall_realignment_2021_shilpi/')
dataDir = os.getcwd()
print(dataDir)
inputDataDir = os.path.join(dataDir, 'Input')
outputDataDir = os.path.join(dataDir, 'Output')


"""
######################################
function to calculate direct distance in Python
######################################
"""
def haversine_dist(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 3960   # in miles
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


"""
######################################
Main model to run for RDC Store Alignment
######################################
"""

""" Main model part 1
### Data Input, reading main excel input file
### Load data in each tab, and create dataframes input
"""
s_file = "2021_Fall_Re-Alignment_D1_Aug5_min_crtns" 
#s_file = "2021_Fall_Re-Alignment_baseline_July14"
#s_file = "2021_Fall_Re-Alignment_C1_July14" 
#2021_Fall_Re-Alignment_A1_June22
excel_input_raw = pd.ExcelFile(os.path.join(inputDataDir, s_file + ".xlsx" ))

"""
---------------------------- Global parameters (Excel Tab: parameters) ----------------------------
"""
df_para = excel_input_raw.parse("parameters")
dict_para = df_para.set_index('parameter_name')['value'].to_dict()
p_year = dict_para['year']
p_RDC_penalty = dict_para['Penalty_RDC']
p_UDC_penalty = dict_para['Penalty_UDC']
p_dock_spare = dict_para['dock_door_spare']
p_storage_cap =  dict_para['storage_cap']
p_ctns_comp = dict_para['ctns_comp']


"""
---------------------------- Business Rules (Excel Tab: rules)----------------------------
"""
df_rules = excel_input_raw.parse("rules")
dict_rules = df_rules.set_index('rule_name')['on/off'].to_dict()

"""
---------------------------- Locations - Stores & RDC ----------------------------
"""
##-------------str location (Excel Tab: store_loc)
df_loc_str = excel_input_raw.parse("store_loc").rename(columns = {'LATITUDE':'str_lat','LONGITUDE':'str_long'})
df_loc_str_m = df_loc_str[df_loc_str['status'] == 'Open'].iloc[:,0:8]
df_loc_str_temp = df_loc_str_m[['t_num','str_lat','str_long']]

##-------------dc location - RDC (Excel Tab: dc_loc)
df_loc_dc = excel_input_raw.parse("dc_loc").rename(columns = {'LATITUDE':'dc_lat','LONGITUDE':'dc_long'})
##-------------create dummy region rdc location table as a placeholder
##-------------this dummy location won't be used in most of cases
df_loc_region = df_loc_dc[df_loc_dc['FacilityType']!= 'FC'].groupby(['Region'], as_index = False).agg({'dc_lat': 'mean', 'dc_long': 'mean'})
df_loc_region['dc_name'] = 'Dummy_Region_' + df_loc_region.Region.astype(str)
df_loc_region = df_loc_region.assign(dc_zip3 = 'zip3_NA')
df_loc_dc_temp = pd.concat([df_loc_dc[['dc_name','dc_lat','dc_long','dc_zip3']], 
                            df_loc_region[['dc_name','dc_lat','dc_long','dc_zip3']]], ignore_index = True)

##-------------Calculate the distance matrix of every possible OD pairs
df_loc_str_temp = df_loc_str_temp.assign(key = 'join_key')
df_loc_dc_temp = df_loc_dc_temp.assign(key = 'join_key')
df_dist_str_rdc = pd.merge(df_loc_str_temp, df_loc_dc_temp, on='key').drop('key',axis=1)
df_dist_str_rdc['dist_haversine'] = haversine_dist(df_dist_str_rdc['str_lat'].values, df_dist_str_rdc['str_long'].values, 
                                                   df_dist_str_rdc['dc_lat'].values, df_dist_str_rdc['dc_long'].values)
del df_loc_str_temp, df_loc_dc_temp
    
"""
---------------------------- RDC Capacity (Excel Tab: rdc_cap)-------------------------
"""
df_cap_rdc = excel_input_raw.parse("rdc_cap")
df_cap_rdc = df_cap_rdc[df_cap_rdc['year'] == p_year]

# Please Do not use dummy node if possible!!! 
# We could add dummy RDC within each region for almost unlimited capacity and assign high penalty cost of it
#-------------add rdc dummy by Region
#list_region = df_cap_rdc.region.unique().tolist()
#for i in list_region:
#    df_cap_rdc = df_cap_rdc.append({'name':'Dummy_Region_'+ str(i), 'building_id':10000+i, 'cap_cube':5000000, 'cap_th': 5000000, 
#                                    'dock_door':1000, 'type':'Dummy_RDC', 'year': p_year, 'region': i}, ignore_index=True)

"""     
---------------------------- Fixed RDC Store (Excel Tab: rdc_str_fixed)---------------------------------------
"""
df_rdc_str = excel_input_raw.parse("rdc_str_fixed")
df_rdc_str = df_rdc_str[df_rdc_str["Included"]==1]
df_rdc_str = df_rdc_str[df_rdc_str['year'] == p_year]

"""
-----------------------------Ib and OB CPC (Excel Tab: CPC) --------------------------------------------------
"""
df_CPC = excel_input_raw.parse("CPC")
dict_CPC_IB = df_CPC.set_index(['dc_name']).IB_CPC.to_dict()
dict_CPC_OB = df_CPC.set_index(['dc_name']).OB_CPC.to_dict()
dict_CPC_Total = df_CPC.set_index(['dc_name']).DC_CPC.to_dict()

"""
---------------------------- Cost per mile (Excel Tab: middle_mile_cpm)---------------------------------------------------
"""
df_middle_cpm = excel_input_raw.parse("middle_mile_cpm")
df_dist_str_rdc['Band'] = df_dist_str_rdc['dist_haversine']
df_dist_str_rdc.loc[(df_dist_str_rdc['dist_haversine'] < 75), ['Band']] = 0
df_dist_str_rdc.loc[(df_dist_str_rdc['dist_haversine'] >= 75) & (df_dist_str_rdc['dist_haversine'] < 150), ['Band']] = 75
df_dist_str_rdc.loc[(df_dist_str_rdc['dist_haversine'] >= 150) & (df_dist_str_rdc['dist_haversine'] < 225), ['Band']] = 150
df_dist_str_rdc.loc[(df_dist_str_rdc['dist_haversine'] >= 225) & (df_dist_str_rdc['dist_haversine'] < 300), ['Band']] = 225
df_dist_str_rdc.loc[(df_dist_str_rdc['dist_haversine'] >= 300) & (df_dist_str_rdc['dist_haversine'] < 375), ['Band']] = 300
df_dist_str_rdc.loc[(df_dist_str_rdc['dist_haversine'] >= 375) & (df_dist_str_rdc['dist_haversine'] < 450), ['Band']] = 375
df_dist_str_rdc.loc[(df_dist_str_rdc['dist_haversine'] >= 450), ['Band']] = 450
df_dist_str_rdc = pd.merge(df_dist_str_rdc, df_middle_cpm, how = 'left').dropna()
df_dist_str_rdc['cost_per_truck'] = df_dist_str_rdc['dist_haversine']*df_dist_str_rdc['CPM']




""" Main model part 2
### Create python objects for Gurobi optimization model
### Mainly uses list and dictionary here for optmization model
"""
#--------------------------- dist & total cost per truck ---------------------------
dict_dist_str_rdc = df_dist_str_rdc.set_index(['t_num', 'dc_name']).dist_haversine.to_dict()
dict_cost_per_truck = df_dist_str_rdc.set_index(['t_num', 'dc_name']).cost_per_truck.to_dict()

#---------------------------- List of strs ----------------------------        
df_str = excel_input_raw.parse("Strs")
df_str = df_str[df_str['Year'] == p_year]
list_strs = df_str.STRs.unique().tolist()
list_udc= df_str.Current_Align_UDC.unique().tolist()
list_strs_new = df_str[df_str['New_STR']==1].STRs.unique().tolist()
dict_str_cube = df_str.set_index(['STRs']).Cube.to_dict()
dict_str_Ctns = df_str.set_index(['STRs']).Str_Ctns.to_dict()
dict_str_truckload = df_str.set_index(['STRs']).Truck_Load.to_dict()

#--------------------------- rdc ---------------------------    
list_rdc = df_cap_rdc[df_cap_rdc['name']!='DC_PERTH_NJ'].name.unique().tolist()  
dict_rdc_cube = df_cap_rdc.set_index('name')['cap_cube'].to_dict()
dict_rdc_th = df_cap_rdc.set_index('name')['cap_th'].to_dict()
dict_rdc_ctns = df_cap_rdc.set_index('name')['ctns_projected'].to_dict()
dict_rdc_ctns_min = df_cap_rdc.set_index('name')['ctns_min'].to_dict()

#-------updated dock door numbers
df_cap_rdc['dock_door'] = df_cap_rdc['dock_door'] - p_dock_spare
dict_rdc_dock = df_cap_rdc.set_index('name')['dock_door'].to_dict()

#--------------------------- fixed RDC stores ---------------------------
dict_rdc_str = df_rdc_str[df_rdc_str['t_num'].isin(list_strs) &  df_rdc_str['rdc'].isin(list_rdc)].set_index(['t_num', 'rdc'])['building_id'].to_dict()

#--------------------------- Penalty dataframe --------------------------
df_str_temp = pd.DataFrame(list_strs, columns=['STRs']).assign(key = 'join_key')
df_rdc_temp = pd.DataFrame(list_rdc, columns=['RDC']).assign(key = 'join_key')

df_str_rdc = pd.merge(df_str_temp, df_rdc_temp, on='key').drop('key',axis=1)
df_str_rdc = pd.merge(df_str_rdc, df_str, left_on = ['STRs','RDC'], right_on = ['STRs','Current_Align_RDC'], how = 'left')
df_str_rdc = df_str_rdc.assign(rdcpenalty = 1)
df_str_rdc.loc[(df_str_rdc.Year.notnull()) | (df_str_rdc.STRs.isin(list_strs_new)), ['rdcpenalty']] = 0
df_rdc_udc = df_str_rdc[['Current_Align_RDC','Current_Align_UDC']].dropna().drop_duplicates()
df_rdc_udc.columns = ['RDC','UDC']
df_str_rdc = pd.merge(df_str_rdc[['STRs','RDC','Year','rdcpenalty']], df_rdc_udc, how = 'left')
df_str_rdc = pd.merge(df_str_rdc, df_str, left_on = ['STRs'], right_on = ['STRs'], how = 'left')
df_str_rdc = df_str_rdc.assign(udcpenalty = 1)
df_str_rdc.loc[(df_str_rdc.UDC == df_str_rdc.Current_Align_UDC) | (df_str_rdc.STRs.isin(list_strs_new)), ['udcpenalty']] = 0
#df_str_rdc.to_csv('test.csv')
df_str_rdc = df_str_rdc[['STRs', 'RDC', 'UDC', 'rdcpenalty', 'udcpenalty', 'New_STR', 'Current_Align_RDC', 'Current_Align_UDC']]

#--------------------------- Use the Penalty dataframe to create  --------------------------
dict_rdc_penalty = df_str_rdc.set_index(['STRs', 'RDC']).rdcpenalty.to_dict()
dict_udc_penalty = df_str_rdc.set_index(['STRs', 'RDC']).udcpenalty.to_dict()

#--------------------------- get the current store mileage --------------------------
df_str_rdc = pd.merge(df_str_rdc, df_dist_str_rdc[['t_num', 'dc_name', 'dist_haversine']], left_on = ['STRs','RDC'], right_on = ['t_num', 'dc_name'], how = 'left')
df_str_rdc_current = df_str_rdc[df_str_rdc.RDC == df_str_rdc.Current_Align_RDC][['STRs','dist_haversine']]
df_str_rdc_current.columns = ['STRs','current_dist']
df_str_rdc = pd.merge(df_str_rdc, df_str_rdc_current, how='left')
#df_str_rdc.to_csv('temp1.csv')
### create a new dataframe for all the decision variable combinations
df_str_rdc_dvar = df_str_rdc.assign(removeddvar = 0)

""" Main model part 3: Create the right level of decision variables based on current rule setting of the model
### Implementing two rules here
"""
if dict_rules['Mileage_limit'] == 1:
####When the store is aligned to current RDC, which is less than 250 miles, don't allow it gets re-aligned to 250 + X miles
    df_str_rdc_dvar = df_str_rdc.assign(removeddvar = 0)
    df_str_rdc_dvar.loc[(df_str_rdc_dvar.current_dist <= dict_para['mileage_limit'] ) & (
                        df_str_rdc_dvar.dist_haversine >= dict_para['mileage_limit'] +  dict_para['mileage_limit_add'] ), ['removeddvar']] = 1

# if dict_rules['mileage_threshold'] == 1:    
#     ####If the stores's curret aligned RDC is more then 250 miles, realign to another RDC which is less or equal to 250 miles.
#     df_str_rdc_dvar.loc[(df_str_rdc_dvar.dist_haversine > dict_para['Mileage_threshold'] ), ['removeddvar']] = 1

if dict_rules['Mileage_add'] == 1:    
    ####Don't re-align to another RDC which is more than X miles than the current one
    df_str_rdc_dvar.loc[(df_str_rdc_dvar.dist_haversine > df_str_rdc_dvar.current_dist +  dict_para['mileage_comp'] ), ['removeddvar']] = 1
    

#df_str_rdc_dvar.to_csv('temp2.csv')
####setting the current dictionary for the possible RDC to Store combinations
dict_dvar_str_rdc = df_str_rdc_dvar[df_str_rdc_dvar['removeddvar'] == 0].set_index(['STRs', 'RDC']).removeddvar.to_dict()

""" Main model part 4: Optimization
### Build the optimization model
"""
##################----------------Now starts OPtimization --------------------
#-----------------Create model and decision variables------------------------------
#mod = Model("Onmi_assign")

mod = plp.LpProblem(name="Omni_assign",sense=plp.LpMinimize)

print ('Creating models and variables\n')

#Variable- if store s is aligned to RDC r or not, binary

#Dvar_X_str_rdc  = {(i,j):plp.LpVariable(cat=plp.LpBinary, name="x_{0}_{1}".format(i,j)) for i,j in dict_dvar_str_rdc}

#Dvar_X_str_rdc = mod.addVars(dict_dvar_str_rdc, vtype = GRB.BINARY, name = "binary_assign_str_rdc")

Dvar_X_str_rdc = plp.LpVariable.dicts("binary_assign_str_rdc", dict_dvar_str_rdc ,cat="Binary")

#Initialize the cost objectives
Exp_cost_mm = 0
Exp_CPC = 0
Exp_penalty_rdc = 0
Exp_penalty_udc = 0
# expression of cost in middle mile, processing cost of ctns, realignment penalty
if dict_rules['Optimize Middle Mile Transportation Cost'] == 1:  
    Exp_cost_mm = plp.lpSum(dict_cost_per_truck[s,r]*dict_str_truckload[s]*Dvar_X_str_rdc[s,r] for s,r in dict_dvar_str_rdc)
if dict_rules['Optimize Ctns Processing Cost'] == 1:
    Exp_CPC = plp.lpSum( (dict_CPC_IB[r]+ dict_CPC_OB[r] + dict_CPC_Total[r])*dict_str_Ctns[s]*Dvar_X_str_rdc[s,r] for s,r in dict_dvar_str_rdc)
if dict_rules['Optimize ReAlignment Penalty'] == 1:
    Exp_penalty_rdc = p_RDC_penalty*plp.lpSum (dict_rdc_penalty[s,r]*Dvar_X_str_rdc[s,r] for s,r in dict_dvar_str_rdc)
    Exp_penalty_udc = p_UDC_penalty*plp.lpSum (dict_udc_penalty[s,r]*Dvar_X_str_rdc[s,r] for s,r in dict_dvar_str_rdc)
# objective expression
print ('Creating Objective Functions!')
obj = Exp_cost_mm + Exp_CPC + Exp_penalty_rdc + Exp_penalty_udc 

mod.setObjective(obj)

print ('Default Constraint - any given store can only be assigned to one rdc')
# any given store can only be assigned to one rdc

for s in list_strs:
    mod += (
        plp.lpSum(Dvar_X_str_rdc[s,r] for r in list_rdc if (s,r) in dict_dvar_str_rdc) == 1.0,
        "store_to_1_rdc_%s" % s,
        )

#mod.addConstrs( (quicksum(Dvar_X_str_rdc[s,r] for r in list_rdc if (s,r) in dict_dvar_str_rdc) == 1.0 for s in list_strs ), "store_to_1_rdc")

if dict_rules['Fixed RDC Store'] == 1:
    print ('Constraint - Fixing Current RDC Store Alignment')
    for s,r in dict_rdc_str:
        mod += (
            Dvar_X_str_rdc[s,r] == 1.0,
            "fixed_rdc_store_%s" % s,
            )
           
# mod.addConstrs( (Dvar_X_str_rdc[s,r] == 1 for s,r in dict_rdc_str ), "fixed_rdc_store")
  
  
if dict_rules['RDC dock doors'] == 1:
    print ('Constraint - RDC dock doors')
    for r in list_rdc:
        mod += (
            plp.lpSum(Dvar_X_str_rdc[s,r] for s in list_strs if (s,r) in dict_dvar_str_rdc) <= dict_rdc_dock[r],
            "rdc_dock_door_%s" % r,
            )
 
#mod.addConstrs( (quicksum(Dvar_X_str_rdc[s,r] for s in list_strs if (s,r) in dict_dvar_str_rdc) <= dict_rdc_dock[r] for r in list_rdc), "rdc_dock_door")    
    
if dict_rules['RDC cube capacity'] == 1:
    print ('Constraint - RDC cube')
    for r in list_rdc:
        mod += (
            plp.lpSum(dict_str_cube[s]*Dvar_X_str_rdc[s,r] for s in list_strs if (s,r) in dict_dvar_str_rdc) <= dict_rdc_cube[r]*p_storage_cap,
            "rdc_cube_constraint_%s" % r,
            )
 
#mod.addConstrs( (quicksum(dict_str_cube[s]*Dvar_X_str_rdc[s,r] for s in list_strs if (s,r) in dict_dvar_str_rdc) <= dict_rdc_cube[r]*p_storage_cap for r in list_rdc), "rdc_constraint")
    
if dict_rules['RDC_cnts_max'] == 1:
    print ('Constraint - RDC cnts max')
    for r in list_rdc:
        mod += (
            plp.lpSum(dict_str_Ctns[s]*Dvar_X_str_rdc[s,r] for s in list_strs if (s,r) in dict_dvar_str_rdc) <= dict_rdc_ctns[r] + p_ctns_comp,
            "rdc_ctns_ub_%s" % r,
            )

#mod.addConstrs( (quicksum(dict_str_Ctns[s]*Dvar_X_str_rdc[s,r] for s in list_strs if (s,r) in dict_dvar_str_rdc) <= dict_rdc_ctns[r] + p_ctns_comp for r in list_rdc), "rdc_ctns_ub")

if dict_rules['RDC_cnts_min'] == 1:
    print ('Constraint - RDC cnts minimal')
    for r in list_rdc:
        mod += (
            plp.lpSum(dict_str_Ctns[s]*Dvar_X_str_rdc[s,r] for s in list_strs if (s,r) in dict_dvar_str_rdc) >= dict_rdc_ctns_min[r],
            "rdc_ctns_lb_%s" % r,
            )
    
#mod.addConstrs( (quicksum(dict_str_Ctns[s]*Dvar_X_str_rdc[s,r] for s in list_strs if (s,r) in dict_dvar_str_rdc) >= dict_rdc_ctns_min[r] for r in list_rdc), "rdc_ctns_lb")

# aarohi code - start 
# if dict_para['Mileage_threshold'] == 1:
#     print ('Constraint - RDC mileage threshold')
#     mod.addConstrs( df_str_rdc_dvar.dist_haversine > df_str_rdc_dvar.current_dist +  dict_para['mileage_comp']  (Dvar_X_str_rdc[s,r] for s in list_strs if (s,r) in dict_dvar_str_rdc)  , "rdc_ctns_lb")
# aarohi code - end
    
#mod.update();

mod.writeLP('dff.txt')
    
solver = plp.getSolver('PULP_CBC_CMD',mip=True)
#solver.solverModel.parameters.timelimit.set(60)
#mod.Params.MIPGap = dict_para['optimization_gap']
#mod.Params.BarHomogeneous = 1
#mod.Params.timelimit = dict_para['solve_time_limit']*60
mod.solve()

print("Status:", plp.LpStatus[mod.status])

""" Main model part 4: Output
### Build the optimization model
"""


opt_df = pd.DataFrame.from_dict(Dvar_X_str_rdc, orient="index", columns = ["variable_object"])
opt_df.index = pd.MultiIndex.from_tuples(opt_df.index, names=["column_i", "column_j"])
opt_df.reset_index(inplace=True)

opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.varValue)

opt_df.drop(columns=["variable_object"], inplace=True)

opt_df.to_csv(os.path.join(outputDataDir, s_file + "_Scenario_plp" + str(datetime.datetime.now().strftime("%Y-%m-%d")) + "_" +
                                  str(int(dict_para['year']))+".csv"), index = False)


# cols= ['t-dc', 'assign']

# df2 = pd.DataFrame(columns=cols)

# for v in mod.variables():
#     df2 = df2.append({'t-dc': v.name, 'assign':v.varValue},ignore_index=True)

#strrdc = mod.getAttr('X', Dvar_X_str_rdc)
# df_output_srdc = pd.Series(strrdc).reset_index()
# df_output_srdc.columns = ['t_num', 'dc_name', 'assign']
# df_output_srdc = df_output_srdc[df_output_srdc['assign'] > 0.01]
    
# df_output_srdc = pd.merge(df_output_srdc, df_dist_str_rdc, how = 'left')
# df_output_srdc = pd.merge(df_output_srdc, df_cap_rdc[['name','building_id']], left_on=['dc_name'], right_on = ['name'], how = 'left')
# df_output_srdc = pd.merge(df_output_srdc, df_str, left_on = ['t_num'], right_on = ['STRs'], how = 'left' )
# df_output_srdc = pd.merge(df_output_srdc, df_CPC, how = 'left')
# df_output_srdc['Cost_Middle_Mile'] = df_output_srdc['cost_per_truck']*df_output_srdc['Truck_Load']
# df_output_srdc['Cost_IB_ctns'] = df_output_srdc['Str_Ctns']* df_output_srdc['IB_CPC']
# df_output_srdc['Cost_OB_ctns'] = df_output_srdc['Str_Ctns']* df_output_srdc['OB_CPC']
# df_output_srdc['Cost_RDC_ctns'] = df_output_srdc['Str_Ctns']* df_output_srdc['DC_CPC']
    
# df_output_srdc = pd.merge(df_output_srdc, df_cap_rdc[['name', 'dock_door', 'cap_cube']], left_on=['dc_name'], right_on=['name'], how = 'left' )
# columnsTitles = ['Year','t_num', 'New_STR', 'RDC', 'dc_name', 'str_lat', 'str_long', 'dc_lat', 'dc_long', 
#                   'dist_haversine', 'Band',  'CPM', 'cost_per_truck', 'Truck_Load', 'Cost_Middle_Mile',  
#                   'Str_Ctns', 'IB_CPC','Cost_IB_ctns', 'OB_CPC','Cost_OB_ctns','Cost_RDC_ctns','dock_door','Cube','cap_cube', 'Current_Align_RDC', 'Current_Align_UDC']
# df_output_srdc = df_output_srdc[columnsTitles]

# df_output_srdc = pd.merge(df_output_srdc, df_str_rdc[['STRs','RDC','UDC','rdcpenalty','udcpenalty']], 
#                           left_on = ['t_num','dc_name'], right_on = ['STRs', 'RDC'], how = 'left')
# df_output_srdc['rdcpenalty_cost'] = df_output_srdc['rdcpenalty']*p_RDC_penalty
# df_output_srdc['udcpenalty_cost'] = df_output_srdc['udcpenalty']*p_UDC_penalty

# df_output_srdc = df_output_srdc.drop(columns=['STRs', 'RDC_y'])
# df_output_srdc_change = df_output_srdc[df_output_srdc['dc_name']!=df_output_srdc['Current_Align_RDC']][['t_num',
#                                       'New_STR','RDC_x','dc_name','Current_Align_RDC','Str_Ctns','dist_haversine','Cost_Middle_Mile']]
# df_output_srdc_change = pd.merge(df_output_srdc_change, df_cap_rdc[['name','building_id']], left_on = 'Current_Align_RDC', right_on = 'name', how = 'left')
# df_output_srdc_change = df_output_srdc_change.drop(columns=['name'])
# df_output_srdc_change.columns = ['t_num','New_STR','Scenario_RDC_id','Scenario_RDC_name','Current_RDC_Name','Str_Ctns',
#                                   'dist_haversine', 'Cost_Middle_Mile', 'Current_RDC_id']
# #    df_output_srdc_change = pd.merge(df_output_srdc_change, df_str[['STRs','Str_Ctns']], left_on = 't_num', right_on = 'STRs', how = 'left')
# df_output_srdc_change = pd.merge(df_output_srdc_change, df_str_rdc[['STRs', 'RDC', 'dist_haversine', 'current_dist']], left_on = ['t_num','Scenario_RDC_name'], 
#                                   right_on = ['STRs','RDC'], how = 'left')
# #    df_str_rdc[['STRs', 'RDC', 'dist_haversine', 'current_dist']]
# df_output_srdc_change = df_output_srdc_change.drop(columns=['STRs', 'RDC', 'dist_haversine_y'])


#df_output_srdc.to_csv(os.path.join(outputDataDir, s_file + "_Scenario_plp" + str(datetime.datetime.now().strftime("%Y-%m-%d")) + "_" +
 #                                 str(int(dict_para['year']))+".csv"), index = False)
# #     df_output_srdc_change.to_csv(os.path.join(outputDataDir, s_file + "_Scenario_Realignment_" + str(datetime.datetime.now().strftime("%Y-%m-%d")) + "_" +
# #                                               str(int(dict_para['year']))+".csv"), index = False)


