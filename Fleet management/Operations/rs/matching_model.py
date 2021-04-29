"""
Momentum : Fleet Management : Operational : Ridesharing
@author: zisis
"""

from pulp import *
import pandas as pd 
import numpy as np 
import itertools
import math
from collections import defaultdict
from  matching_utils import *
import networkx as nx 
# import folium
import json

def decompose_data(df):
    clients_list = []
    drivers_list = []
    dest_list = []
    
    drivers = []
    paths = ['pt1', 'pt2', 'pt3']
    clients = []
    dests= []

    time_lims_cl = []
    time_lims_dr = []
    
    capacity = []
    dep_arr = 0
    for idx, row in df.iterrows():
        if row.role == 0 :
            clients_list.append((row.lat, row.lon))
            clients.append(idx)
            time_lims_cl.append((row.time_dep, row.time_arr))

        elif row.role == 1:
            drivers_list.append((row.lat, row.lon))
            drivers.append(idx)
            time_lims_dr.append((row.time_dep, row.time_arr))
            capacity.append(row.capacity)
        else: 
            dest_list.append((row.lat, row.lon))
            dests.append(idx)
            dep_arr  = row.dep_arr

    return clients_list, drivers_list, dest_list, drivers, paths, clients, dests,dep_arr,\
        time_lims_cl, time_lims_dr, capacity


            
            
            
def calc_distance_matrix(drivers_list, dest_list, clients_list,graph,graph_sec,depArr = 0):    
    od_list = drivers_list + dest_list
        
    origin_dest = from_points_to_nodes(graph, od_list)    
    
    list_of_all_paths = []
    if depArr  == 0:
        for i in range(len(origin_dest)-1):
            all_paths = get_3_shortest_paths(graph_sec, origin_dest[i], origin_dest[-1])
            list_of_all_paths.append(all_paths)
    else:
        for i in range(len(origin_dest)-1):
            all_paths = get_3_shortest_paths_departure(graph_sec, origin_dest[-1], origin_dest[i])
            list_of_all_paths.append(all_paths)

    # new_paths = [node_list_to_path(graph, nl) for nl in all_paths]
    new_paths_list = [[node_list_to_path(graph, nl) for nl in all_paths] for all_paths in list_of_all_paths]
    
    
    # final_dist_matrix = get_distance_matrix(clients, new_paths_list)
    
    
    # dist_matrix drivers by clients ........
    
    dist_matrix, points_matrix_lat , points_matrix_lon,f_path_dist,\
         f_full_path_len = get_distance_matrix(graph,clients_list, new_paths_list, list_of_all_paths)
    
    return dist_matrix, points_matrix_lat, points_matrix_lon, f_path_dist, f_full_path_len, new_paths_list

#menbers of the system..................

def solve_carpool_matching(clients_list, clients,drivers_list, drivers,
                           paths, dist_matrix, points_matrix_lat, points_matrix_lon, \
                               f_path_dist,f_full_path_len, time_lims_cl, time_lims_dr,\
                                    depArr, capacity_list):
    
    clients_dict = {key : val for key, val in zip(clients,clients_list)}
    drivers_dict = {key : val for key, val in zip(drivers,drivers_list)}
    
    clients_time_dict = {key : val for key, val in zip(clients,time_lims_cl)}
    drivers_time_dict = {key : val for key, val in zip(drivers,time_lims_dr)}

    penalty = [[20 for i in range(len(clients))] for j in range(len(drivers))]
    
    drivers_capacity = { key: val for key, val in  zip(drivers, capacity_list)}

    #cost dictionary 
    costs = makeDict([drivers,paths, clients], dist_matrix,0)

    #up to client time dictionary..
    ut_client_time = makeDict([drivers, paths, clients], f_path_dist,0)

    #up to client time dictionary..
    # ut_client_dist= makeDict([drivers, paths, clients], f_path_dist*33000/60,0)
    # print(ut_client_dist)
    #total_path time dictionary..
    tp_len = makeDict([drivers, paths], f_full_path_len,0)

    penalty_cost = makeDict([drivers, clients], penalty,0)
    #set the problem object...................
    
    prob = LpProblem("Carpool Matching Problem",LpMinimize)
    
    
    #variables......................................
    
    Routes = [(d, p, cl) for d in drivers for p in paths for cl in clients]
    
    Routes_match = [(d, cl) for d in drivers  for cl in clients]
    
    Paths_match = [(d, cl) for d in drivers  for cl in paths]
    # driver i goes from path k and pick-up client j ..............
    
    vars = LpVariable.dicts("Route",(drivers, paths, clients),0,1,LpInteger)
    
    # driver i takes the path k.......................
    vars_2 = LpVariable.dicts("One_path",(drivers, paths),0,1,LpInteger)
    
    # lpSum([vars[d][p][cl]*costs[d][p][cl] for (d, p, cl) in Routes]) 
    # prob +=  lpSum([penalty_cost[d][cl]*(- lpSum(vars[d][p][cl] for p in paths)) for (d, cl) in Routes_match])\
    #     + lpSum([tp_len[d][p]*vars_2[d][p] for d,p in  Paths_match]), "Sum_of_Transporting_Costs"
    prob +=  lpSum([penalty_cost[d][cl]*(- lpSum(vars[d][p][cl] for p in paths)) for (d, cl) in Routes_match])\
        + lpSum([vars[d][p][cl]*costs[d][p][cl]*0.2 for d, p, cl in Routes])\
            + lpSum([tp_len[d][p]*vars_2[d][p]*3 for d,p in  Paths_match]), "Sum_of_Transporting_Costs"
    # prob += lpSum([penalty_cost[d][cl]*(1 - lpSum(vars[d][p][cl] for p in paths)) for (d, cl) in Routes_match]), "Sum_of_Penalty_Costs"
    
    #add contraints
    
    for d in drivers:
        for p in paths:
            prob += lpSum([vars[d][p][cl] for cl in clients]) <= drivers_capacity[d]*vars_2[d][p]
    
    for d in drivers:
        prob += lpSum([vars_2[d][p] for p in paths]) == 1
    
    for cl in clients:
        prob += lpSum([vars[d][p][cl] for d in drivers for p in paths]) <= 1
     
    for cl in clients:
        prob += lpSum([vars[d][p][cl]*costs[d][p][cl] for d in drivers for p in paths]) <= 1

    # for p in paths:
    #     prob += lpSum([vars[d][p][cl] for d in drivers for cl in clients ]) == 1 
    # for d in drivers:
    #     for p in paths:
    #         for cl in clients:
    #             print(ut_client_time[d][p][cl])
    #             prob += ut_client_time[d][p][cl]*vars[d][p][cl] >=  clients_time_dict[cl][0] - drivers_time_dict[d][0] 
    if  depArr == 0:
        for d in drivers:
            for p in paths:
                for cl in clients:
                    # print(tp_len[d][p])
                    prob += tp_len[d][p]*vars[d][p][cl] <= abs(min(max(drivers_time_dict[d][0],\
                        clients_time_dict[cl][0] - np.round(ut_client_time[d][p][cl],0) + 2) - \
                        min(drivers_time_dict[d][1],clients_time_dict[cl][1]),0))
    else: 
        for d in drivers:
            for p in paths:
                for cl in clients:
                    # print(tp_len[d][p])
                    if drivers_time_dict[d][1] <= clients_time_dict[cl][1]:
                        prob += tp_len[d][p]*vars[d][p][cl] <= abs(min(max(drivers_time_dict[d][0],\
                            clients_time_dict[cl][0]) - \
                            min(drivers_time_dict[d][1],clients_time_dict[cl][1]),0))
                    else:
                        prob += np.round(ut_client_time[d][p][cl],0)*vars[d][p][cl] <= abs(min(max(drivers_time_dict[d][0],\
                            clients_time_dict[cl][0]) - \
                            min(drivers_time_dict[d][1],clients_time_dict[cl][1]),0))


    # The problem data is written to an .lp file
    prob.writeLP("carpool_sol.lp")
    

    # The problem is solved using PuLP's choice of Solver
    prob.solve()
    
    
    
    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])
        
    
    
    # for v in prob.variables():
    #     if v.varValue != 0:
            
    #         print(v.name, "=", v.varValue)
    
    
    #pairnw ta routes
    res = []
    if LpStatus[prob.status] == 'Optimal':
        for v in prob.variables():
                if v.varValue != 0  and v.name.split('_')[0] == 'Route':
                    scs = v.name.split('_')[1:]
                    res.append(scs)
    else:
        return 'Cant Foound Optimal Solution'

    if len(res) == 0:
        return 'No one matched'
    print(res)
    assigments_df  = pd.DataFrame(np.array(res), columns= ['driver', 'path', 'client'])
    
    point_temp_lat = []
    point_temp_lon = []
    dist_from_start = []
    for _, row in assigments_df.iterrows():
        temp_dr = drivers.index(row.driver)
        temp_path = paths.index(row.path)
        temp_cl = clients.index(row.client)
    
        point_temp_lat.append(points_matrix_lat[temp_dr][temp_path][temp_cl])
        point_temp_lon.append(points_matrix_lon[temp_dr][temp_path][temp_cl])
        dist_from_start.append(f_path_dist[temp_dr][temp_path][temp_cl])
        # print(dist_from_start)
    assigments_df['point_on_road'] = [list(i) for i in zip(point_temp_lat,point_temp_lon,dist_from_start)]


    return assigments_df


def get_json_final(df):
    groups_df = df.groupby(['driver', 'path'])\
        .apply(lambda x: {'{}'.format(i): j for i,j in \
                          zip(list(x['client']), list(x['point_on_road']))}).reset_index(name='assigments')

    for i in range(len(groups_df)):
        a = groups_df.assigments[i]
        b = sorted(a.items(), key=lambda x: x[1][2]) 
        groups_df.assigments[i] = dict(b)

    groups_df.index = groups_df.driver
    groups_df = groups_df.drop('driver',1)
    
    return groups_df.to_json(orient = 'index', indent = 4)