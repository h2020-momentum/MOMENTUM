from pulp import * 
import numpy as np 
import pandas as pd 
import json 
from sklearn.metrics.pairwise import haversine_distances
from math import radians

def take_dist_mat(df):
    '''
    in km
    '''
    coords_temp = [[d1, d2] for d1, d2 in zip(df.lat.tolist() , df.lon.tolist())]
    coords_rad = [[radians(_) for _ in a1] for a1 in coords_temp ] 
    hav_mat_ = haversine_distances(coords_rad, coords_rad)*6371
    hav_mat_ = np.round(hav_mat_, 2)
    return hav_mat_

def take_dist_from_bus(df_cents, df_stops):
    '''
    in km
    '''
    coords_or = [[d1, d2] for d1, d2 in zip(df_cents.lat.tolist() , df_cents.lon.tolist())]
    coords_or_rad = [[radians(_) for _ in a1] for a1 in coords_or ] 

    bus_or = [[d1, d2] for d1, d2 in zip(df_stops.lat.tolist() , df_stops.lon.tolist())]
    bus_or_rad = [[radians(_) for _ in a1] for a1 in bus_or ] 

    hav_mat_or = haversine_distances(coords_or_rad, bus_or_rad)*6371000/1000
    # print(hav_mat_or.shape)
    dists_closest = np.sort(hav_mat_or, axis = 1)[:,0].tolist()

    return dists_closest

def take_dist_from_bike(df_cents, df_lanes):
    '''
    in km
    '''
    coords_or = [[d1, d2] for d1, d2 in zip(df_cents.lat.tolist() , df_cents.lon.tolist())]
    coords_or_rad = [[radians(_) for _ in a1] for a1 in coords_or ] 

    bike_or = [[d1, d2] for d1, d2 in zip(df_lanes.lat.tolist() , df_lanes.lon.tolist())]
    bike_or_rad = [[radians(_) for _ in a1] for a1 in bike_or ] 

    hav_mat_or = haversine_distances(coords_or_rad, bike_or_rad)*6371000/1000
    # print(hav_mat_or.shape)
    dists_closest = np.sort(hav_mat_or, axis = 1)[:,0].tolist()
    return dists_closest

# OBJ_MODE = 'FULL' # FULL, NBIKE, NUMSTOPS

# DICT_OF_PARAMETERS = {}

#take data and prepare them for dist matrix (the station candidates)
#... 
# # cent_data = pd.read_csv('centers_180_new.csv') 
# cent_data = pd.read_csv('cents_with_bal.csv') 

# fixed_stations = pd.read_excel('στασεις τελικο.xlsx')
# fixed_stations_list = fixed_stations.stations.tolist()

# flows_df = pd.read_csv('flows_bounds.csv')

# dist_from_lanes = pd.read_csv('dist_from_bikeLANE.csv')

# distFromLane = dist_from_lanes.dist_from_bikeLane.tolist()
# dict_of_penalties = {'inner_Dist': True, 'dist_from_lane': True, 'dist_from_bus': True, 
#     'num_of_stops': True, 'size_of_stop': True}
# dict_of_penalties_values = {'inner_Dist': 1, 'dist_from_lane': 0.3, 'dist_from_bus': 0.4,
#      'num_of_stops': 0.5, 'size_of_stop': 1}


def facility_location(centers_coordinates = None,
    bike_lanes_nodes = None, 
    bus_lanes_nodes = None,
    needs_capacity = True,
    SUB = None, SLB = None, LOWDEM = None, HIGHDEM = None,
    INNER_DENSITY = None,  density_constraint = True,
    dict_of_penalties = None, dict_of_penalties_values = None,
    fixed_station_list = [],
    maximum_num_of_stops = None,
    demand_vector = []):

    NUM_STOPS = centers_coordinates.shape[0]

    places = [ 'station_{0}'.format(i) for i in range(NUM_STOPS)]

    overall_distances = take_dist_mat(centers_coordinates)

    busLane_dist, bl_dist = np.zeros(NUM_STOPS).tolist(), np.zeros(NUM_STOPS).tolist()

    if dict_of_penalties['dist_from_lane']:

        bl_dist = take_dist_from_bike(centers_coordinates, bike_lanes_nodes)

    if dict_of_penalties['dist_from_bus']:

        busLane_dist = take_dist_from_bus(centers_coordinates, bus_lanes_nodes)

    if dict_of_penalties['size_of_stop']:
        size_penalty = [-i for i in demand_vector]

    # The cost data is made into a dictionary
    # costs = makeDict([places,places],overall_distances,0)

    # Creates the 'prob' variable to contain the problem data
    prob = LpProblem("P-Median Problem",LpMinimize)


    vars = LpVariable.dicts("assigment",(places,places),0,1,LpInteger) #Yij

    vars_1 = LpVariable.dicts("location",places,0,1,LpInteger) #Xj

    if needs_capacity: 

        vars_cap = LpVariable.dicts("capacity",places,0,None,LpInteger) #Cj

    Routes = [(w,b) for w in places for b in places]


    # The cost data is made into a dictionary
    costs = makeDict([places,places],overall_distances,0)

    #objective 
    prob += lpSum([vars[w][b]*costs[w][b]*dict_of_penalties['inner_Dist'] for (w,b) in Routes])\
        + lpSum([dict_of_penalties_values['num_of_stops']*vars_1[h]*dict_of_penalties['num_of_stops'] for h in places]) \
        + lpSum([dict_of_penalties_values['dist_from_lane']*vars_1[oneplace]*bl_dist[i]*dict_of_penalties['dist_from_lane'] for i, oneplace in enumerate(places)])\
        + lpSum([dict_of_penalties_values['dist_from_bus']*vars_1[oneplace]*busLane_dist[i]*dict_of_penalties['dist_from_bus'] for i, oneplace in enumerate(places)])\
        + lpSum([dict_of_penalties_values['size_of_stop']*vars_cap[oneplace]*size_penalty[i]*dict_of_penalties['size_of_stop'] for i, oneplace in enumerate(places)]), "Sum_of_values"



    # prob += lpSum([vars[w][b]*costs[w][b] for (w,b) in Routes] + [0.3*vars_1[h] for h in places] \
    #     + [0.2*vars_1[oneplace]*distFromLane[i] for i, oneplace in enumerate(places)]), "Sum_of_values"


    for place in places: 
        prob += lpSum(vars[place][m] for m in places) == 1

    if not maximum_num_of_stops == None:
        prob += lpSum(vars_1[m] for m in places) <= maximum_num_of_stops

    if density_constraint:

        for cstop in places: 
            for fstop in places: 
                # if i  in fixed_stations_list:
                    prob += vars[cstop][fstop]*costs[cstop][fstop] <= INNER_DENSITY


    for cstop in places: 
        for fstop in places: 
            prob += vars[cstop][fstop] <= vars_1[fstop]

    if needs_capacity:
        prob += lpSum(vars_cap[oneplace] for oneplace in  places) >= LOWDEM

        prob += lpSum(vars_cap[oneplace] for oneplace in  places) <= HIGHDEM


        for i, oneplace in enumerate(places): 

            if not SUB == None:
                prob += vars_cap[oneplace] <=  SUB*vars_1[oneplace]

            if not SLB == None:
                prob += vars_cap[oneplace] >=  SLB*vars_1[oneplace]

    if len(fixed_station_list) >0 :
        for i, oneplace in enumerate(places):
            if i in fixed_stations_list:
                prob +=  vars_1[oneplace] == 1


    prob.solve(PULP_CBC_CMD(msg=0))

    if not LpStatus[prob.status] == 'Optimal':
        return -1, -1


    # The optimised objective function value is printed to the screen    
    print("Total Cost of Transportation = ", value(prob.objective))


    final_stops= []
    final_stops_caps = []

    for v in prob.variables():
        if v.varValue != 0 and v.name.split('_')[0] == 'location':
            # print(v.name.split('_'))
            final_stops.append(int(v.name.split('_')[2]))

        if v.varValue != 0 and v.name.split('_')[0] == 'capacity':
            final_stops_caps.append(v.varValue)
            # print(v.name, "=", v.varValue)


    return final_stops, final_stops_caps







