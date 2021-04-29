
import json
import pandas as pd 
from  matching_model import *
from  matching_utils import *
import networkx as nx
import pandas as pd


with open('rs/test_input.json') as myfile:
    data = json.load(myfile)

# print(type(data))
# print(data)
# print(data["mode"])
if data["mode"] == 'solution_general':
    data = data['input_data']
    data = json.dumps(data, indent = 4)

    graph = nx.read_gpickle("rs/full_road_netw_thess.gpickle")
    graph_sec = nx.read_gpickle("rs/full_road_render.gpickle")

    data = pd.read_json(data, orient = 'index')
    data = data.fillna(-1)
    # # convert data into dataframe
    clients_list, drivers_list, dest_list, drivers, paths, clients, dests, dep_arr,\
        time_lims_cl, time_lims_dr, capacity = decompose_data(data)

    dist_matrix, points_matrix_lat,\
        points_matrix_lon, f_path_dist, f_full_path_len, new_paths_list = calc_distance_matrix(drivers_list,\
        dest_list,clients_list, graph, graph_sec,depArr = dep_arr)            

    solved_df = solve_carpool_matching(clients_list, clients, drivers_list, drivers, paths, dist_matrix,\
                                    points_matrix_lat, points_matrix_lon,f_path_dist, f_full_path_len,\
                                        time_lims_cl, time_lims_dr,depArr= dep_arr,capacity_list= capacity)


    result = {}

    dm = {dr : {pth : {cl: dist_matrix[i][j][k] for k, cl in enumerate(clients)}\
                for j, pth in enumerate(paths)} for i, dr in enumerate(drivers)}
    result['distance_matrix'] = dm

    p_lat = {dr : {pth : {cl: points_matrix_lat[i][j][k] for k, cl in enumerate(clients)}\
                for j, pth in enumerate(paths)} for i, dr in enumerate(drivers)}
    result['point_lat'] = p_lat

    p_lon = {dr : {pth : {cl: points_matrix_lon[i][j][k] for k, cl in enumerate(clients)}\
                for j, pth in enumerate(paths)} for i, dr in enumerate(drivers)}
    result['point_lon'] = p_lon

    dfs = {dr : {pth : {cl: f_path_dist[i][j][k] for k, cl in enumerate(clients)}\
                for j, pth in enumerate(paths)} for i, dr in enumerate(drivers)}
    result['dist_from_start'] = dfs
    
            
            
            
    # data = data.to_json(orient = 'index', indent = 4)
    data = data.to_dict(orient = 'index')
    result['input_data'] = data

    if isinstance(solved_df, pd.DataFrame):
        solved_df = solved_df.fillna(-1)
        final_df = get_json_final(solved_df)
        final_df = json.loads(final_df)
        result['output'] = final_df       
    else:
        result['output'] = solved_df
        
    result["mode"] = 'solution_general'
    result["cancel_drivers"] = []


with open('rs/output_test.json', 'w') as outfile:
    json.dump(result, outfile, indent= 2)