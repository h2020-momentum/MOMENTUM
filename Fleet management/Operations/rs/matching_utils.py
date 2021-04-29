import pandas as pd 
import numpy as np 
import itertools
import math
from collections import defaultdict
import networkx as nx 
from sklearn.neighbors import KDTree
from math import radians, cos, sin, asin, sqrt


    
def get_nearest_node_zm(list_of_nodes, point_tuple,tree):
    point = np.array(point_tuple).reshape(-1,2)
    index = tree.query(point)[1][0][0]
    temp_node = list_of_nodes[index][0]
    return temp_node

def from_points_to_nodes(G,list_points):  
    list_of_nodes = list(G.nodes(data=True))
    bs_ar = np.array([(node[1]['y'] ,node[1]['x'] ) for node in list_of_nodes])
    tree = KDTree(bs_ar, leaf_size=2)  
    new_list = []
    for point in list_points:
        temp_node = get_nearest_node_zm(list_of_nodes,point,tree)
        new_list.append(temp_node)
    return new_list

    
def get_path_lenght(G, list_of_paths):  
    paths_length = []
    for one_path in  list_of_paths:
        edge_nodes = list(zip(one_path[:-1], one_path[1:]))
        temp = 0
        for u, v in edge_nodes:
            # if there are parallel edges, select the shortest in length
            temp += G.get_edge_data(u, v)[0]['length']
    
        if temp < 10:
            for u, v in edge_nodes[::-1]:
            # if there are parallel edges, select the shortest in length
                temp -= G.get_edge_data(u, v)[0]['length']

        paths_length.append(temp)
    return paths_length


def get_3_shortest_paths(G,or_node, dest_node) :
    graph_2 = G.copy()
    all_paths = []
    for i in range(4):
        if i == 0: 
            route = nx.shortest_path(graph_2,source= or_node,target=dest_node,weight = 'new_val')
            edge_nodes = list(zip(route[:-1], route[1:]))
            for j ,u in enumerate(edge_nodes):
                # if there are parallel edges, select the shortest in length
                try :
                    lanes_factor = int(graph_2.get_edge_data(u[0], u[1])[0]['lanes'])
                except:
                    lanes_factor = 1
                if j >= 0.8*len(edge_nodes):    
                    # graph_2.get_edge_data(u[0], u[1])[0]['new_val'] *= 1.3**np.exp((1/len(edge_nodes))/100/lanes_factor)
                    graph_2.get_edge_data(u[0], u[1])[0]['new_val'] *= 1.4**np.exp((len(edge_nodes)-j)/100)
        else:
            route = nx.shortest_path(graph_2,source= or_node,target=dest_node ,weight = 'new_val')
            edge_nodes = list(zip(route[:-1], route[1:]))     
              
            for j ,u in enumerate(edge_nodes):
                # if there are parallel edges, select the shortest in length
                # try :
                #     lanes_factor = int(graph_2.get_edge_data(u[0], u[1])[0]['lanes'])
                # except:
                #     lanes_factor = 1
                if j <= 0.8*len(edge_nodes):    
                    # graph_2.get_edge_data(u[0], u[1])[0]['new_val'] *= 1.3**np.exp((1/len(edge_nodes))/100/lanes_factor)
                    graph_2.get_edge_data(u[0], u[1])[0]['new_val'] *= 1.4**np.exp((len(edge_nodes)-j)/100)
                else:
                    graph_2.get_edge_data(u[0], u[1])[0]['new_val'] /= 1.4**np.exp((len(edge_nodes)-j)/100)
        
            all_paths.append(route)
    return all_paths

def get_3_shortest_paths_departure(G,or_node, dest_node):
    graph_2 = G.copy()
    all_paths = []
    for i in range(4):
        if i == 0: 
            route = nx.shortest_path(graph_2,source= or_node,target=dest_node,weight = 'new_val')
            edge_nodes = list(zip(route[:-1], route[1:]))
            for j ,u in enumerate(edge_nodes):
                # if there are parallel edges, select the shortest in length
                try :
                    lanes_factor = int(graph_2.get_edge_data(u[0], u[1])[0]['lanes'])
                except:
                    lanes_factor = 1
                if j >= 0.85*len(edge_nodes):    
                    # graph_2.get_edge_data(u[0], u[1])[0]['new_val'] *= 1.3**np.exp((1/len(edge_nodes))/100/lanes_factor)
                    graph_2.get_edge_data(u[0], u[1])[0]['new_val'] *= 1.2**np.exp((len(edge_nodes)-j)/100)

        else:
            route = nx.shortest_path(graph_2,source= or_node,target=dest_node ,weight = 'new_val')
            edge_nodes = list(zip(route[:-1], route[1:]))     
              
            for j ,u in enumerate(edge_nodes):
                # if there are parallel edges, select the shortest in length
                # try :
                #     lanes_factor = int(graph_2.get_edge_data(u[0], u[1])[0]['lanes'])
                # except:
                #     lanes_factor = 1
                if j <= 0.15*len(edge_nodes):
                    graph_2.get_edge_data(u[0], u[1])[0]['new_val'] /= 1.4**np.exp((len(edge_nodes)-j)/100)
                    
                elif j <= 0.85*len(edge_nodes):    
                    # graph_2.get_edge_data(u[0], u[1])[0]['new_val'] *= 1.3**np.exp((1/len(edge_nodes))/100/lanes_factor)
                    graph_2.get_edge_data(u[0], u[1])[0]['new_val'] *= 1.4**np.exp((len(edge_nodes)-j)/100)
                else:
                    graph_2.get_edge_data(u[0], u[1])[0]['new_val'] /= 1.4**np.exp((len(edge_nodes)-j)/100)
        
            all_paths.append(route)
    return all_paths

def node_list_to_path(G, node_list):
    """
    Given a list of nodes, return a list of lines that together
    follow the path
    defined by the list of nodes.
    Parameters
    ----------
    G : networkx multidigraph
    route : list
        the route as a list of nodes
    Returns
    -------
    lines : list of lines given as pairs ( (x_start, y_start), 
    (x_stop, y_stop) )
    """
    edge_nodes = list(zip(node_list[:-1], node_list[1:]))
    lines = []
    for u, v in edge_nodes:
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), 
                   key=lambda x: x['length'])
        # if it has a geometry attribute
        if 'geometry' in data:
            # add them to the list of lines to plot
            xs, ys = data['geometry'].xy
            lines.append(list(zip(ys, xs)))
        else:
            # if it doesn't have a geometry attribute,
            # then the edge is a straight line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(y1, x1), (y2, x2)]
            lines.append(line)
    return lines

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r



def tree_dicts(new_pl):    
    tree_data = {}
    for i, all_paths in enumerate(new_pl):
        for j, path in enumerate(all_paths):
            # tree_data['cord_data'] = {'path_{0}_{1}'.format(i,j) : KDTree(np.array(path).reshape(-1,2), leaf_size=2)}
            path = [link[0] for link in path]
            tree_data['path_{0}_{1}'.format(i,j)] = {'tree': KDTree(np.array(path).reshape(-1,2), leaf_size=2), 'path' :path}
    
    return tree_data

def get_dist_from_route(G, trees , client, old_pl):
    dist_list = []
    nearest_point_list_lat = []
    nearest_point_list_lon = []
    temp_path = []
    for key in trees.keys():
        # print(key)
        k = trees[key]['tree']
        index = k.query(np.array(client).reshape(-1,2))[1][0][0]
        nearest_point = trees[key]['path'][index]
        #we can use get_path_length for :index path than can easily derived from here ..
        
        dist = haversine(client[1], client[0], nearest_point[1], nearest_point[0])
        
        
        nearest_point_list_lat.append(nearest_point[0])
        nearest_point_list_lon.append(nearest_point[1])
        
        idx_dr = int(key.split('_')[1])
        idx_path = int(key.split('_')[2])
        temp_path.append(old_pl[idx_dr][idx_path][:index])

        nearest_edge = old_pl[idx_dr][idx_path][index: index +2]
        road_type = G.get_edge_data(nearest_edge[0], nearest_edge[1])[0]['highway']
        try :
            road_name = G.get_edge_data(nearest_edge[0], nearest_edge[1])[0]['name']
        except:
            road_name = 'None'


        if road_type == 'motorway' and road_name != 'Εγνατία Οδός':
            dist = 10.5

        dist_list.append(dist)
        # print(idx_dr)
        # print(idx_path)
        # print(temp_path)
        # print(temp_path)
    temp_path_length = get_path_lenght(G, temp_path)
    
    return dist_list, nearest_point_list_lat, nearest_point_list_lon, temp_path_length

# dist_1 = get_dist_from_route(tree_data, costumer_1)


def get_distance_matrix(G, cl_list,new_pl, old_pl):
    tree_data = tree_dicts(new_pl)
    temp_cl_to_route = []
    point_cl_to_route_lat = []
    point_cl_to_route_lon = []
    temp_cl_to_dist = []

    for client in cl_list:
        dist_vec, nearest_point_vec_lat ,nearest_point_vec_lon, path_dist_vec \
            = get_dist_from_route(G, tree_data, client, old_pl)

        temp_cl_to_route.append(dist_vec)
        point_cl_to_route_lat.append(nearest_point_vec_lat)
        point_cl_to_route_lon.append(nearest_point_vec_lon)
        temp_cl_to_dist.append(path_dist_vec)



    temp_full_length = [get_path_lenght(G, all_paths)  for all_paths in old_pl]

    dist_mat = np.vstack(temp_cl_to_route)
    dist_mat = dist_mat.T
    
    nearest_point_mat_lat = np.vstack(point_cl_to_route_lat)
    nearest_point_mat_lat = nearest_point_mat_lat.T
    
    nearest_point_mat_lon = np.vstack(point_cl_to_route_lon)
    nearest_point_mat_lon = nearest_point_mat_lon.T
    
    path_dist_mat = np.vstack(temp_cl_to_dist)
    path_dist_mat = path_dist_mat.T
    path_dist_mat = path_dist_mat/1000/30*60
    
    temp_full_length = np.vstack(temp_full_length)
    temp_full_length = temp_full_length.T
    temp_full_length = temp_full_length/1000/30*60

    f_dist = dist_mat.reshape(len(new_pl),3,-1)
    
    f_point_lat = nearest_point_mat_lat.reshape(len(new_pl),3,-1)
    f_point_lon = nearest_point_mat_lon.reshape(len(new_pl),3,-1)
    
    f_path_dist = path_dist_mat.reshape(len(new_pl),3,-1)

    f_full_path_len = temp_full_length.reshape(len(new_pl),-1)
    
    return f_dist, f_point_lat, f_point_lon, f_path_dist, f_full_path_len


# 