from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from beeprint import pp
from outputData import *
from inputData import *

import json
import math


#============================================================
# Function to create data dictionary that is required to solve 
# the VRP problem using or-tools. See InputData.py for a 
# description of the classes Depot, RequestList (job_list), Fleet and 
# Time_Matrix
#============================================================

def create_data_model(depot, job_list, fleet, time_matrix):
    data = {}
    data['depot'] = depot.id
    data['vehicle_capacities'] = []
    data['vehicle_ids'] = []
    data['num_vehicles'] = len(fleet)
    for i in range(len(fleet)):
        data['vehicle_capacities'].append(fleet[i].capacity)   
        data['vehicle_ids'].append(fleet[i].id)   
    data['demands'] = []
    data['demands'].append(0)
    data['time_windows'] = []
    data['time_windows'].append([depot.time_window.start, depot.time_window.end])
    data['pickups_deliveries'] = []
    index_temp = []
    index_temp.append(depot.id)
    pos = 1
    priorities = []
    for i in range(len(job_list)):
        data['demands'].append(job_list[i].demand)
        data['demands'].append(-job_list[i].demand)
        data['time_windows'].append([job_list[i].origin.time_window.start, job_list[i].origin.time_window.end])
        data['time_windows'].append([job_list[i].destination.time_window.start, job_list[i].destination.time_window.end])
        #data['pickups_deliveries'].append([job_list[i].origin.station.id, job_list[i].destination.station.id])
        data['pickups_deliveries'].append([pos, pos+1]);
        index_temp.append(job_list[i].origin.station.id) 
        index_temp.append(job_list[i].destination.station.id)
        priorities.append(job_list[i].priority)
        priorities.append(job_list[i].priority)
        pos = pos + 2 
    time_matrix_new = [[0 for x in range(len(index_temp))] for y in range(len(index_temp))] 
    for o in range(len(index_temp)):
        for d in range(len(index_temp)):
            time_matrix_new [o][d] = time_matrix[index_temp[o]][index_temp[d]]
    data['time_matrix'] = time_matrix_new
    return [data,index_temp,priorities]





#===================================================================================
# Funtion to create a more friendly output data structure from the classes 
# return by or-tools when solving VRP problems. See OutputData.py for a description
# of the content of clases Result, Route and KPIs.
#===================================================================================
def create_output_data_model(data, manager, routing, solution, real_node_indexes) -> [Result,[Route],KPIs]:

    #Checks if a solution has been obtained
    if routing.status() != routing.ROUTING_SUCCESS:
      status = "Error"
      error_message = "Solution not found. Search limit exceeded"

      result = Result(status,error_message)

      return [result,[],None]

    # Display dropped nodes.
    dropped_nodes = 'Dropped nodes:'
    dropped = []
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if solution.Value(routing.NextVar(node)) == node:
            dropped_nodes += ' {}'.format(manager.IndexToNode(node))
            id_request =  (manager.IndexToNode(node) - 1) // 2
            dropped.append(id_request)
    print(dropped_nodes)

    # A solution has been found ...

    status = "OK"
    error_message = ""

    result = Result(status,error_message)

    time_dimension = routing.GetDimensionOrDie('Time')
    total_distance = 0
    total_load = 0
    routes = []
    num_vehicles = 0

    for vehicle_iter in range(data['num_vehicles']):
        index = routing.Start(vehicle_iter)

        time_var = time_dimension.CumulVar(index)
        start_time = solution.Value(time_var)
        #If route is empty, it means that the vehicle has not been used,
        #and we skip it.
        if not routing.IsVehicleUsed(solution,vehicle_iter):
          continue

        #The vehilce is used
        num_vehicles = num_vehicles + 1
        
        route_distance = 0
        route_load = 0
        stop_index = 0
        stops = []

        while not routing.IsEnd(index):

            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            previous_index = index
            time_var = time_dimension.CumulVar(index)

            if data['demands'][node_index] > 0:
              pickup = data['demands'][node_index]
              delivery = 0
            else:
              delivery = abs(data['demands'][node_index])
              pickup = 0

            id_request =  (node_index - 1) // 2
            stop = Stop(stop_index,real_node_indexes[node_index],solution.Value(time_var),solution.Value(time_var),route_load,pickup,delivery, id_request)
            

            #print(' Vehicle {0} Node {1} Time({2}) Load({3})\n'.format(vehicle_id, real_node_indexes[manager.IndexToNode(index)],
            #                                     solution.Value(time_var), route_load))

            stops.append(stop)

            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_iter)
            
            stop_index = stop_index + 1

        end_time = solution.Value(time_var)

        route = Route(data['vehicle_ids'][vehicle_iter],start_time,end_time,stops)
        routes.append(route)
        total_distance += route_distance
        total_load += route_load



    kpis = KPIs(end_time-start_time,num_vehicles)

    return [result,routes,kpis,dropped]



#===========================================================================================
# Function that solves the Dial-a-Ride problem with Time Windows using the 
# library or-tools. The parameter data is the dictionary return by the
# function create_data_model
#===========================================================================================
def solve_CVRPTW_PD(data, solverParams: SolverParameters, priorities) -> [pywrapcp.RoutingIndexManager, pywrapcp.RoutingModel, pywrapcp.Assignment]:

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

     # Add Time Windows constraint.
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        30000,  # allow waiting time
        3000,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # Add time window constraints for each vehicle start node.
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data['time_windows'][0][0],
                                                data['time_windows'][0][1])

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')
     # Allow to drop nodes.
    #penalty = solverParams.unServedDemandPenalty
    for node in range(1, len(data['time_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], priorities[node-1])



     #Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
         transit_callback_index,
        0,  # no slack
         100,  # vehicle maximum travel distance
         True,  # start cumul to zero
         dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

     # Define Transportation Requests.
    for request in data['pickups_deliveries']:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(
                    delivery_index))
    
   

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(solverParams.maxTimeLimitSecs)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)


    return [manager, routing, solution]


#==============================================================================
# This function imports input data from a json file in the format pre-specified
# for this module
#==============================================================================

def import_data_from_json (jsonInputFile) -> [SolverParameters,Depot,[Request],[Vehicle],[[]]] :

    with open(jsonInputFile) as f:
        data = json.load(f)
    print (data)

    solverParams = SolverParameters(data['SolverParameters']['maxTimeLimitSecs'])

    timeWindowsDepot = TimeWindow(data['Depot']['time_window']['start'], data['Depot']['time_window']['end'])
    depot = Depot (data['Depot']['id'], timeWindowsDepot, data['Depot']['name'], data['Depot']['x'], data['Depot']['y'])
    vehicles = []
    for i in range(len(data['Vehicle'])):
        vehicle = Vehicle(data['Vehicle'][i]['id'], data['Vehicle'][i]['capacity'], data['Vehicle'][i]['id_depot'])
        vehicles.append(vehicle)
    requests = []
    for i in range(len(data['Requests'])):
        origin = Station(data['Requests'][i]['origin']['station_id'], data['Requests'][i]['origin']['name'])
        destination = Station(data['Requests'][i]['destination']['station_id'],data['Requests'][i]['destination']['name'])
        origin_timewindow = TimeWindow(data['Requests'][i]['origin']['time_window']['start'], data['Requests'][i]['origin']['time_window']['end'])
        destination_timewindow = TimeWindow(data['Requests'][i]['destination']['time_window']['start'], data['Requests'][i]['destination']['time_window']['end'])
        locationTWOrigin = LocationAndTimeWindow (origin, origin_timewindow)
        locationTWDestination = LocationAndTimeWindow (destination, destination_timewindow)
        request = Request (data['Requests'][i]['idRequest'], locationTWOrigin, locationTWDestination, data['Requests'][i]['dwell_time_origin'], data['Requests'][i]['dwell_time_destination'], data['Requests'][i]['demand'], data['Requests'][i]['priority'])
        requests.append(request)

    time_matrix = data['Time Matrix']

    return [solverParams, depot, requests, vehicles, time_matrix]

#==========================================================================
#
# Example of use of the library developed to solve the Dial-a-Ride problem
# with Time Windows 
#==========================================================================

def main():

    # First we input data from JSON file
    [solverParams, depot, requests, vehicles, time_matrix] = import_data_from_json('dataInput.json')
    
    # Second we call create_data_model to create the or-tool data model for VRP
    [data, real_node_indexes, priorities] = create_data_model(depot, requests, vehicles, time_matrix) 

    # Third we call solve_CVRPTW_PD to solve the Dial-a-Ride problem with TW
    [manager, routing, solution] = solve_CVRPTW_PD(data, solverParams, priorities)

    #print_solution(data, manager, routing, solution)

    # Finally we call create_output to create the data structure with the 
    # results obtained
    [results, routes, kpis, dropped] = create_output_data_model(data, manager, routing, solution, real_node_indexes)

    #The solution is printed

    print(results.status)

    
    if results.status == "OK":
        
        export_solution_to_json('dataOutput.json',routes, requests, depot, dropped)
    

    
#==========================================================================
# This function export the solution obtained to a json file in the 
# predefined format for this VRP module
#==========================================================================

def export_solution_to_json (outputJsonFile, routes, requests, depot, dropped):
    data = {}
    data['name'] = "Fleet Assigment"
    data['assignment'] = []
    for j in range(len(routes)):
        #data['assignment'].append({
        #    'idVehAssigned': routes[j].id
        #})
        actions = []
        
        for i in range(1, len(routes[j].stops)):
            if (routes[j].stops[i].delivery == 0):
                actions.append ({
                    'idRequest': requests[routes[j].stops[i].id_request].idRequest,
                    'action': 'Pickup',
                    'arrival time': routes[j].stops[i].arrival_time,
                    'depature_time': routes[j].stops[i].depature_time + requests[routes[j].stops[i].id_request].dwell_time_origin,
                    'vehicle_ocupation': routes[j].stops[i].load
                })
                
            else:
                actions.append ({
                    'idRequest': requests[routes[j].stops[i].id_request].idRequest,
                    'action': 'Drop',
                    'arrival time': routes[j].stops[i].arrival_time,
                    'depature_time': routes[j].stops[i].depature_time + requests[routes[j].stops[i].id_request].dwell_time_destination,
                    'vehicle_occupancy': routes[j].stops[i].load
                })
        
        data['assignment'].append({
            'idVehAssigned': routes[j].id,
            'actions' : actions
        })

    vehicles = []
    for j in range(len(routes)):
        vehicles.append ({
            'idVeh': routes[j].id,
            "position" : {
                'object': depot.name,
                'x': depot.x,
                'y': depot.y           
            }
        })
    data['finalCommand'] = vehicles
    #data['finalCommand'].append(vehicles)
    data['unassigned'] = []
    unassignment_request = []
    pos = 0
    while pos < len(dropped):
        unassignment_request.append({
            'idRequest' : requests[dropped[pos]].idRequest
        })
        pos = pos + 2
    #data['unassigned'].append(unassignment_request)      
    data['unassigned'] = unassignment_request

    with open(outputJsonFile, 'w') as f:
        json.dump(data, f, indent=4)


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    total_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)
        total_distance += route_distance
    print('Total Distance of all routes: {}m'.format(total_distance))
    # [END solution_printer]


#createFileInput ()
if __name__ == '__main__':
    main()