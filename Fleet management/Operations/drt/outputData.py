# ==================================================================================
# 
# This classes corresponds to diferente information about the result of the
# optimization of the Dial-a-Ride problem.
#
# Result class contains the state of the solver. If a solution was obtained,
# the status attribute is set to "OK" and otherwise, it is set to "Error". 
# In the latter case, an error message is included in the attribute error_message.
# If the status attribute is set to "Error", Route and KPIs do not contain any value
# since no solution has been found.
#
# Route class represents the solution obtained, it has as attributes: the route 
# identifier, stop arrangement, start minute and end minute.
# 
# The stop array has the following information:
#  - index: order of the stop
#  - station_id: identifier of the station of the stop as given in the input data
#  - arrival_time: minute of arrival at the station
#  - depature_time: minute of exit at the station
#  - load: current load in the vehicle
#  - pickup: how many people is pick-up at that location
#  - delivery: how many people is dropped at that location
# 
# KPIs class contains the KPIs values of the obtained solution: total time 
# of the routes that compose the solution in minutes and total of routes obtained.
# ==================================================================================

class Result:
    status: str
    error_message: str

    def __init__(self, status: str, error_message: str) -> None:
        self.status = status
        self.error_message = error_message


class KPIs:
    total_time: int
    total_routes: int

    def __init__(self, total_time: int, total_routes: int) -> None:
        self.total_time = total_time
        self.total_routes = total_routes


class Stop:
    index: int
    station_id: int
    arrival_time: int
    depature_time: int
    load: int
    pickup: int
    delivery: int
    id_request: int

    def __init__(self, index: int, station_id: int, arrival_time: int, depature_time: int, load: int, pickup: int, delivery: int, id_request: int) -> None:
        self.index = index
        self.station_id = station_id
        self.arrival_time = arrival_time
        self.depature_time = depature_time
        self.load = load
        self.pickup = pickup
        self.delivery = delivery
        self.id_request = id_request


class Route:
    id: str
    start_time: int
    end_time: int
    stops: [Stop]

    def __init__(self, id: str, start_time: int, end_time: int, stops: [Stop]) -> None:
        self.id = id
        self.start_time = start_time
        self.end_time = end_time
        self.stops = stops