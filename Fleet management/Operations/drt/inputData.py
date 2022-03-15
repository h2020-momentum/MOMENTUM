# ====================================================================================
# Input class represents the input of the data required for the Dial-a-Ride problem.
# It has as attributes: the depot, a request list, a matrix with the data of the travel
# times between locations, an array with the vehicle fleet.
# Each request has the following information: 
#
#  - origin: data type LocationAndTimeWindow, which has as attributes Station with the 
#           identifier of the pickup location, and its time window.
#  - destination: data type LocationAndTimeWindow, which has as attributes Station with the 
#           identifier of the delivery location, and its Start and End time window.
#  - demand: integer value that represents the demand for that Job.
#  - dwell_time_origin: dwell time at the origin
#  - dwell_time_destination: dwell time at the destination
#  - priority: priority of the request. The higher the priority the most problable it will be covered by the vehicle routes.
#
# The array with the vehicle fleet data it contains the capacity of each 
# vehicle as well as the id of the depot associated to it.
#
# Apart from this, there is a class SolverParameters to establish the maximum time 
# allowed for the solver to find a solution
# ====================================================================================

class SolverParameters:
   maxTimeLimitSecs : int
   
   def __init__(self, maxtimeLimitSecs: int) -> None:
       self.maxTimeLimitSecs = maxtimeLimitSecs


class TimeWindow:
    start: int
    end: int

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end


class Depot:
    id: int
    time_window: TimeWindow
    name: str
    x: float
    y: float

    def __init__(self, id: int, time_window: TimeWindow, name: str, x: float, y: float) -> None:
        self.id = id
        self.time_window = time_window
        self.name = name
        self.x = x
        self.y = y

class Vehicle:
    id: str
    capacity: int
    id_depot: int

    def __init__(self, id: str, capacity: int, id_depot: int) -> None:
        self.id = id
        self.capacity = capacity
        self.id_depot = id_depot


class Station:
    id: int
    name: str

    def __init__(self, id: int, name: str) -> None:
        self.id = id
        self.name = name


class LocationAndTimeWindow:
    station: Station
    time_window: TimeWindow

    def __init__(self, station: Station, time_window: TimeWindow) -> None:
        self.station = station
        self.time_window = time_window


class Request:
    idRequest : str
    origin: LocationAndTimeWindow
    destination: LocationAndTimeWindow
    dwell_time_origin: float
    dwell_time_destination: float
    demand: int
    priority: int

    def __init__(self, idRequest : str, origin: LocationAndTimeWindow, destination: LocationAndTimeWindow, dwell_time_origin: float, dwell_time_destination: float, demand: int, priority: int) -> None:
        self.idRequest = idRequest
        self.origin = origin
        self.destination = destination
        self.dwell_time_origin = dwell_time_origin
        self.dwell_time_destination = dwell_time_destination
        self.demand = demand
        self.priority = priority


class Input:
    depot: Depot
    requests: []
    fleet: []
    time_matrix: [[]]

    def __init__(self, depot: Depot, requests: [], fleet: [], time_matrix: [[]]) -> None:
        self.depot = depot
        self.requests = requests
        self.fleet = fleet
        self.time_matrix = time_matrix
