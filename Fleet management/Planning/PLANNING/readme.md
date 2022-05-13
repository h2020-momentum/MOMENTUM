# **Fleet planning**
This folder contain all the modules used to define the system parameters of MOMENTUM mobility services. 

# **Pre-requisites**
Python 3.8.8

# **Description of the model files**
* planning_objective.py : This module is on progress. The final form must consider AIMSUN generated data. It contains the methods to compute system costs.
* simulation_outputs.json : Sample data generated from random samppling and don't reflect the real simulation results. It works as an example of how the inputs for objective file must be. 
* stop_candidates.py : It takes as an input disaggregated demand of OD trips with timestamp and approximate the best stop candidates. Planner can insert the desired
maximum distance each user must walk to reach a available station. 
* vehicles_size.py: This module solve the VRP to define the fleet size for the operation. The user can choose the maximum route distance and the size of fleet. 
* facility_location.py : The implementation of p-median alternative for the selection of the optimal subset from station candidates 
* generate_aimsun_requests.ipynb : a guide to generate requests for the Aimsun Ride tool
# **Citing this model**
In progress... 


# **Contact**
Zisis Maleas, zisismaleas@certh.gr

