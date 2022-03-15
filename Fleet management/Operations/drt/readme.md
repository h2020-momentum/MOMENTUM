
# Dial-a-Ride problem

This model defining a set of minimum cost routes in order to satisfy a set of transportation requests.

# Pre-requisites

This model is coded in 'Python 3.8.8' language. The next two libraries are required:

- or-tools:	pip install --upgrade --user ortools
- beeprint:	pip install beeprint

# Description of the model files

This module for the Dial-a-Ride problem is composed of the next three files:

- CVRPTW-PD_final.py: Main file with the methods to solve the Dial-a-Ride problem given the input data. It also contains an example of how to use the different methods.
- InputData.py: Classes to model the input data. Please, check in order to see the information needed to solve the problem.
- OutputData.py: Classes to model the solution found for the Dial-a-Ride problem. Please, check in order to see the data available about the solution found.

It includes two files in JSON format with an example of the input and output data:
- InputData.json: sample input data file
- OutputData.json: sample output data file

# Citing this model

@software{drt-MOMENTUM,    
  title = {DRT optimisation module - EU H2020 MOMENTUM Project (GA 815069)},  
  version = {1.0},  
  author = {Jenny Fajardo and Antonio D. Masegosa and Pablo Fernandez},  
  organization = {University of Deusto},  
  url = {https://github.com/h2020-momentum/MOMENTUM/tree/main/Fleet%20management/Operations/drt},  
  date = {2022-3-15}  
}  

# License

Distributed under the MIT License.

# Contact

- Jenny Fajardo (University of Deusto, Bilbao) – fajardo.jenny@deusto.es
- Antonio D. Masegosa (University of Deusto, Bilbao) – ad.masegosa@deusto.es
- Pablo Fernandez (University of Deusto, Bilbao) - pablo.fernandez@deusto.es