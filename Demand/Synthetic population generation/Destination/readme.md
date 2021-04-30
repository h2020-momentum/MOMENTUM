# Destination choice model

This model is intended to add a destination to each synthesised person, generated using the PopGen tool. The model can be used in cases where information related to the trip distances and destinations are not available both in the census and travel survey data so that they cannot be mapped using PopGen in the initial population synthesis.  The destination and trip distances for each individual are imputed using a statistical approach and information from available Origin-Destination matrices.

CSV files with the available OD demand matrices, the synthesised population, the zoning system as well as a street map are taken as input. The output is also a CSV file, containing the synthetic population with an assigned destination zone or specific location. The method is adopted from (*Hörl, S. and Balac, M., 2020. Reproducible scenarios for agent-based transport simulation: A case study for Paris and Île-de-France.*)
Pre-requisites
This model will be coded in Python language. More details regarding any special packages that may be required will be provided when the model will be completed. For details about Python, kindly check https://www.python.org/.

# Description of the model files

1.	*ODMatrices.csv* (in progress) – File with available aggregate Origin-Destination matrices (could be provided per mode, trip purpose)
2.	*InputSyntheticPopulation.csv* (in progress) – File with synthesised population observations derived from the PopGen tool
3.	*ZoningSystem.csv* (in progress) – File with information related to the geographic resolution of the zones in the OD matrices (e.g. census tract, TAZ)
4.	Map (in progress) – To obtain specific locations within the zones (x,y coordinates)
5.	DestinationChoice.py (in progress) – Main model script

# Citing this model
In progress.

# License

# Contact
Athina Tympakianaki, athina.tympakianaki@aimsun.com
