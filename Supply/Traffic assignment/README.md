# Stochastic User Equilibrium

This model predicts network flows under the assumption that drivers distribute over routes according to stochastic user equilibrium. Every link in a zone is considered a potential source and sink of traffic. 
The inputs are:
-	an origin-destination matrix of demand between origin and destination zones. 
-	tables with all the links and nodes in the network with the required characteristics. 
-	parameter for distributing the flow similar to the scaling parameter of logit.


# Pre-requisites

This model is coded in ‘Matlab’. For details about Matlab, kindly https://nl.mathworks.com. This model uses the base function in Matlab and hence, no special package is required. 

# Description of the model files

A detailed description is provided at the header of each file. 
1.	SUE.m –  Main entry point
2.	calculateCostBPR.m –
3.	Stoch_noCon_SQZ.m –
4.	sparse_to_csr.m –
5.	dijkstra.m –
6.	find_destination_links.m –
7.	find_access_nodes.m –
8.	compute_acces_weight.m –
9.	init_flows_SQZ.m –
10.	get_weights.m –
11.	propagate_flows_SQZ.m –
12.	distribute_flows_SQZ.m –


# Citing this model
Himpe, W. and Frederix, R. (2021). Stochastic network loading in urban environments. Manuscript in preperation.

# License
Distributed under the CC BY-NC-SA License

# Contact

- Willem Himpe, willem.himpe@tmleuven.be
- Rodric Frederix, rodric.frederix@tmleuven.be

