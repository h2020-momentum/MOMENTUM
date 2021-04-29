# Roiundtrip station based car-sharing

In certain cities (e.g., Regensburg), a roundtrip station based car-sharing system is implemented to complement public transport, by designing the car-sharing business 
model to focus on serving special trips (e.g., trips to furniture stores). Many European cities, especially small and medium sized cities, continue to use the traditional 
strategic 4-step modelling approach. The OD matrices from these models, usually, do not adequately cover the demand stratum (i.e., special trips) of the aforementioned 
car-sharing system. Furthermore, when a car-sharing system is small, the modal split for them is very less (< 50 trips per day for the whole system), 
making their use a very rare event, thereby resulting in a situation where it is not possible to account them through the traditional mode choice models.  
Therefore, the demand has to be additionally estimated using an external approach. Besides demand, it is also beneficial to profile the users of such a system, 
so that the demand can be linked to individuals. 

Therefore, this model is intended to estimate a data-driven demand for the aforementioned system, using a multi-method framework. A linear regression is used to estimate the
total demand for the system per day. Subsequently, a Dirichlet regression is used to split the total demand to individual stations. Then, statistical sampling is used to 
assign trip distances for the trips. Assignment of trip origin is based on the station locations, while nearest value matching is used to compare the OD distances
and the distances from the statistical sampling, to assign the trip destination. Finally, a multinomial logit model is used to profile the users.

CSV files consisting of station data, trip distance distribution (could be fetched from existing operator data) and analysis period (day & month) are taken as input 
for estimating the data-driven demand. A CSV file of synthetic population is used for profiling the users. Following outputs are saved: (i) Trip requests and (ii) Synthetic 
population with the frequency of use of the sharing system. It is to be noted that random sampling of the synthetic population, based on use frequency, could be done to 
assign users for the trips requests.

# Pre-requisites

This model is coded in 'R' language. For details about R, kindly check https://www.r-project.org. RStudio IDE is suggested (https://www.rstudio.com). To run this model, 'dplyr'
package is needed.


# Citing this model

Narayanan, S. and Antoniou, C. (2021).  Development of a data-driven demand model for a roundtrip station-based car-sharing system. Manuscript in preparation.

# License

Distributed under the MIT License.

# Contact

Santhanakrishnan Narayanan (TU, Munich) – santhanakrishnan.narayanan@tum.de
