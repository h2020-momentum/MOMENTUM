# New mode choice / Disaggregate mode choice model

This model is intended to estimate modal split between (i) conventional transport modes as a whole, (ii) bike-sharing, (iii) car-sharing, and (iv) ride-sharing. 
A CSV file with synthetic population data, trip travel times and sharing vehicle availability are taken as an input and the output is also a CSV file, with mode choice combined with the input data. 

# Pre-requisites

This model is coded in 'R' language. For details about R, kindly check https://www.r-project.org. RStudio IDE is suggested (https://www.rstudio.com). This model uses the base functions in R and hence, no special package is required.

# Description of the model files
1.	CoefficientValues.csv – Model coefficients; unless you want to use custom values, do not change the values in the file
2.	NewModeChoiceModelPrediction.R – Main model script
3.	InputSampleFile.csv – Sample input file, for understanding the inputs and testing the model

# Citing this model

Narayanan, S. and Antoniou, C. (2022). Shared mobility services towards Mobility as a Service (MaaS): What, who and when?. Manuscript under review.

# License

Distributed under the MIT License.

# Contact

Santhanakrishnan Narayanan (TU, Munich) – santhanakrishnan.narayanan@tum.de
