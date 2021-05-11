# Shared mobility demand prediction
This model uses segmented general OD matrices to predict the expected trips performed in a shared mobility service on an OD pair basis. The model takes as input features the general volumes of trips, weather information and land use at origin and destination.

The model is a regressor that estimates the number of trips observed per OD pair and day of week. To facilitate the prediction of a scarce variable, the output is temporally aggregated by day of the week and segmented according to the distance quantiles of the OD pairs included in the set (according to each service).

# Pre-requisites
The module works on Python 3.7.6 and requires the following libraries:
  * numpy (1.18.1)
  * pandas (1.0.1)
  * scikit-learn (0.23.2)
  * matplotlib (3.1.3)

# Usage

*python demand_model.py <base_path> <num_weeks_train> [<output_path>]*

* *base_path*: The path to a directory where required datasets will be stored. This directory should contain two files:
  * *train_data.csv*: A CSV containing the training data, with a column per feature and an additional column (shared_mobility) to work as target variable.
  * *centroids.csv*: A CSV file containing for each zone ID the longitude and latitude values of its centroid.
* *num_weeks_train*: The total number of weeks to be used in training to adjust prediction.
* *output_path*: the path to the output directory. OPTIONAL, if not used output directory will be the same as base_path.

# License

MIT license

# Contact

  Ignacio Martin, ignacio.martin@nommon.es
