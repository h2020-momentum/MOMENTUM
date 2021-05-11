# OD matrix classification
This module is intended to support the prediction and assignment of mobility trends at the city or region level by predicting the expected OD matrix to be observed on a given day. For that purpose, OD matrices are clustered according to their both trip volumes and structures. Once clustered the present code trains a machine learning model to predict the cluster for a given day.

The model takes as input a collection of features related to the day to be predicted, including day of week, day of year, expected weather or schedule relevant events (such as major sport events). The prediction corresponds to the cluster that has been derived using actual past OD matrices. Once predicted, the OD matrix can be approximated by assigning to each cluster the average of all OD matrices within.

In addition, the node2VecClustering.R file allows testing an alternative clustering approach based on Graph Embedding techniques. This methodology is mainly based on two concepts: a) the creation of a similarity graph for the set of OD matrices to be compared and b) the application of Graph Embedding techniques. Once estimated the embeddings, the present code applies hierarchical clustering to obtain clusters of OD matrices at a specific aggregation level.

# Pre-requisites

The module works on Python 3.7.6 and requires the following libraries:
  * scipy (1.4.1)
  * numpy (1.18.1)
  * pandas (1.0.1)
  * scikit-learn (0.23.2)
  * matplotlib (3.1.3)

The clustering file works in R 4.0.3 and requires the following R libraries:
  * lubridate (1.7.9.2)
  * tidyverse (1.3.0)
  * pals (1.7)
  * ape (5.4-1)
and the following Python library
  * pecanpy (1.0.1)


# Description of the model files

* *od_matrix_classifier.py*: general pipeline for the clustering, feature appending and model development.
* *node2VecClustering.R*: alternative clustering scheme based on graph embeddings with node2Vec and hierarhical clustering.

# Usage (od_matrix_classifier.py)

*python od_matrix_classifier.py <distance_path> <feature_path> <aggregation_level>*

* *distance_path*: the path to a CSV file containing distances between each pair of matrix days. Day format: YYYYMMDD
* *feature_path*: the path to a CSV file containing features assigned to each day. Day format: YYYYMMDD
* *aggregation_level*: the aggregation level of the hierarchical clustering as defined in MOMENTUM deliverable D4.1

# Usage (node2VecClustering.R)

*RScript node2VecClustering.R <similarity_measures.csv> <clusters_info.csv> <cluster_height_threshold>*

* *similarity_measure.csv*: the path to a CSV file containing the similarity measure for each pair of matrix days. Day format: YYYYMMDD
* *clusters_info.csv*: the path to the CSV file where the results will be saved. This output file has two columns: one with the date of the OD matrix as included in the input file, and another one with the cluster in which the OD matrix is included.
* *cluster_height_threshold*: similarity cut-off threshold for defining the aggregation level of the hierarchichal clustering.

# License

MIT license

# Contact

* Ignacio Martin, ignacio.martin@nommon.es
* Antonio Masegosa, ad.masegosa@deusto.es
