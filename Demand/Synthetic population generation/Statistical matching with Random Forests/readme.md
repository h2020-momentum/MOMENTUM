# Statistical matching with Random Forests

This module provides a way of extending the attributes of a target population (agents.csv in the example), by training
a random forest with a smaller (less samples) but wider (more attributes) dataset (survey.csv in the example) and 
using it to estimate the missing attributes on the target population.

The module take advantage of the ability of random forests for modelling conditional probabilities to predict the 
missing attributes of the target population. Moreover, the module takes advantage of the efficient implementation of 
scikit-learn.

## Pre-requisites

This module is coded in Python language and works with Python >= 3.7.6. The following packages are required:

- scikit-learn >= 1.0.2
- numpy >= 1.21.5
- pandas >= 1.4.2

## Description of the files
1.	extend_with_rf_test.py – Main model script
2.	data_extension.py – Python file with useful functions
3.	*survey_sample.csv* – Input CSV file with the training dataset from household travel survey observations (example indicating the structure of the       input survey file)
4.	*agents_sample.csv* – Input CSV file with the dataset to extend (example indicating the structure of the input agents file)
5.	
## How to use it

extend_wih_rf_test.py is the main script (the one to run). Before running it:
- Edit from line 9 to 17 to set the folder with the user provided survey.csv and agents.csv files,
the output file, the common attributes in both datasets and the target attributes (the ones to estimate for the target
population).
- Edit "n_estimators" or "n_splits" (line 33) to increase/decrease the regularization of the random forest. "n_splits"
can be used to control the size of each decision tree, the smaller the more regularized, so the less overfitting.
"n_estimators" can be used to improve the accuracy of the overall random forest. The more trees the more accurate w/o
an increase of the overfitting.

# License
Distributed under the MIT License.

# Contact
Athina Tympakianaki, athina.tympakianaki@aimsun.com

Ferran Torrent, ferran.torrent@aimsun.com
