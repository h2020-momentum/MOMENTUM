#!/usr/bin/python
import os
import numpy as np
import pandas as pd
import dataset_extension


data_folder = ("C:/Users/ferran.torrent/Aimsun SLU/PD Technical task forces - Documents/TF Data Science/"
               "Experimentation/Projects/Momentum/Statistical_matching/Data/compatible_sp_survey")
output_file = ("")

df1 = pd.read_csv(os.path.join(data_folder, 'survey.csv'))
df2 = pd.read_csv(os.path.join(data_folder, 'agents.csv'))

# Hyperparams
M = 20
mutual_attrs = df1.columns[1:-2]
# mutual_attrs = ['whatever1', 'whatever2']
target_attrs = ['belgian', 'public_transport_passes_hh']
K = len(mutual_attrs)

# Initialize x/y arrays
x_mat_1 = np.array(df1[mutual_attrs], dtype=str)
y_mat_1 = np.array(df1[target_attrs], dtype=np.int32)
y_mat_1 = y_mat_1 > 0
x_mat_2 = np.array(df2[mutual_attrs], dtype=str)
y_mat_2 = np.zeros((x_mat_2.shape[0], y_mat_1.shape[1]), np.int32)

# Cross-validation testing
dataset_extension.statistical_matching_cv_test(x_mat_1, y_mat_1, n_folds=5)

# Get extended datset - use verbose to keep track of the evolution of the algorithm (it is very slow)
y_mat2 = dataset_extension.statistical_matching(x_mat_1, y_mat_1, x_mat_2, M=20, verbose=1)

# Save the dataset
pd.concat((df2, pd.DataFrame(data=y_mat2, columns=target_attrs)), axis=1, ignore_index=True).to_csv(output_file,
                                                                                                    index=False)
print()



