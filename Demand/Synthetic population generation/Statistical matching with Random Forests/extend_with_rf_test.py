#!/usr/bin/python
import os
import numpy as np
import pandas as pd
import dataset_extension


# Load files and set mutual and target attributes
data_folder = ("C:/Users/ferran.torrent/Aimsun SLU/PD Technical task forces - Documents/TF Data Science/"
               "Experimentation/Projects/Momentum/Statistical_matching/Data/compatible_sp_survey")
output_file = os.path.join(data_folder, "agents_extended.csv")

df1 = pd.read_csv(os.path.join(data_folder, 'survey.csv'))
df2 = pd.read_csv(os.path.join(data_folder, 'agents.csv'))

mutual_attrs = df1.columns[1:-2]
target_attrs = ['belgian', 'public_transport_passes_hh']
K = len(mutual_attrs)

# Onehot encoding of attributes
x_mat1 = np.array(df1[mutual_attrs], dtype=str)
y_mat1 = np.array(df1[target_attrs], dtype=np.int32)
y_mat1 = y_mat1 > 0
x_mat2 = np.array(df2[mutual_attrs], dtype=str)

x_mat1, x_mat2 = dataset_extension.to_onehot(x_mat1, x_mat2)

# CV testing - CV testing assumes binary target variables
dataset_extension.rf_cv_test(x_mat1, y_mat1, n_folds=5)

# Complete dataset
# Use n_splits=None to let trees expand until leaves are pure or have less support than min_samples_split=2
y_mat2 = dataset_extension.extend_with_rf(x_mat1, y_mat1, x_mat2, n_estimators=1000, n_splits=None)
y_mat2_sampled = dataset_extension.sample_from_prob(y_mat2)
pd.concat((df2, pd.DataFrame(data=y_mat2_sampled, columns=target_attrs, index=df2.index)),
          axis=1).to_csv(output_file, index=False)

print()





