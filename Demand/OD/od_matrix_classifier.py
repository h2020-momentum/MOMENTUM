#!/usr/bin/python

"""
This file contains the code for an OD matrix classification scheme that aggregates OD matrices by
means of their matrix-similarity and develops a classification model that determines which cluster
each matrix belongs to using date-related features, such as date features, weather or scheduled
events.

For further details, please refer to the Deliverable 4.1 of the H2020 MOMENTUM project.

https://h2020-momentum.eu/

This module was developed by the Nommon Team


"""

__author__ = 'imartin'
__copyright__ = '(c) Nommon 2021'

import os
import sys

from joblib import dump
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plotDendrogram( model, **kwargs ):
    '''
    This function plots a dendrogram represented by the model passed as parameter. This function
    has been copied from sklearn agglomerative cluster repository

    Args:
        model(sklearn.AgglomerativeClusteringModel): A trained agglomerative clustering object that
            has been trained using the distance_threshold approach.
        **kwargs (list): A list of other arguments required for the dendrogram plot.

    '''
    # create the counts of samples under each node
    counts = np.zeros( model.children_.shape[0] )
    n_samples = len( model.labels_ )
    for i, merge in enumerate( model.children_ ):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack( [model.children_, model.distances_,
                                      counts] ).astype( float )

    # Plot the corresponding dendrogram
    dendrogram( linkage_matrix, **kwargs )


def validateClusterScheme( distances, labels, metric_type='precomputed' ):
    '''
    This function computes and reports a validation report on the clustering metric based on
    silhouette score and the counts of DoW in each cluster.

    Args:
        distances (pd.DataFrame): The distance matrix used for the clustering.
        labels (pd.Series): The labels assigned to each element.
        output_path (str): Path to the output directory where the txt report will be stored.

    '''
    silhouette = silhouette_score( distances, labels, metric=metric_type )
    print( 'Silhouette score: ', silhouette )
    return silhouette


def trainLinearModel( x_train, y_train, output_path=None ):
    '''
    This function is in charge of creating a new model and training it using the training data
    passed as a parameter.

    Args:

        x_train (pd.DataFrame): A pandas dataset containing the features for the training set.
        y_train (pd.Series): A pandas series that contains the target observations of each
            data point observed in x_train.
        output_path (str): if set stores the model in the directory provided. If not set the model
            is not stored.

    Returns:
        skelarn.Model: A linear model object that has been trained using the given training data.
            Only the model is returned.


    '''
    param_grid = {'C': np.linspace( 100, 190, 10 )}
    print( param_grid )
    clf = LogisticRegression( multi_class='multinomial', solver='lbfgs' )
    grid = GridSearchCV( clf, param_grid, scoring='f1_macro' )
    grid.fit( x_train, y_train )
    clf = grid.best_estimator_
    coeffs = pd.DataFrame( clf.coef_ )
    coeffs.columns = x_train.columns
    print ( grid.best_params_ )
    if output_path:
        if not os.path.exists( output_path ):
            os.makedirs( output_path )
        model_path = os.path.join( output_path, 'linear_model.joblib' )
        dump( clf, model_path )
        coefficients_path = os.path.join( output_path, 'model_coeficients.csv' )
        coeffs.to_csv( coefficients_path )
    return clf


def trainTreeModel( x_train, y_train, output_path=None ):
    '''
    This function is in charge of creating a new model and training it using the training data
    passed as a parameter.

    Args:

         x_train (pd.DataFrame): A pandas dataset containing the features for the training set.
         y_train (pd.Series): A pandas series that contains the target observations of each
             data point observed in x_train.
         output_path (str): if set stores the model in the directory provided. If not set the model
             is not stored.

     Returns:
         skelarn.Model: A tree model object that has been trained using the given training data.
             Only the model is returned.

    '''
    param_grid = {'max_depth': [2, 3, 4, 5, 6], 'criterion': ['entropy', 'gini'],
                  'min_samples_leaf': [2, 3, 4, 5, 6]}
    clf = DecisionTreeClassifier( class_weight='balanced' )  #
    grid = GridSearchCV( clf, param_grid, scoring='f1_macro', n_jobs=4 )
    grid.fit( x_train, y_train )
    clf = grid.best_estimator_
    coeffs = pd.DataFrame( clf.feature_importances_ )
    coeffs.columns = ['importance']
    coeffs.index = x_train.columns
    print( 'Tree Feature importances:' )
    print( coeffs['importance'].sort_values( ascending=False ).to_string() )
    print( 'Tree Selected parameters:' )
    print( grid.best_params_ )
    plt.subplots( nrows=1, ncols=1, figsize=( 16, 12 ), dpi=300 )
    plot_tree( clf, filled=True, label='root', feature_names=x_train.columns.values,
               proportion=True )
    if output_path:
        figure_path = os.path.join( output_path, 'resulting_tree.png' )
        plt.savefig( figure_path, dpi=300, bbox_inches='tight', transparent=True )
        model_path = os.path.join( output_path, 'tree_model.joblib' )
        dump( clf, model_path )
        importances_path = os.path.join( output_path, 'importances.csv' )
        coeffs.to_csv( importances_path )
    return clf


def validateModel( model, x_train, x_test, y_train, y_test ):
    '''
    This function performs model validation following the f_score and accuracy metrics.

    Args:
        model (sklearn.model): A trained sklearn model to be validated.
        x_train (pd.DataFrame): A pandas dataframe containing the features for the train set.
        x_test (pd.DataFrame): A pandas dataframe containing the features for the test set.
        y_train (pd.Series): A pandas Series containing the target variable for the train set.
        y_test (pd.Series): A pandas Series containing the target variable for the test set.

    '''

    train_accuracy = model.score( x_train, y_train )
    test_accuracy = model.score( x_test, y_test )
    pred_train = model.predict( x_train )
    pred_test = model.predict( x_test )
    f1_train = f1_score( y_train, pred_train, average='macro' )
    f1_test = f1_score( y_test, pred_test, average='macro' )
    print( 'Accuracy (train/test): {:.3f}/{:.3f}'.format( train_accuracy, test_accuracy ) )
    print( 'F1 score (train/test): {:.3f}/{:.3f}'.format( f1_train, f1_test ) )


def computeDendrogram( input_path, output_path, distance_threshold ):
    '''
    This function loads data from CSV, processes input data, computes the hierarchical clustering
    and computes and stores a dendrogram. As a result it plots a dendrogram and stores a CSV file
    with the cluster ID to the given output folder.

    Args:
        input_path (str): The path to the input matrix path. Such matrix should contain the input
            features and the distance for each given OD matrices.
        output_path (str): The path to the dir where results will be stored.
        distance_threshold (float): The selected cutt-off distance for the cluster.

    Returns:
        str: The path where the cluster path has been effectively stored.

    '''
    distances = pd.read_csv( input_path, index_col=0 )
    cluster = AgglomerativeClustering( distance_threshold=distance_threshold, n_clusters=None )
    cluster_predictions = cluster.fit_predict( distances )
    validateClusterScheme( distances, cluster_predictions )
    cluster_corr = pd.DataFrame( {'day': distances.index.values, 'cluster': cluster_predictions} )
    plt.figure( figsize=( 15, 40 ) )
    plotDendrogram( cluster, labels=distances.index.values, show_leaf_counts=True,
                    color_threshold=distance_threshold, orientation='left' )
    output_fig = os.path.join( output_path, 'dendrogram.png' )
    plt.savefig( output_fig, dpi=300 )

    output_file = os.path.join( output_path, 'cluster_file.csv' )
    cluster_corr.to_csv( output_file, index=False )
    return output_file


def determineHierarchicalLevel( selected_level ):
    '''
    This function translates a given clustering level into a hierarhical clustering level.

    Args:
        selected_level (str): A clustering level of those defined in the MOMENTUM Project
            deliverable 4.1.

    Returns:
        float: The cut-off distance associated to the given level.

    '''
    LEVEL0_CUTOFF = 5e-8
    LEVEL1_CUTOFF = 3e-8
    LEVEL2_CUTOFF = 2e-8
    LEVEL3_CUTOFF = 1e-8
    if '0' in selected_level:
        return LEVEL0_CUTOFF
    if '1' in selected_level:
        return LEVEL1_CUTOFF
    if '2' in selected_level:
        return LEVEL2_CUTOFF
    if '3' in selected_level:
        return LEVEL3_CUTOFF


def appendFeaturesExternal( input_path, output_path, feature_path ):
    '''
    This function is the main process of the module that applies feature extraction from the day of
    the matrix.

    Args:
        input_path (str): The path where the relation of days and clusters are stored. The main
            index of this matrix is day in YYYYMMDD format.
        output_path (str): The path where the resulting matrix containing all new features will be
            stored.
        feature_path (str): The path to the table containing the input features organised by
            day in YYYYMMDD format.

    '''
    data_frame = pd.read_csv( input_path )
    feature_frame = pd.read_csv( feature_path )
    result_frame = data_frame.merge( feature_frame, on='day' )
    result_frame.to_csv( output_path, index=False )


def trainAndValidateSet( input_path, target_variable='cluster', model_type='linear',
                         output_path=None, proposed_test=0.2 ):
    '''
    This function performs a training and validation process that initially splits into train and
    test set and performs the training and posterior validation of a machine learning model of the
    specified type.

    Args:
        input_path (str): The path to the input training data.
        target_variable (str): The name of the variable to be used as target. If unset defaults to
            "cluster".
        model_type (str): A identifier to the model type to be developed. Currently supported:
            linear-> Multinomial logistic regression; Tree-> Decission Tree.
        output_path (str): If set stores the model as a joblib as well as the model validation
            (results and plot).
        proposed_test (float): Amount of data kept for the test set over a total of 1.

    '''
    data_frame = pd.read_csv( input_path )
    y_values = data_frame[target_variable]
    x_values = data_frame.drop( [target_variable], axis=1 )
    x_train, x_test, y_train, y_test = train_test_split( x_values, y_values,
                                                         test_size=proposed_test )
    if model_type == 'linear':
        model = trainLinearModel( x_train, y_train )
    if model_type == 'tree':
        model = trainTreeModel( x_train, y_train, output_path=output_path )
    validateModel( model, x_train, x_test, y_train, y_test )


if __name__ == '__main__':
    if len( sys.argv ) != 4:
        print( 'ERROR. Usage: python {} <distance_path> <feature_path> <aggregation_level>'
               .format( sys.argv[0] ) )
        exit( -1 )

    distance_path = sys.argv[1]
    features_path = sys.argv[2]
    level = sys.argv[3]
    base_dir = os.path.dirname( distance_path )
    train_path = os.path.join( base_dir, 'train.csv' )

    # STEP 1: develop hierarchical clustering scheme.
    cluster_distance = determineHierarchicalLevel( level )
    cluster_file = computeDendrogram( distance_path, base_dir, cluster_distance )
    # STEP 2: Append variables
    appendFeaturesExternal( cluster_file, train_path, features_path )
    # STEP 3: Perform model
    trainAndValidateSet( train_path, output_path=base_dir, model_type='tree' )

