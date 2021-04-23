#!/usr/bin/python

"""
This file contains the code for the training of a shared mobility demand prediction module. The
file defines all the necessary functions to develop the training and prediction of a machine
learning model that takes as input a set of mobility variables and provides the prediction of the
expected shared mobility demand to be observed at each OD pair for a given service.

Input variables to the model can be any set of variables that relate with mobility. For more
detailed information, the user can refer to Deliverable 4.1 of the H2020 MOMENTUM project.

https://h2020-momentum.eu/

This module was developed by the Nommon Team

"""

__author__ = 'imartin'
__copyright__ = '(c) Nommon 2021'

from math import asin
from math import radians
from shutil import copyfile
import os
import re
import sys
import time

from joblib import dump, load
from numpy import cos
from numpy import sin
from numpy import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import ujson

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# GLOBAL VARIABLES ###############################################################################
SHARED_MOB_TRIPS = 'shared_mobility'
TEST_SIZE = 0.2


# MACHINE LEARNING BASED AUXILIARY FUNCTIONS ######################################################


def createModel( model_type, model_params ):
    '''
    This function creates a new model object including their regularization and hyper-parameters.

    Args:
        model_type (str): A identifier for the expected type of model. Currently supported:
            linear regression (l2 and l1), decision tree regression and random forest regression.
        model_params (dict): A dict containing selected hyper-parameters relevant for the model.

    Returns:
        sklearn.model: The model object and a meta-data object associated to it.

    Note:
        If the model indicated in model_type is not supported, raises a Not implemented Error.

    '''
    if model_type == 'linear':
        penalty = model_params.get( 'penalty', 'l2' )
        if penalty == 'l2':
            from sklearn.linear_model import RidgeCV
            if model_params is not None:
                c_value = model_params.get( 'C', [0.01, 0.5, 1] )
                model = RidgeCV( alphas=c_value, normalize=False )
        else:
            from sklearn.linear_model import LassoCV
            if model_params is not None:
                c_value = model_params.get( 'C', 1 )
                model = LassoCV( alphas=c_value, normalize=False, tol=1e-12 )
    elif model_type == 'tree':
        from sklearn.tree import DecisionTreeRegressor
        if model_params is not None:
            max_depth = model_params.get( 'max_depth', 5 )
            model = DecisionTreeRegressor( max_depth=max_depth )
        else:
            model = DecisionTreeRegressor( max_depth=5 )
    elif model_type == 'forest':
        from sklearn.ensemble import RandomForestRegressor
        if model_params is not None:
            n_estimators = model_params.get( 're', 30 )
            max_depth = model_params.get( 'max_depth', 5 )
            model = RandomForestRegressor( n_estimators=n_estimators, max_depth=max_depth,
                                           n_jobs=4 )
        else:
            model = RandomForestRegressor()

    else:
        raise NotImplementedError

    return model


def loadData( training_data_path, dependent_var_col, threshold_drop_warning=0.1 ):
    '''
    This function loads the specified dataset into two numpy arrays: a matrix containing features
    and a vector containing target variables.

    Args:
        training_data_path (str): The path to the csv containing the data batch.
        dependent_var_col (str): column in the data for the dependent variable (the y).
        threshold_drop_warning (float): if the amount  of invalid rows that are dropped is greater
            than this value, a warning is printed.

    Returns:
        np.array, np.array: Features and target variable as numpy arrays.

    '''
    data = pd.read_csv( training_data_path, delimiter=',', engine='c', index_col=None )
    total_rows = data.shape[0]
    data = data.dropna()  # In case some point is NA
    clean_rows = data.shape[0]
    try:
        amount_droped = ( total_rows - clean_rows ) / total_rows
    except ZeroDivisionError:
        amount_droped = 0
    if amount_droped > threshold_drop_warning:
        print( 'WARNING: File: {} suffers from {} point drop from na'.format( training_data_path,
                                                                              amount_droped ) )
    try:
        # Remove Impossible points where there is shared mobility and not general mobility (Type 1)
        data = data.loc[-( ( data[dependent_var_col] != 0 ) & ( data.ALL == 0 ) )]
        type_1_shape = data.shape[0]
        num_type1_points = float( clean_rows - type_1_shape )
        # Remove impossible points where shared mobility has a very large modal share (Type 2).

        data = data.loc[( data[dependent_var_col] / data['ALL'] ).fillna( 0 ) <= 1]
        num_type2_points = float( type_1_shape - data.shape[0] )

        print( 'Impossible point_count: Type 1: {:,}({:.2f}%); Type 2: {:,}({:.2f}%)'
               .format( num_type1_points, num_type1_points / clean_rows * 100, num_type2_points,
                        num_type2_points / clean_rows * 100 ) )
    except AttributeError:
        pass

    y_data = data[dependent_var_col]
    x_data = data.drop( [dependent_var_col], axis=1 )
    # Drop variables not needed for training
    dropable = ['origin', 'destination', 'date']
    x_data = x_data.drop( dropable, axis=1 )
    x_data = x_data.values
    y_data = y_data.values
    return x_data, y_data


def trainModel( model, training_data_path ):
    '''
    This method is in charge of executing the model training using the data passed as a parameter.

    Args:
        model (sklearn.model): The model object, may have been already trained.
        training_data_path (str): The path to the csv containing new data to be used in training.

    Returns:
        sklearn.model: A model that has been fitted with the given data

    '''
    global SHARED_MOB_TRIPS
    try:
        x_data, y_data = loadData( training_data_path, SHARED_MOB_TRIPS )
        model.fit( x_data, y_data )
    except IOError:
        print( 'Warning, file not found: {}'.format( training_data_path ) )
        return None
    return model


def predict( model, predict_data_path ):
    '''
    This method performs the prediction of the data stored in the path passed as a parameter.

    Args:
        model (sklearn.model): The model object, may have been already trained.
        predict_data_path (str): The path to the CSV containing new data to be used in training.

    Returns:
        np.array, np.array: Actual target and predicted target.

    '''
    global SHARED_MOB_TRIPS
    try:
        x_data, y_data = loadData( predict_data_path, SHARED_MOB_TRIPS )
        y_pred = model.predict( x_data )
    except IOError:
        print 'Warning, file not found: {}'.format( predict_data_path )
        return None, None
    return y_data, y_pred


def validateModel( model, test_data_path ):
    '''
    This method performs a set of performance metrics of the model using the data collected as
    test_data_path.

    Args:
        model (sklearn.model): The model object, may have been already trained.
        test_data_path (str): The path to the csv containing new data to be used in training.

    Returns:
        dict: Validation results of the system for the given dataset.

    '''
    result_dict = {}
    y_data, y_pred = predict( model, test_data_path )
    if y_data is not None:
        result_dict['r_squared'] = r2_score( y_data, y_pred )
        result_dict['mse'] = mean_squared_error( y_data, y_pred )
        result_dict['rmse'] = sqrt( mean_squared_error( y_data, y_pred ) )
        result_dict['mae'] = mean_absolute_error( y_data, y_pred )
        result_dict['meae'] = median_absolute_error( y_data, y_pred )
        total_counts = y_data.sum()
        predicted_counts = y_pred.sum()
        perc_difference = ( predicted_counts - total_counts ) / predicted_counts * 100
        print( 'Actual trip count: {act:,.0f}; Predicted trip count: {pred:,.0f} -> {pred:,.0f}/{act:,.0f}'
               .format( act=total_counts, pred=predicted_counts ) )
        print 'Difference: {:,.2f}%'.format( perc_difference )
        result_dict['trip_count_pred'] = total_counts
        result_dict['trip_count_act'] = predicted_counts
        result_dict['variation_trip_count'] = perc_difference
    return result_dict


def splitDataset( training_data_path, t_size=0.2, append='' ):
    '''
    This function splits the provided data into training and testing sets. Supports the data split
    under a dictionary-based distance segmentation.

    Args:
        training_data_path (str): The path to the original training data.
        t_size (str): Size of the testing set in percentage (under 1).
        append (str): append string to title if more models are to be created at once.

    Returns:
        str, str: The train and test data path.

    '''
    global SHARED_MOB_TRIPS

    def simpleSplit( data_path, t_size, append ):
        '''
        This function is in charge of splitting each simple dataset into train and test according to
        the parameters passed as parameters, including test_size and additional text appends into
        files.

        Args:
            data_path (str): The path to the data to be split.
            t_size (float): The amount of data over a total of 1 to be used as test set.
            append (str): Additional text to append to the file names of each data subset.
        '''
        entire_frame = pd.read_csv( data_path, index_col=None )
        y_data = entire_frame[SHARED_MOB_TRIPS]
        x_data = entire_frame.drop( [SHARED_MOB_TRIPS], axis=1 )
        x_train, x_test, y_train, y_test = train_test_split( x_data, y_data, test_size=t_size )
        train_data = x_train
        train_data[SHARED_MOB_TRIPS] = y_train
        test_data = x_test
        test_data[SHARED_MOB_TRIPS] = y_test
        data_dir = os.path.dirname( data_path )
        train_data_path = os.path.join( data_dir, 'train{}.csv'.format( append ) )
        test_data_path = os.path.join( data_dir, 'test{}.csv'.format( append ) )
        train_data.to_csv( train_data_path, index=False )
        test_data.to_csv( test_data_path, index=False )
        return train_data_path, test_data_path

    train_dict = {}
    test_dict = {}
    for split in training_data_path:
        new_append = '{}_{}_{}'.format( append, split[0], split[1] )
        train, test = simpleSplit( training_data_path.get( split ), t_size, new_append )
        train_dict[split] = train
        test_dict[split] = test

    test_result = {'test': test_dict}

    return train_dict, test_result


# MAIN PROCESS FUNCTIONS #########################################################################


def trainAggregatedModel( train_data_path, model_type, model_params ):
    '''
    This function performs a round of model training and the corresponding training score
    generation.

    Args:
        train_data_path (str): The path to the training dataset. Must be a path to a file.
        model_type (str): An idenfier for the model underlying algorithm.
        model_params (dict): A dictionary containing model hyper-parameters. Only those parameters
            related to the algorithm defined in model_type will be used.

    Returns:
        sklearn.model, dict, tuple: The trained model and a dictionary containing for each
        phase a report dictionary. Also the predictions->(actual, predicted)

    '''
    global SHARED_MOB_TRIPS

    # Create and train model.
    all_distance_train = []
    y_train_g = []
    pred_train_g = []
    distance_models = {}
    for distance in train_data_path:
        print ( 'Distance_range: {}-{}'.format( distance[0], distance[1] ) )
        model = createModel( model_type, model_params )
        train_path = train_data_path.get( distance )
        model = trainModel( model, train_path )
        distance_models[distance] = model
        y_train, pred_train = predict( model, train_path )
        result_dict_train = validateModel( model, train_path )
        result_dict_train['distance_range'] = '{}_{}'.format( distance[0], distance[1] )
        all_distance_train.append( result_dict_train )
        pred_train_g.append( pred_train )
        y_train_g.append( y_train )
    y_train_g = np.concatenate( y_train_g, axis=0 )
    pred_train_g = np.concatenate( pred_train_g, axis=0 )
    return distance_models, all_distance_train, ( y_train_g, pred_train_g )


def performValidation( model, validation_data_path ):
    """
    This function performs the validation of the given model using the left out data that has been
    already processed and only has to be used for prediction.

    Args:
        model (dict of sklearn.model): The trained model to be used in a dict with a model for each
            distance range.
        validation_data_path (str): The absolute path to the dataset to be used in validation.
        target_column (str): The name of the dependent variable.

    Return:
        dict: A dictionary containing another dictionary with the validation report of each date.

    """
    global SHARED_MOB_TRIPS
    all_result_dicts = {}
    for date in validation_data_path:
        # Maintain only last date for plot.
        pred_test_g = []
        y_test_g = []
        all_distances = []
        for distance in model:
            print( 'Distance range: {}-{}'.format( distance[0], distance[1] ) )
            result_dict = {}
            data_set = validation_data_path.get( date, {} ).get( distance )
            y_test, pred_test = predict( model.get( distance ), data_set )
            result_dict = validateModel( model.get( distance ), data_set )
            result_dict['distance_range'] = '{}-{}'.format( distance[0], distance[1] )
            all_distances.append( result_dict )
            pred_test_g.append( pred_test )
            y_test_g.append( y_test )
        all_result_dicts[date] = all_distances
        # This just takes the last day of study for the test prediction
    y_test_g = np.concatenate( y_test_g, axis=0 )
    pred_test_g = np.concatenate( pred_test_g, axis=0 )
    return all_result_dicts, ( y_test_g, pred_test_g )


# MODEL MANAGEMENT ###############################################################################


def storeModel( output_path, model_id, model, result_frame, training_days, model_params,
                other_params={} ):
    '''
    This function stores the given model inside a folder with the model_id name in the output_path.
    Model is stored in python pickle format, so it is reusable and validation results are stored
    together as a CSV.

    Args:
        output_path (str): The path where the model folder will be stored.
        model_id (str): A unique identifier for the model. The model will be stored inside a folder
            which name is the model_id.
        model (sklearn.model): The trained model object to be stored.
        result_frame (pandas.DataFrame): a pandas dataframe containing the results of the model for
            each validation date.
        training_days (list of str): A list containing all the dates that have been used to train
            the model.
        model_params (dict): A dictionary containing model hyper-parameters to be stored in the
            META file.
        other_params (dict): A dictionary containing other parameters to be appended to model
            meta-data.

    Return:
        str: The path to the folder were all model elements are stored.

    '''
    def storeSubModel( models, output_dir ):
        '''
        This function iterates over the different segmentations of a model and stores the submodel
        in the corresponding folder.

        Args:
            models (dict): A dict containing (distance_range)-> sklearn.model.
            output_dir (str): The model type (origin or destination) path.

        '''
        for dist in models:
            model_dir = os.path.join( output_dir, '{}-{}'.format( dist[0], dist[1] ) )
            if not os.path.exists( model_dir ):
                os.makedirs( model_dir )
            model_path = os.path.join( model_dir, "model.joblib" )
            dump( models.get( dist ), model_path )

    output_dir = os.path.join( output_path, model_id )
    storeSubModel( model, output_dir )
    validation_path = os.path.join( output_dir, "validation.csv" )
    result_frame.to_csv( validation_path, index=False )
    model_meta = {}
    model_meta['training_days'] = training_days
    model_meta['model_id'] = model_id
    model_meta['model_params'] = model_params
    for param in other_params:
        model_meta[param] = other_params.get( param )
    # model_meta['features'] =
    meta_path = os.path.join( output_dir, "model_meta.json" )
    with open( meta_path, 'wb' ) as fout:
        fout.write( ujson.dumps( model_meta ) )
    return output_dir


def loadModel( model_path, model_id ):
    '''
    This function loads the selected model from those stored in the model_path folder.

    Args:
        model_path (str): The path to the model storage directory. This path must point to the
            folder where all models are stored, not the the model folder.
        model_id (str): The identifier of the model. If set to None it is assumed to be directly
            embeded into the model_path variable.

    Returns:
        sklearn.model or dict: A loaded model ready to perform predictions, if distances are
        supported each model is stored as an entry to a dict.

    '''
    if model_id:
        model_path = os.path.join( model_path, model_id )
    all_models = os.listdir( model_path )
    if re.search( '[0-9\.]+-[0-9\.]+', all_models[0] ):
        model = {}
        for submodel in all_models:
            submodel_path = os.path.join( model_path, submodel, "model.joblib" )
            if os.path.exists( submodel_path ):
                rang = tuple( ( float( item ) for item in submodel.split( '-' ) ) )
                model[rang] = load( submodel_path )
    else:
        model_path = os.path.join( model_path, "model.joblib" )
        model = load( model_path )
    return model


def computeScores( train_pair, test_pair, names=( 'train', 'test' ) ):
    '''
    This function takes as input a training pair and a testing pair and returns the equivalent
    scores in a plot-friendly format.

    Args:
        train_pair (tuple of series): A tuple containing in the first position the truth values and
            in the second position the predicted values for the training set.
        test_pair (tuple of series): A tuple containing in the first position the truth values and
            in the second position the predicted values for the testing set.

    Returns:
        dict: A dictionary to be fed to plotRegressionModel function.

    '''
    total_counts = train_pair[0].sum()
    predicted_counts = train_pair[1].sum()
    perc_difference = ( predicted_counts - total_counts ) / predicted_counts * 100
    param_dict = {}
    train_dict = {'r_squared': r2_score( train_pair[0], train_pair[1] ),
                  'rmse': sqrt( mean_squared_error( train_pair[0], train_pair[1] ) ),
                  'mae': mean_absolute_error( train_pair[0], train_pair[1] ),
                  'trip_count_act': total_counts, 'trip_count_pred': predicted_counts,
                  'variation_trip_count': perc_difference }
    param_dict[names[0]] = train_dict
    total_counts = test_pair[0].sum()
    predicted_counts = test_pair[1].sum()
    perc_difference = ( predicted_counts - total_counts ) / predicted_counts * 100
    test_dict = {'r_squared': r2_score( test_pair[0], test_pair[1] ),
                 'rmse': sqrt( mean_squared_error( test_pair[0], test_pair[1] ) ),
                 'mae': mean_absolute_error( test_pair[0], test_pair[1] ),
                 'trip_count_act': total_counts, 'trip_count_pred': predicted_counts,
                 'variation_trip_count': perc_difference}
    param_dict[names[1]] = test_dict
    return param_dict


def plotRegressionValidationCurves( y_train, pred_train, y_test, pred_test, output_path,
                                    validation_metrics, f_size=( 16, 7 ),
                                    names=( 'train', 'test' ) ):
    """
    This function plots the validation curves for the training and testing sets and stores it in the
    path passed as parameter.

    Args:
        y_train (pandas.DataFrame): Truth labels of training set.
        pred_train (pandas.DataFrame): Predicted labels of the training set.
        y_test (pandas.DataFrame): Truth labels of the testing set.
        pred_test (pandas.DataFrame): Predicted labels of the testing set.
        validation_metrics (dict): Dictionary containing validation metrics to be appended in the
            figure title. Expected {tr: {val1: <number>}, test: {val1: <number>}}.
        output_path (str): The path to store the figure in.
        validation_metrics (dict): Dictionary containing metrics for train and test.
        f_size (tuple of int): Figure size.
        names (tuple of str): A tuple containing the names of each of the two sets depicted.

    """

    max_train = max( max( y_train ), max( pred_train ) )
    max_test = max( max( y_test ), max( pred_test ) )

    plt.figure( figsize=f_size )
    plt.subplot( 121 )
    plt.scatter( y_train, pred_train, alpha=0.1 )
    plt.plot( [0, 1e5], [0, 1e5], color="red" )
    plt.grid()
    train_metrics = validation_metrics.get( names[0], {} )
    train_title = "{} Set RMSE: {:.2f}; R^2: {:.2f}".format( names[0].capitalize(),
                                                             train_metrics.get( "rmse", 0 ),
                                                             train_metrics.get( "r_squared", 0 ) )
    plt.title( train_title, fontsize=14 )
    plt.xlabel( "Actual {} labels".format( names[0] ), fontsize=12 )
    plt.ylabel( "Predicted {} labels".format( names[0] ), fontsize=12 )
    plt.xlim( 0, max_train )
    plt.ylim( 0, max_train )
    plt.xticks( fontsize=12 )
    plt.yticks( fontsize=12 )
    plt.subplot( 122 )
    plt.scatter( y_test, pred_test, alpha=0.1 )
    plt.plot( [0, 1e5], [0, 1e5], color="red" )
    plt.grid()
    test_metrics = validation_metrics.get( names[1], {} )
    test_title = "{} Set RMSE: {:.2f}; R^2: {:.2f}".format( names[1].capitalize(),
                                                            test_metrics.get( "rmse", 0 ),
                                                            test_metrics.get( "r_squared", 0 ) )
    plt.title( test_title, fontsize=14 )
    plt.xlabel( "Actual {} labels".format( names[1] ), fontsize=12 )
    plt.ylabel( "Predicted {} labels".format( names[1] ), fontsize=12 )
    plt.xlim( 0, max_test )
    plt.ylim( 0, max_test )
    plt.xticks( fontsize=12 )
    plt.yticks( fontsize=12 )
    plt.savefig( output_path )


def filterDistanceRange( data_path, filter_distance, centroid_file, precomputed_distances=None ):
    '''
    This function filters out any OD pair outside the ranges provided in filter_distance ([lower,
    upper]). The distances are computed as haversine distance between each OD pair. The function is
    capable of working with str paths and pandas dataframes, and will return the same type of object
    that was provided with.

    Args:
        data_path (str or pd.DataFrame): The path to the dataset to be filtered, supports both types
            and returns the same as it was sent.
        filter_distance (list of float): A list containing two floats: lower bound and upper bound.
        centroid_file (str): Path to the file containing all the centroids to be used.
        precomputed_distances (dict): Optional, if set to a dict, it will be used as precomputed
            distances dict. If set to None, a new distance dict will be created.

    Returns:
        str: The path where the reduced dataset is stored.

    '''
    if precomputed_distances:
        distance_dict = precomputed_distances
    else:
        distance_dict = precomputeDistances( centroid_file )
    if isinstance( data_path, str ):
        data_frame = pd.read_csv( data_path )
    else:
        data_frame = data_path
    filtered_df = []
    for _, row in data_frame.iterrows():
        distance = distance_dict.get( ( row.origin, row.destination ), 0 )
        if distance > filter_distance[0]:
            if distance < filter_distance[1]:
                filtered_df.append( row )

    new_data_frame = pd.DataFrame( filtered_df )
    if isinstance( data_path, str ):
        new_data_frame.to_csv( data_path, index=False )
        return data_path
    else:
        return new_data_frame


def splitDistances( data_path, filter_distances, zone_centroid_file ):
    '''
    This function splits the training data into as many chunks as ranges are defined in the
    filter_distances list. Stores each data file under the name of the input file appended with
    the distance range.

    Args:
        data_path (str): The path to the dataset to be split.
        filter_distance (list of list): A list containing for each range another list with two float
            values ([lower,upper]).
        zone_centroid_file (str): The path to a file containing the centroids of the OD matrix
            zoning files associating to each zone ID their longitude and latitude.

    Returns:
        dict: A dictionary containing for each distance range the path in to the file containing
            such data subset.

    '''
    distance_subsets = {}
    print 'Filtering distance'
    for distance in filter_distances:
        distance_data_path = '{}_{}-{}.csv'.format( data_path[:-4], distance[0],
                                                    distance[1] )
        copyfile( data_path, distance_data_path )
        distance_data_path = filterDistanceRange( distance_data_path, distance, zone_centroid_file )
        distance_subsets[tuple( distance )] = distance_data_path
    return distance_subsets


def precomputeDistances( centroid_file, zone_code='ID' ):
    '''
    This function iterates twice over the given centroid_file and computes all possible distances
    that are returned in a dictionary.

    Args:
        centroid_file (str): The file path to the centroid file.
        zone_code (str): The primary key of the zoning.

    Returns:
        dict: A dict containing (origin, dest) -> distance per each OD pair.

    '''
    def haversine( lon1, lat1, lon2, lat2 ):
        """
        Calculate the great circle distance between two points on the earth (specified in decimal
        degrees).

        Args:
            lon1 (float): The longitude of the initial point.
            lat1 (float): The latitude of the initial point.
            lon2 (float): The longitude of the final point.
            lat2 (float): The latitude of the final point.

        Return:
            float: distance between both points in kilometres.

        """
        # convert decimal degrees to radians
        earth_kms = 6371.  # Radius of earth in kilometers is 6371
        lon1, lat1, lon2, lat2 = map( radians, [lon1, lat1, lon2, lat2] )
        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        trig_distance = sin( dlat / 2 ) ** 2 + cos( lat1 ) * cos( lat2 ) * sin( dlon / 2 ) ** 2
        radial_dist = 2 * asin( sqrt( trig_distance ) )
        total_kms = earth_kms * radial_dist
        return total_kms

    all_centroids = pd.read_csv( centroid_file )
    result_dict = {}
    for _, row in all_centroids.iterrows():
        for _, row2 in all_centroids.iterrows():
            if row[zone_code] == row2[zone_code]:
                result_dict[( row[zone_code], row2[zone_code] )] = 0
                continue
            result_dict[( row[zone_code], row2[zone_code] )] = \
                haversine( row.longitude, row.latitude, row2.longitude, row2.latitude )
    return result_dict


# MAIN FUNCTIONALITY WRAPPERS #####################################################################


def performPrediction( data_frame, model_dir, model_id, output_path, num_weeks, centroid_file ):
    '''
    This function executes a prediction routine over the dataset provided. For prediction, the model
    is selected, executed (over the corresponding pipeline) and the predictions (in trips) per OD
    pair, date and time frame are returned.

    Args:
        data_frame (pd.DataFrame): A dataframe containing the input features aligned along with the
            fields origin, destination, period(range) and date.
        model_dir (str): The path to the model directory.
        model_id (str): The identifier of the model to be used.
        output_path (str): The path to the directory where predictions will be stored.
        num_weeks (int): The number of weeks integrated in the training set to divide by them.
        centroid_file (str): Path to the centroid file needed for distance filtering.

    Returns:
        pd.DataFrame: Dataframe containing the trip estimations per OD pair, time period and date.

    '''

    model = loadModel( model_dir, model_id )
    if isinstance( model, dict ):
        all_results = []
        for filter_distance in model:
            reduce_frame = filterDistanceRange( data_frame, filter_distance, centroid_file )
            if reduce_frame.shape[0] == 0:
                continue
            features = reduce_frame.drop( ['origin', 'destination', 'date'], axis=1 )
            features = features.fillna( 0 )
            predictions = model.get( filter_distance ).predict( features )
            predictions = predictions / num_weeks
            result_df = reduce_frame[['origin', 'destination', 'date']]
            result_df['predicted_trips'] = predictions
            all_results.append( result_df )
        result_df = pd.concat( all_results, axis=0 )
    else:
        features = data_frame.drop( ['origin', 'destination', 'date'], axis=1 )
        predictions = model.predict( features )
        result_df = data_frame[['origin', 'destination', 'date']]
        result_df['predicted_trips'] = predictions
        # Truncate predictions below 0
    result_df.loc[result_df['predicted_trips'] < 0, 'predicted_trips'] = 0
    result_df.to_csv( output_path, index=False )


def trainModelRoutine( data_path, output_path, model_type='forest', model_params=None,
                       model_id=None ):
    '''
    This function loads the training data, creates and trains the selected prediction model,
    performs its validation and stores the functional model into the given output_path.

    Args:
        data_path (str): The path to the training dataset.
        output_path (str): The path to the directory where the model will be stored.
        model_type (str): The model type to be trained. Currently supported: linear, tree and
            forest.
        model_params (dict): A set of hyper-parameters for the selected algorithms. If it is not
            set, the model is trained with default hyper-parameters.
        model_id (str): A specified model identifier. If not set, it is assigned based on the
            date and time at training time.

    '''
    global TEST_SIZE
    plot_dict = {}
    train_dataset, test_dataset = splitDataset( data_path, TEST_SIZE )
    print( 'Training' )
    model, result_train, pred_train = trainAggregatedModel( train_dataset, model_type,
                                                            model_params )

    plot_dict['train'] = result_train
    print( 'Testing' )
    result_dict_test, pred_test = performValidation( model, test_dataset )

    plot_dict['test'] = result_dict_test.get( 'test' )

    df_list = []
    for frame in plot_dict:
        current = plot_dict.get( frame, {} )
        for item in current:
            item['dataset'] = frame
            df_list.append( item )

    result_frame = pd.DataFrame( df_list )
    training_days = pd.read_csv( data_path.values()[0] )['date'].drop_duplicates().values.tolist()
    other_params = {}
    other_params['distance_ranges'] = model.keys()
    other_params['model_type'] = model_type
    if not model_id:
        model_id = time.strftime( '%Y%m%d%H%M' )
    model_path = storeModel( output_path, model_id, model, result_frame, training_days,
                             model_params, other_params=other_params )

    plot_path = os.path.join( model_path, 'validation_plot.png' )
    score_dict = computeScores( pred_train, pred_test, names=( 'train', 'test' ) )
    plotRegressionValidationCurves( pred_train[0], pred_train[1], pred_test[0], pred_test[1],
                                    plot_path, score_dict )

    validation_score_path = os.path.join( model_path, 'validation_scores.json' )
    with open( validation_score_path, 'wb' ) as fout:
        fout.write( ujson.dumps( score_dict ) )

    return model_path


if __name__ == '__main__':

    if len( sys.argv ) < 2:
        print( 'ERROR. Usage: python {} <base_path> <num_weeks_train> [<output_path>]'
               .format( sys.argv[0] ) )
        exit( -1 )
    base_path = sys.argv[1]
    num_weeks = int( sys.argv[2] )  # number of weeks that have been aggregated for training.
    if len( sys.argv ) == 4:
        output_path = sys.argv[3]
    else:
        output_path = base_path
    # Distance ranges are set to inter-quantile ranges of Madrid's BiciMAD service. Different
    # numbers of ranges are directly supported.
    distance_ranges = [[0, 1.256], [1.256, 1.893], [1.893, 2.734], [2.734, 10]]
    # File names are preset
    data_path = os.path.join( base_path, 'train_data.csv' )
    centroid_path = os.path.join( base_path, 'centroids.csv' )
    # STEP 1: TRAINING
    # Training dataset is automatically split by distance ranges.
    training_distances = splitDistances( data_path, distance_ranges, centroid_path )
    model_path = trainModelRoutine( training_distances, output_path, model_type='tree' )
    # STEP 2: PREDICTION
    # ~Prediction data names are preset
    prediction_path = os.path.join( base_path, 'prediction_data.csv' )
    predicted_path = os.path.join( base_path, 'predicted_data.csv' )
    # For prediction, load the entire prediction set and the system will split by distances.
    prediction_frame = pd.read_csv( prediction_path )
    performPrediction( prediction_frame, model_path, None, predicted_path, num_weeks,
                       centroid_path )
