import json
import pandas as pd 
import numpy as np 

def access_time(df):
    ts = sum(df.access_time)
    tsavg = ts/df.shape[0]
    return ts, tsavg

def waiting_time(df):
    tw = sum(df.waiting_time)
    twavg = tw/df.shape[0]
    return tw, twavg

def trip_time(df):
    tt = sum(df.trip_time)
    ttavg = tt/df.shape[0]
    return tt,ttavg


def accept_rate(req_list):
    tot_acc = sum(req_list)
    perc_acc = tot_acc/len(req_list)

    return tot_acc, perc_acc

def maintenance(df, mcost):
    pass 

def fuel_cost(df, cost):
    pass

def outsourcing_cost(df, cost):
    pass 

def fixed_cost(df):
    pass