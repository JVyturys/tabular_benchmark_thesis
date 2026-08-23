##################################################
'''
Define functions for procduction branch.
'''
##################################################
import pandas as pd
import numpy as np
import config as con
import math

def variance_loss(x):
    '''determine the changes in variance (VL) for all variables of a dataset 
        that occur due to the imputation of their columns with the respective median.

        input: X (n,v) data set containing numerical columns
        output: a data frame containing:
            variance loss: VL
            missingness fraction: p and (1-p)
            sample size of observed values: n_o
            sample size: n
                
    '''

    # define placeholders for results per column
    variables = [] 
    vl_values = [] 
    p_fractions = []
    minus_p_fractions = [] 
    n_obs = []
    means = []
    medians = []
    variances = []


    # determine sample size
    n = len(x)

    for column in x.columns:
        if x[column].isna().sum() == n:
                vl = 1

                # update results
                variables.append(column)
                vl_values.append(vl)
                p_fractions.append(1)
                minus_p_fractions.append(0)
                n_obs.append(0)
                means.append(float('nan'))
                medians.append(float('nan'))
                variances.append(float('nan'))
                
        
        else:
            # calculate missingness fraction p and (1-p)
            n_o = len(x[column].dropna())
            p = 1- (n_o/n)
    
            # filter to observed data 
            x_o = x[column].dropna()
    
            # calculate median and mean
            mean_x_o = np.mean(x_o)
            median_x_o = np.median(x_o)
    
            # calculate variance
            variance_x_o = np.var(x_o)

            # calculate variance loss 
            if math.isclose(variance_x_o, 0):
                print(f"WARINGIN: {column} has a has a degenerate variance, VL set to 1")
                vl = 1
            elif math.isnan(variance_x_o):
                 print(f"WARINGIN: {column} has a var of NaN, setting VL({column}) == 1")
                 vl = 1
            else:
                 vl = p - p*(1-p)*((mean_x_o-median_x_o)**2)/(variance_x_o)

            variables.append(column)
            vl_values.append(vl)
            p_fractions.append(p)
            minus_p_fractions.append((1-p))
            n_obs.append(n_o)
            means.append(mean_x_o)
            medians.append(median_x_o)
            variances.append(variance_x_o)

    df_vl = pd.DataFrame({
         "variable":variables,
         "variance_loss":vl_values,
         "p":p_fractions,
         "1-p":minus_p_fractions,
         "n_observed":n_obs,
         "mean":means,
         "median":medians,
         "variance":variances
    })

    return df_vl