##################################################
'''
Define functions for procduction branch.
'''
##################################################
import pandas as pd
import numpy as np
import config as con
import math
import matplotlib.pyplot as plt

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

    # placeholder for degenerate variance columns
    deg_var_cols = []

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
            variance_x_o = np.var(x_o, ddof=0)

            # calculate variance loss 
            if math.isclose(variance_x_o, 0):
                print(f"WARINGIN: {column} has a has a degenerate variance, VL set to 'NaN'")
                vl = float('nan')
                deg_var_cols.append(column)
            elif math.isnan(variance_x_o):
                 print(f"WARINGIN: {column}'s variance not defined, setting VL({column}) == 'NaN'")
                 vl = float('nan')
                 deg_var_cols.append(column)
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

    return df_vl, deg_var_cols

def vl_cut_off(x):
    '''
    Determine set of column indicies to be dropped from the dataset based on their variance losses.
    Determine the set by:
        - plotting the VL in ascending order against column indicies,
        - calculating a straight diagonal "baseline" connecting the start and end points of the curve,
        - finding the point of maximum distance between the curve and the baseline.
    input: - (v,2) data frame sorted ascendingly by column 1,
            column 1: VL values, column 2: variable indicies / names

    output: 
        - tupel (x, VL(x)) that determines the cutoff
        - plotted "kneedle" graph
    '''

    # determine linear function between start and endpoint
    ## set start & end points

    x = x.reset_index(drop=True)

    n_points = len(x)
    x_indices = np.arange(n_points)

    vl_start = x.iloc[0]
    vl_end = x.iloc[-1]

    # determine linear function between start and endpoint
    m = (vl_end - vl_start) / (n_points -1)
    lin_func = m * x_indices + vl_start

    # determine cut off idx
    distance_to_curve = pd.Series(lin_func-x)
    max_dist_vl = max(distance_to_curve)

    max_dist_idx = distance_to_curve.argmax() 
    kneedle_curve_val = x.iloc[max_dist_idx]
    kneedle_lin_val = lin_func[max_dist_idx]
        


    # plot curves
    plt.style.use('seaborn-v0_8-whitegrid')

    color_curve = "#1f4e79"
    color_linear = "#a6a6a6"
    color_kneedle = "#d9534f"

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        x_indices, 
        x, 
        label="Variance Loss", 
        color=color_curve, 
        linewidth=1.5,
        zorder=3
    )

    ax.plot(
        x_indices,
        lin_func,
        label="Linear Connection",
        linestyle="--",
        color=color_linear,
        linewidth=1.0,
        zorder=2
    )

    ax.vlines(
        x=max_dist_idx,
        ymin=min(kneedle_curve_val, kneedle_lin_val),
        ymax=max(kneedle_curve_val, kneedle_lin_val),
        color=color_kneedle,
        linestyle=":",
        linewidth=1.5,
        label=f"Cut-off Point",
        zorder=4
    )

    ax.scatter(
        max_dist_idx, 
        kneedle_curve_val, 
        color=color_kneedle, 
        s=30, 
        zorder=5, 
        label=f"VL: {x.iloc[max_dist_idx]:.3f}\nDist: {max_dist_vl:.3f}"
    )

    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Variance Loss")
    ax.set_title(
        f"Variance Loss Based Feature Selection\n"
        f"Number of excluded fetures: {(x>x.loc[max_dist_idx]).sum()}", 
        fontweight='bold'
    )

    ax.legend(fontsize=9, frameon=True, fancybox=True, shadow=False)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(con.VIZ_CUTOFF, dpi=600, bbox_inches='tight')
    plt.show()

    

    return max_dist_idx, x.loc[max_dist_idx]

