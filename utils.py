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


### feature cut-off helpers ---------------------------------------------------------

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

### reporting metrics helpers ---------------------------------------------------------

def per_region_metrics(y_true: pd.Series, *, y_pred: pd.Series, geoID: pd.Series) -> tuple[pd.DataFrame, float]:
    '''Calculate the per region RMSE and R².
    input: pd.Series: pr y,
        pd.Series: target y,
        pd.Series: pd.Series: geoID (lvl3permid indicators)
    output: dataframe of per region rmse, r^2, SSR and the sum over all regional SSRs
    '''
    # assert index allignment
    assert y_true.index.equals(y_pred.index), "index misallignment of target&predition"
    assert y_true.index.equals(geoID.index), "index misallignment of target&geoID"
    assert y_true.notna().all() and y_pred.notna().all(), "NaN in target or prediction"

    # calculate global denominator
    denominator = len(y_true) * np.var(y_true, ddof=0)
    # per region nominator and RMSE
    ## cut regional slices
    ### merge target and pred with geoIDs
    merged_data = pd.concat([y_true, y_pred, geoID], axis=1)
    merged_data.columns = ["y_true", "y_pred", "lvl3permid"]
    regions = merged_data['lvl3permid'].unique().tolist()
    assert set(regions) == set([*con.TIER1_REGS, *con.TIER2_REGS]), f"expected regions: {[*con.TIER1_REGS, *con.TIER2_REGS]} , saw {regions}"

    ### iterate over regions
    results = {}
    nominators = []
    for region in regions:
        data = merged_data.loc[merged_data['lvl3permid'] == region]
        y_pred_r = data['y_pred']
        y_target_r = data['y_true']
        n_r = len(y_target_r)
        ssr_r = np.sum((y_pred_r - y_target_r)**2)
        rmse_r = np.sqrt(ssr_r / n_r)
        # region specific denominator
        denominator_r = n_r * np.var(y_target_r, ddof=0)
        nominators.append(ssr_r)
        results.update({region: [rmse_r, ssr_r, n_r, denominator_r]})
    global_sum_residuals = np.sum(nominators)
    df_metrics = pd.DataFrame.from_dict(results, orient='index', columns=['rmse_r', 'ssr_r', 'n_r', 'denominator_r'])
    df_metrics.index.name = 'lvl3permid'
    # calculate R² against the shared global scale, per observation
    df_metrics['r_sq'] = 1 - (df_metrics['ssr_r'] / df_metrics['n_r']) / (denominator / len(y_true))
    # calculate R² against the region own mean
    df_metrics['r_sq_reg'] = 1 - df_metrics['ssr_r'] / df_metrics['denominator_r']
    return (df_metrics, global_sum_residuals)


def pooled_metrics(y_true: pd.Series, *, y_pred: pd.Series) -> tuple[float, float, float]:
    # assert index alligment
    assert y_true.index.equals(y_pred.index), "index misallignment of target&predition"
    assert y_true.notna().all() and y_pred.notna().all(), "NaN in target or prediction"
    # calculate global denominator
    denominator = len(y_true) * np.var(y_true, ddof=0)
    # nominator and RMSE
    ssr = np.sum((y_pred - y_true)**2)  # sum of squared errors in region r
    rmse = np.sqrt(ssr / len(y_true))
    r_sqrd = 1 - ssr / denominator
    return (rmse, r_sqrd, ssr)


def macro_average_metrics(per_reg_metrics: tuple[pd.DataFrame, float]) -> tuple[float, float, float, float]:
    df_metrics = per_reg_metrics[0]
    average_rmse = df_metrics['rmse_r'].mean()
    average_r_sqrd = df_metrics['r_sq'].mean()
    average_r_sqrd_reg = df_metrics['r_sq_reg'].mean()
    # quadratic mean: comparable to the pooled rmse, free of the sqrt averaging bias
    average_rmse_q = np.sqrt((df_metrics['rmse_r']**2).mean())
    return (average_rmse, average_r_sqrd, average_r_sqrd_reg, average_rmse_q)


def assert_ss_res_decomposition(per_region_metrics: tuple[pd.DataFrame, float], pooled_metrics: tuple[float, float, float]) -> None:
    # same non-negative sum, only the accumulation order differs (worst observed 3e-16);
    # the np.isclose default rtol=1e-5 would mask a genuine partition error
    assert np.isclose(per_region_metrics[1], pooled_metrics[2], rtol=1e-9, atol=1e-12), "SSE global vs summed SSE per region do not match"


def report_metrics(per_region_metrics: tuple[pd.DataFrame, float],
                   pooled_metrics: tuple[float, float, float],
                   macro_average_metrics: tuple[float, float, float, float],
                   tier1_regions: list) -> tuple[pd.DataFrame, tuple[float, float], float, tuple[float, float, float, float], float, float, float, float]:
    df_region_report = per_region_metrics[0].copy()
    df_region_report = df_region_report.loc[tier1_regions]
    df_region_report['rmse_100'] = df_region_report['rmse_r'] * 100
    pooled_metrics_tupel = (pooled_metrics[0], pooled_metrics[1])
    pooled_rmse_100 = pooled_metrics_tupel[0] * 100
    macro_average_metrics_100 = macro_average_metrics[0] * 100
    macro_average_metrics_q_100 = macro_average_metrics[3] * 100
    # equal region weights vs size weights: positive -> large regions perform worse
    regional_bias_gap = macro_average_metrics[1] - pooled_metrics[1]
    # same comparison in target units: positive -> small regions perform worse
    regional_bias_gap_rmse = macro_average_metrics[3] - pooled_metrics[0]
    print(f'regional performance metrics:')
    print(f'{df_region_report}\n')
    print(f'pooled performance metrics:')
    print(f'{pooled_metrics_tupel}\n')
    print(f'pooled rmse*100:')
    print(f'{pooled_rmse_100}\n')
    print(f'average performance (equal region weights):')
    print(f'{macro_average_metrics}\n')
    print(f'average rmse (equal region weights)*100 :')
    print(f'{macro_average_metrics_100}\n')
    print(f'quadratic mean rmse (equal region weights)*100 :')
    print(f'{macro_average_metrics_q_100}\n')
    print(f'regional bias gap (macro r_sq - pooled r_sq):')
    print(f'{regional_bias_gap}\n')
    print(f'regional bias gap rmse (quadratic mean rmse - pooled rmse):')
    print(f'{regional_bias_gap_rmse}\n')

    return df_region_report, pooled_metrics_tupel, pooled_rmse_100, macro_average_metrics, macro_average_metrics_100, macro_average_metrics_q_100, regional_bias_gap, regional_bias_gap_rmse