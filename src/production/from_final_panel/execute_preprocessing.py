##################################################
'''
src.production.from_final_panel.execute_preprocessing

input: panel_parquet,  split.parquet
purpose: define the preprocessing constants for feature selection 
        and scaling using the median-imputation induced variance loss.
        
output: pre_processing_constants.parquet,

        dropped variables due to degenereate variances:
        {item4450, item3448, , item4452 , item4799, item3257}

        dropped variables due to post-immputation variances > loss cut-off threshold (0.06939):
         ['item4057', 'item3499', 'item18188', 'item4150', 'item1084', 'item1254',
          'item1269', 'item7011', 'item18183', 'item8256', 'item18184', 'item8351', 'item8406', 'item1155',
            'item1801', 'item4148', 'item1267', 'item1266', 'item8346', 'item8906', 'item1149', 'item4053',
              'item4840', 'item1306', 'item3260', 'item1253', 'item18280', 'item2513', 'item1352', 'item1503',
                'item4149', 'item4651', 'item3261', 'item18324', 'item2514', 'item2515', 'item18140',
                  'item2654', 'item2655', 'item18274', 'item18299', 'item4056', 'item2516', 'item2517',
                    'item4821', 'item18352', 'item4058', 'item3493', 'item2502', 'item2503', 'item1157',
                      'item4052', 'item1301', 'item2510', 'item18571', 'item2511', 'item2507', 'item2512',
                        'item18852', 'item2509', 'item1268', 'item2508', 'item1204', 'item18286', 'item18574',
                          'item4055', 'item3491', 'item1154', 'item18293', 'item18408', 'item18187', 'item1152',
                            'item18275', 'item18224', 'item18854', 'item1302', 'item18226', 'item2504', 'item2505',
                              'item2653', 'item18851', 'item18225', 'item2506', 'item18189', 'item18185', 'item18215',
                                'item18353', 'item1153', 'item18175', 'item18572', 'item18165', 'item4892',
                                  'item4891', 'item18575', 'item1156', 'item3494', 'item18159', 'item18173',
                                    'item18172', 'item18170', 'item18171', 'item18166', 'item3450', 'item4054',
                                      'item18168', 'item18167', 'item18853', 'item1351', 'item18065', 'item3490',
                                        'item1265', 'item3449']
        

'''
##################################################
import pandas as pd
import config as con
from utils import variance_loss as vl
from utils import vl_cut_off as vlc
import numpy as np



# load data
panel = pd.read_parquet(con.PANEL)
panel = panel.drop(columns=['year', 'esg_combined_score'])
split = pd.read_parquet(con.SPLIT)

# slice fit partition
fit = panel.merge(split, on='orgpermid', how='left')
fit = fit.query('partition=="fit"')
assert fit['partition'].unique() == ['fit'], "conatminated fit partition"
fit_features = fit.drop(columns=['orgpermid', "partition"])

# determine variance loss
cuoff_metrics, deg_var_cols = vl(fit_features)
print(f"{len(deg_var_cols)} columns with degenerate variances identified")
cuoff_metrics = cuoff_metrics.loc[cuoff_metrics['variance_loss'].isna()==False].sort_values(by='variance_loss')

# determine kneedle
co_idx, co_vl = vlc(cuoff_metrics['variance_loss'])
print(co_vl)

prepros_constants = cuoff_metrics.query('variance_loss <= @co_vl').copy()
dropped_metric = cuoff_metrics.query('variance_loss > @co_vl').copy()
dropped_variables = dropped_metric['variable'].tolist()

assert len(dropped_variables) + len(deg_var_cols) + len(prepros_constants) == 242, "variables (cut+drop+keep) do not add up to 242 "


prepros_constants = prepros_constants[['variable', 'mean', 'median', 'variance']]
prepros_constants['variance'] = np.sqrt(prepros_constants['variance'])
prepros_constants = prepros_constants.rename(columns={'variance':"std"})
prepros_constants.to_parquet(con.PRE_PROS_CONTS)

