##################################################
'''
src.exploration.final_panel.determine_nmb_of_clusters

input: ref_cluster_keys.parquet
purpose: determine the number of clusters based on the cluster key
        assignment
output: clusters-based on ultpar: 8865
        clusters-based on immpar: 847
        singleton-clusters: 3
'''
##################################################

import pandas as pd
import config as con

# load data
cluster_keys = pd.read_parquet(con.REF_CLUSTER_KEYS)

# slice data frame
print(cluster_keys.groupby('key_source').size())
print(cluster_keys.groupby('key_source').size().sum())
