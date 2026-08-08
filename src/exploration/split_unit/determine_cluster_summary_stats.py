##################################################
'''
src.exploration.split_unit.determine_nmb_of_clusters

input: panel.parquet, ref_cluster_keys.parquet
purpose: determine the number of clusters based on the cluster key
        assignment,
        determine number of singleton clusters
output: entities clustered based on ultpar: 8865
        entities clustered based on immpar: 847
        entities with no-parent clusters: 3


        number of singleton clusters in panel:7898
'''
##################################################

import pandas as pd
import config as con

# load data
panel = pd.read_parquet(con.PANEL)
cluster_keys = pd.read_parquet(con.REF_CLUSTER_KEYS)

# slice data frame
print(cluster_keys.groupby('key_source').size())
print(cluster_keys.groupby('key_source').size().sum())

# dertermin number of singleton clusters
df_org = panel[['orgpermid']]
df_clstky = cluster_keys[['orgpermid', 'cluster_key']] 
df_clorg = df_org.merge(df_clstky, on="orgpermid", how="left") 
df_clorg = df_clorg.drop_duplicates(subset=['orgpermid'])
df_clorg = pd.DataFrame({
    "n_ents_in_clust": df_clorg.groupby('cluster_key').size()
})
print(f'''number of singleton clusters in panel:''')
print(len(df_clorg.query('n_ents_in_clust==1')))

# determine number of unique clusters
print(f'''total number of clusters in panel:{df_clstky['cluster_key'].nunique()}''')


