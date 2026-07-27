##################################################
'''
Create the cluster assigment that is needed for the 
stratified split. 

Read parent_ent_type.parquet + ref_parents.parquet.
Produce ref_cluster_keys.parquet with orgpermid,
cluster_key, parent_typecode, key_source.

Cast all ID columns to a non-float type. 

The assignment logic is build from two masks.

Close with four assertions:
i) cluster_key has exactly 3 nulls before the singleton fill
ii) the ultimate-branch and immediate-branch key sets
    are disjoint 
iii) entity count in == entity count out
iv) every orgpermid in the panel has exactly one cluster_key
'''
##################################################


