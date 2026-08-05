##################################################
'''
src.exploration.split_unit.determine_per_region_coverage_growth

input: panel.parquet, ref_geo.parquet
purpose: determine the rating coverage development in total and
        per region.
output:        
        
        output: Number of new entities per year globally (excluding 2009):
        year
        2009.0    2187
        2010.0     614
        2011.0     193
        2012.0     150
        2013.0     142
        2014.0     173
        2015.0     251
        2016.0     224
        2017.0     477
        2018.0     702
        2019.0     949
        2020.0    1190
        2021.0     972
        2022.0    1266
        2023.0     126
        2024.0      90
        2025.0       9

        distinct entitiy coverage ratios for each year 
        min_year  compl_ratio
        2009.0    0.411765          1
                0.764706          1
                0.875             2
                0.888889          1
                0.9               1
                0.909091          1
                0.923077          1
                0.928571          1
                0.9375            2
                0.941176         12
                1.0            2164
        2010.0    0.333333          1
                0.75              1
                0.9375            6
                1.0             606
        2011.0    0.933333          2
                1.0             191
        2012.0    0.357143          1
                0.928571          1
                1.0             148
        2013.0    0.923077          1
                1.0             141
        2014.0    0.833333          1
                0.916667          1
                1.0             171
        2015.0    1.0             251
        2016.0    0.7               1
                0.888889          1
                1.0             222
        2017.0    0.888889          3
                1.0             474
        2018.0    0.5               1
                0.75              2
                0.875             7
                1.0             692
        2019.0    0.6               1
                0.666667          1
                0.714286          1
                0.8               1
                0.857143          3
                1.0             942
        2020.0    0.666667          1
                0.75              1
                0.833333          7
                1.0            1181
        2021.0    0.8              21
                1.0             951
        2022.0    0.75             14
                1.0            1252
        2023.0    1.0             126
        2024.0    1.0              90
        2025.0    1.0               9
'''
##################################################

import pandas as pd
import config as con

# load data
panel = pd.read_parquet(con.PANEL)
ref_geo = pd.read_parquet(con.REF_GEOGRAPHY)

# slice relevant vars
df_org = panel[['orgpermid', 'year']]
df_geo = ref_geo[['orgpermid', 'lvl3permid']]

# extract entitiy entry year per orgpermid globally 
print(f'''number of new entities per year globally (excluding 2009):''')
print(df_org.groupby('orgpermid', dropna=False).year.min().value_counts().sort_index())
print(f'''\ntotal entities (sanity check):''')
print(df_org.groupby('orgpermid', dropna=False).year.min().value_counts().sort_index().sum())

# extract entitiy entry year per orgpermid per region
df_entity_year = df_org.groupby('orgpermid').year.min().reset_index()
df_merged = df_entity_year.merge(df_geo, on="orgpermid", how='left')
print(df_merged.groupby(['lvl3permid', 'year'], dropna=False).size())
print(f'''\ntotal entities (sanity check):''')
print(df_merged.groupby(['lvl3permid', 'year'], dropna=False).size().sum())
print(f'''\n''')

# calculate completeness ratio by entry year
df_cohort_compl = df_org.groupby(['orgpermid', 'year'], dropna=False).size().reset_index()

df_cohort_compl = df_cohort_compl.groupby('orgpermid').agg(
    total_entries=('year', 'count'),  
    min_year=('year', 'min'),
    max_year=('year', 'max')
).reset_index()

df_cohort_compl['compl_ratio'] = df_cohort_compl['total_entries'] /  (df_cohort_compl['max_year']-(df_cohort_compl['min_year']-1))
print(f'''\ndistinct entitiy coverage ratios for each year ''')
print(df_cohort_compl.groupby(['min_year', 'compl_ratio']).size())
print(df_cohort_compl.groupby(['min_year', 'compl_ratio']).size().sum())