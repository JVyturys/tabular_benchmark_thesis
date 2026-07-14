##################################################
'''
This script visualizes:
i) lvl3permid regions of in panel included entities on the worldmap
ii) frequencies of observations (entities) per region
'''
##################################################

import pandas as pd
import geopandas as gpd
import config as con
import matplotlib.pyplot as plt
import utils.custom_colormaps
from matplotlib.colors import PowerNorm
from matplotlib.ticker import ScalarFormatter

# Initializing paths
data_path = con.FINAL_PANEL_PARQUET
output_path = con.PROJECT_ROOT / 'data' / "raw"
results_path = con.RESULTS_DIR/ 'joined_panel'

df = pd.read_parquet(data_path)
geo = pd.read_csv(con.PROJECT_ROOT / 'data' / 'raw' / 'geo_ref.csv')

# Mapping IDs
geo_subset = geo[['orgpermid', 'lvl3permid']]
df_subset = df[['orgpermid']]
df_merged = df_subset.merge(geo_subset, on="orgpermid", how='left')

# counting frequency for lvl3permids
reg_counts = df_merged['lvl3permid'].value_counts().reset_index()
reg_counts.columns = ['lvl3permid', 'observation_count']
# print(reg_counts.head())

# load ZIP from Natural Earth
url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(url)

# construction geo_lookup
geo_lookup = geo[['lvl3permid','lvl5isocntry']]
geo_lookup = geo_lookup.drop_duplicates()

# merging count and lookup data
reg_count_enriched = reg_counts.merge(geo_lookup, on='lvl3permid', how='left')

# merging with spatial data
world_merged = world.merge(reg_count_enriched, left_on='ISO_A2', right_on='lvl5isocntry')
regional_map = world_merged.dissolve(by='lvl3permid')

# plot
## foundational gray map
fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))

world.plot(ax=ax1,
          color='lightgray',
          edgecolor='white')

## regional count map
regional_map.plot(ax=ax1,
                 column='observation_count',
                 cmap='viridis',
                 edgecolor='black',
                 linewidth=0.5,
                 norm=PowerNorm(gamma=0.5),
                 legend='True',
                 legend_kwds={'format':ScalarFormatter(),
                              'ticks': [10, 25, 50, 100,
                                        200, 400, 800, 1600, 
                                        3200, 6400, 10000, 20000]})


# plotting new regions
## foundation in gray
fig, ax2 = plt.subplots(1,1, figsize=(15,10))
world.plot(ax=ax2,
           color='lightgray',
           edgecolor='white')

## predissolved map to show countries within regions

world_merged.plot(ax=ax2,
                  column='lvl3permid',
                  categorical=True,
                  cmap='high_contrast_tab20',
                  edgecolor='black',
                  linewidth=0.5,
                  legend=True)

plt.title("Entities Grouped by Subcontinental Region")
plt.show()
