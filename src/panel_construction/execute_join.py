##################################################################
# This script executes the join of the distinct wrds tables,based on
# the results of the data exploration. 
##################################################################

import wrds
import config as con
import log_config as ln
import pandas as pd
from utils.reporting import AnalysisLogger as Al

log_path = con.RESULTS_DIR / "exploration" /"raw_data"


db = wrds.Connection(wrds_username = ln.wrds_log)

# 1. Master entity-year list
query_1 = f"""
    SELECT DISTINCT p.orgpermid,
                    p.worldscopecmpid,
                    e.year
    FROM tr_common.permorgref p 
    JOIN tr_esg.wrds_ref_esg e
        ON p.orgpermid = e.orgpermid
    WHERE p.typecode = 'COM'
        AND p.worldscopecmpid IS NOT NULL
        AND e.fieldname = 'ESGCombinedScore'
        AND e.valuescore IS NOT NULL
        AND e.value IS NOT NULL
        AND e.year BETWEEN 2009 AND 2025
    ORDER BY orgpermid
"""
output = db.raw_sql(query_1)

# log_cmp_yr = Al(output_path=log_path, filename="company_year_list.txt")
# log_cmp_yr.section("Number of distinct company-year observations in the panel.")
# log_cmp_yr.log(query_1)
# log_cmp_yr.log(output)

## Making sure there are no duplicates within the company-year-financials table
# query_2 = """
# WITH skeleton AS (
#     SELECT DISTINCT 
#         p.orgpermid, 
#         p.worldscopecmpid, 
#         e.year
#     FROM tr_common.permorgref p
#     JOIN tr_esg.wrds_ref_esg e ON p.orgpermid = e.orgpermid
#     WHERE p.typecode = 'COM'
#     AND p.worldscopecmpid IS NOT NULL
#     AND e.fieldname = 'ESGCombinedScore'
#     AND e.valuescore IS NOT NULL
#     AND e.year BETWEEN 2009 AND 2025
# )
# SELECT s.orgpermid, s.year, COUNT(DISTINCT f.code) as n_codes
# FROM skeleton s
# JOIN tr_worldscope.wrds_ws_funda f
#     ON s.worldscopecmpid = f.item6105
#     AND s.year = f.year_
# WHERE f.freq = 'A'
# GROUP BY s.orgpermid, s.year
# HAVING COUNT(DISTINCT f.code) > 1
# """
# slice_ws = db.raw_sql(query_2)
# print(f"Data frame with duplicate entires:\n {slice_ws}")

# 2. Defining variable set
## Loading the selected worldscope variable codes set 
ws_variables = pd.read_csv(con.RESULTS_DIR / "exploration" / "raw_data" / "ws_variables_final.csv") 
ws_items = ws_variables.iloc[:,0].tolist()

## transforming the codes back to wrds_ws_funda column names
ws_items_col = [f"item{i:.0f}" for i in ws_items]

## defining SELECT clause for join operation in SQL
feature_cols = ','.join(f"f.{item}" for item in ws_items_col)
 
# 3. Executing join for panel data

# query_4 = f"""
# WITH skeleton AS (
#     SELECT DISTINCT 
#         p.orgpermid, 
#         p.worldscopecmpid, 
#         e.year,
#         e.valuescore as esg_combined_score
#     FROM tr_common.permorgref p
#     JOIN tr_esg.wrds_ref_esg e ON p.orgpermid = e.orgpermid
#     WHERE p.typecode = 'COM'
#     AND p.worldscopecmpid IS NOT NULL
#     AND e.fieldname = 'ESGCombinedScore'
#     AND e.valuescore IS NOT NULL
#     AND e.year BETWEEN 2009 AND 2025
# )
# SELECT s.orgpermid, s.year, s.esg_combined_score, {feature_cols}
# FROM skeleton s
# JOIN tr_worldscope.wrds_ws_funda f
#     ON s.worldscopecmpid = f.item6105
#     AND s.year = f.year_
# WHERE f.freq = 'A'
# """
# panel = db.raw_sql(query_4)

# 4. Comparing entity converage before and after the join:
## Number of entities after join:
# query_5 = f"""
# WITH skeleton AS (
#     SELECT DISTINCT 
#         p.orgpermid, 
#         p.worldscopecmpid, 
#         e.year,
#         e.valuescore as esg_combined_score
#     FROM tr_common.permorgref p
#     JOIN tr_esg.wrds_ref_esg e ON p.orgpermid = e.orgpermid
#     WHERE p.typecode = 'COM'
#     AND p.worldscopecmpid IS NOT NULL
#     AND e.fieldname = 'ESGCombinedScore'
#     AND e.valuescore IS NOT NULL
#     AND e.year BETWEEN 2009 AND 2025
# ),
# panel AS (
#     SELECT s.orgpermid, s.year, s.esg_combined_score
#     FROM skeleton s
#     JOIN tr_worldscope.wrds_ws_funda f
#         ON s.worldscopecmpid = f.item6105
#         AND s.year = f.year_
#     WHERE f.freq = 'A'
# )
# SELECT COUNT(DISTINCT orgpermid) as n_entities
# FROM panel
#  """

# ## Number of entities before join.
# query_6 = f"""
# WITH skeleton AS (
#     SELECT DISTINCT 
#         p.orgpermid, 
#         p.worldscopecmpid, 
#         e.year,
#         e.valuescore as esg_combined_score
#     FROM tr_common.permorgref p
#     JOIN tr_esg.wrds_ref_esg e ON p.orgpermid = e.orgpermid
#     WHERE p.typecode = 'COM'
#     AND p.worldscopecmpid IS NOT NULL
#     AND e.fieldname = 'ESGCombinedScore'
#     AND e.valuescore IS NOT NULL
#     AND e.year BETWEEN 2009 AND 2025
# ),
# panel AS (
#     SELECT s.orgpermid, s.year, s.esg_combined_score
#     FROM skeleton s
#     JOIN tr_worldscope.wrds_ws_funda f
#         ON s.worldscopecmpid = f.item6105
#         AND s.year = f.year_
#     WHERE f.freq = 'A'
# )
# SELECT COUNT(DISTINCT orgpermid) as n_entities
# FROM skeleton
#  """



# output_after_join = db.raw_sql(query_5) 
# output_before_join = db.raw_sql(query_6)

# 5. Checking distributional characteristics of dropped values:
# query_7 = f"""
#     WITH skeleton AS ( 
#         SELECT DISTINCT
#             p.orgpermid, 
#             p.worldscopecmpid, 
#             e.year,
#             e.valuescore as esg_combined_score
#         FROM tr_common.permorgref p
#         JOIN tr_esg.wrds_ref_esg e ON p.orgpermid = e.orgpermid
#         WHERE p.typecode = 'COM'
#             AND p.worldscopecmpid IS NOT NULL
#             AND e.fieldname = 'ESGCombinedScore'
#             AND e.valuescore IS NOT NULL
#             AND e.year BETWEEN 2009 AND 2025
#     ),  
#     panel AS (
#          SELECT s.orgpermid, s.year, s.esg_combined_score
#         FROM skeleton s
#         JOIN tr_worldscope.wrds_ws_funda f
#         ON s.worldscopecmpid = f.item6105
#         AND s.year = f.year_
#          WHERE f.freq = 'A'
#         )           

#     SELECT g.lvl3permid, COUNT(DISTINCT s.orgpermid) as n_before
#     FROM skeleton s
#     JOIN tr_common.permorgref p ON s.orgpermid = p.orgpermid
#     JOIN tr_common.tmcregncntrymap g ON p.domcntrypermid = g.lvl5permid
#     GROUP BY g.lvl3permid

# """

# query_8 = f"""
#     WITH skeleton AS ( 
#         SELECT DISTINCT
#             p.orgpermid, 
#             p.worldscopecmpid, 
#             e.year,
#             e.valuescore as esg_combined_score
#         FROM tr_common.permorgref p
#         JOIN tr_esg.wrds_ref_esg e ON p.orgpermid = e.orgpermid
#         WHERE p.typecode = 'COM'
#             AND p.worldscopecmpid IS NOT NULL
#             AND e.fieldname = 'ESGCombinedScore'
#             AND e.valuescore IS NOT NULL
#             AND e.year BETWEEN 2009 AND 2025
#     ),  
#     panel AS (
#          SELECT s.orgpermid, s.year, s.esg_combined_score
#         FROM skeleton s
#         JOIN tr_worldscope.wrds_ws_funda f
#         ON s.worldscopecmpid = f.item6105
#         AND s.year = f.year_
#          WHERE f.freq = 'A'
#         )           

#     SELECT g.lvl3permid, COUNT(DISTINCT s.orgpermid) as n_after
#     FROM panel s
#     JOIN tr_common.permorgref p ON s.orgpermid = p.orgpermid
#     JOIN tr_common.tmcregncntrymap g ON p.domcntrypermid = g.lvl5permid
#     GROUP BY g.lvl3permid

# """



# output_before = db.raw_sql(query_7)
# output_after = db.raw_sql(query_8)

# attrition_df = pd.merge(output_before, output_after, on=output_before.columns[0])
# attrition_df["attrition"] = attrition_df.iloc[:,1] - attrition_df.iloc[:,2]
# # attrition_df["attrition_rel"] = attrition_df[attrition_df].copy()

# rel_attrition = [attrition_df.iloc[i,3]/attrition_df.iloc[i,1] for i in range(len(attrition_df))]
# attrition_df["rel_attrition"]= rel_attrition

# attrition_df.to_csv("attrition_after_join_geo_codes.csv")

query_9= f"""
    SELECT g.lvl3permid, c.lvl5isocntry, COUNT(*) as n
    FROM tr_common.tmcregncntrymap g
    JOIN tr_common.tmcregncntrymap c ON g.lvl3permid = c.lvl3permid
    WHERE g.lvl3permid IN (100219.0, 100060.0)
    GROUP BY g.lvl3permid, c.lvl5isocntry
    
"""

output_geo = db.raw_sql(query_9)
print(output_geo)