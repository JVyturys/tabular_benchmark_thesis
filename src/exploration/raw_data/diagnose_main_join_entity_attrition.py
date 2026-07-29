#######################################################################
''' 
src.exploration.raw_data.diagnose_main_join_entity_attrition

input: raw SQL
purpose: check entity coverage after jpining the esg and financials datasets.
output: coverage after join: 9.721 entities
        coverage before join: 13.841

        Highest attrition: in region 100219:
        4575 -> 609, delta: 3966 -> 86,6885%                 
'''
#######################################################################

import wrds
import log_config as ln
import pandas as pd

# initialize database connection
db = wrds.Connection(wrds_username = ln.wrds_log)

# pull number of entities after join:
try:
        query_5 = f"""
        WITH skeleton AS (
            SELECT DISTINCT 
                p.orgpermid, 
                p.worldscopecmpid, 
                e.year,
                e.valuescore as esg_combined_score
            FROM tr_common.permorgref p
            JOIN tr_esg.wrds_ref_esg e ON p.orgpermid = e.orgpermid
            WHERE p.typecode = 'COM'
            AND p.worldscopecmpid IS NOT NULL
            AND e.fieldname = 'ESGCombinedScore'
            AND e.valuescore IS NOT NULL
            AND e.year BETWEEN 2009 AND 2025
        ),
        panel AS (
            SELECT s.orgpermid, s.year, s.esg_combined_score
            FROM skeleton s
            JOIN tr_worldscope.wrds_ws_funda f
                ON s.worldscopecmpid = f.item6105
                AND s.year = f.year_
            WHERE f.freq = 'A'
        )
        SELECT COUNT(DISTINCT orgpermid) as n_entities
        FROM panel
        """

        # pull number of entities before join
        query_6 = f"""
        WITH skeleton AS (
            SELECT DISTINCT 
                p.orgpermid, 
                p.worldscopecmpid, 
                e.year,
                e.valuescore as esg_combined_score
            FROM tr_common.permorgref p
            JOIN tr_esg.wrds_ref_esg e ON p.orgpermid = e.orgpermid
            WHERE p.typecode = 'COM'
            AND p.worldscopecmpid IS NOT NULL
            AND e.fieldname = 'ESGCombinedScore'
            AND e.valuescore IS NOT NULL
            AND e.year BETWEEN 2009 AND 2025
        ),
        panel AS (
            SELECT s.orgpermid, s.year, s.esg_combined_score
            FROM skeleton s
            JOIN tr_worldscope.wrds_ws_funda f
                ON s.worldscopecmpid = f.item6105
                AND s.year = f.year_
            WHERE f.freq = 'A'
        )
        SELECT COUNT(DISTINCT orgpermid) as n_entities
        FROM skeleton
        """

        output_after_join = db.raw_sql(query_5) 
        output_before_join = db.raw_sql(query_6)

        print(f'''Output after join: {output_after_join}''')
        print(f'''Output before join: {output_before_join}''')

except: 
     print(f"An error occured during attrition (query 5&6) SQL pull!")

try:
    query_7 = f"""
        WITH skeleton AS ( 
            SELECT DISTINCT
                p.orgpermid, 
                p.worldscopecmpid, 
                e.year,
                e.valuescore as esg_combined_score
            FROM tr_common.permorgref p
            JOIN tr_esg.wrds_ref_esg e ON p.orgpermid = e.orgpermid
            WHERE p.typecode = 'COM'
                AND p.worldscopecmpid IS NOT NULL
                AND e.fieldname = 'ESGCombinedScore'
                AND e.valuescore IS NOT NULL
                AND e.year BETWEEN 2009 AND 2025
        ),  
        panel AS (
            SELECT s.orgpermid, s.year, s.esg_combined_score
            FROM skeleton s
            JOIN tr_worldscope.wrds_ws_funda f
            ON s.worldscopecmpid = f.item6105
            AND s.year = f.year_
            WHERE f.freq = 'A'
            )           

        SELECT g.lvl3permid, COUNT(DISTINCT s.orgpermid) as n_before
        FROM skeleton s
        JOIN tr_common.permorgref p ON s.orgpermid = p.orgpermid
        JOIN tr_common.tmcregncntrymap g ON p.domcntrypermid = g.lvl5permid
        GROUP BY g.lvl3permid

    """

    query_8 = f"""
        WITH skeleton AS ( 
            SELECT DISTINCT
                p.orgpermid, 
                p.worldscopecmpid, 
                e.year,
                e.valuescore as esg_combined_score
            FROM tr_common.permorgref p
            JOIN tr_esg.wrds_ref_esg e ON p.orgpermid = e.orgpermid
            WHERE p.typecode = 'COM'
                AND p.worldscopecmpid IS NOT NULL
                AND e.fieldname = 'ESGCombinedScore'
                AND e.valuescore IS NOT NULL
                AND e.year BETWEEN 2009 AND 2025
        ),  
        panel AS (
            SELECT s.orgpermid, s.year, s.esg_combined_score
            FROM skeleton s
            JOIN tr_worldscope.wrds_ws_funda f
            ON s.worldscopecmpid = f.item6105
            AND s.year = f.year_
            WHERE f.freq = 'A'
            )           

        SELECT g.lvl3permid, COUNT(DISTINCT s.orgpermid) as n_after
        FROM panel s
        JOIN tr_common.permorgref p ON s.orgpermid = p.orgpermid
        JOIN tr_common.tmcregncntrymap g ON p.domcntrypermid = g.lvl5permid
        GROUP BY g.lvl3permid

    """
except:
    print(f"An error occured during attrition (query 7&8) SQL pull!")
    

output_before = db.raw_sql(query_7)
output_after = db.raw_sql(query_8)

attrition_df = pd.merge(output_before, output_after, on=output_before.columns[0])
attrition_df["attrition"] = attrition_df.iloc[:,1] - attrition_df.iloc[:,2]

rel_attrition = [attrition_df.iloc[i,3]/attrition_df.iloc[i,1] for i in range(len(attrition_df))]
attrition_df["rel_attrition"]= rel_attrition

print(attrition_df)

try:
    query_9= f"""
        SELECT g.lvl3permid, c.lvl5isocntry, COUNT(*) as n
        FROM tr_common.tmcregncntrymap g
        JOIN tr_common.tmcregncntrymap c ON g.lvl3permid = c.lvl3permid
        WHERE g.lvl3permid IN (100219.0, 100060.0)
        GROUP BY g.lvl3permid, c.lvl5isocntry
        
        """
except:
    print(f"An error occured during country code sanity check SQL pull ")

output_geo = db.raw_sql(query_9)
print(output_geo)