import wrds
import config as con
import log_config as lg
from utils.reporting import AnalysisLogger as Al

log_dir = con.RESULTS_DIR / "exploration" / "raw_data"

log_esg = Al(output_path=log_dir, filename= "esg_panel.txt")

db = wrds.Connection(wrds_username=lg.wrds_log)

# How many NaNs are in the ESGCombinedScore Panel?
query = """
    SELECT 
        valuescore
    FROM tr_esg.wrds_ref_esg
    WHERE fieldname = 'ESGCombinedScore' AND valuescore IS NULL
"""

# logging results:
output = db.raw_sql(query)
log_esg.section("NaN within the tr_esg.wrds_ref_esg 'valuescore' column:")
log_esg.log(query)
log_esg.log(output)
 
# Summary statistics ESGScores:
query = """
    SELECT 
        MIN(valuescore),
        AVG(valuescore),
        MAX(valuescore)
    FROM tr_esg.wrds_ref_esg
    WHERE fieldname = 'ESGCombinedScore'
"""
# logging results:
output = db.raw_sql(query)
log_esg.section("Summary statistics: raw ESG Data")
log_esg.log(query)
log_esg.log(output)

query = """
    SELECT         
        year,
        orgpermid,
        COUNT(*) as n
    FROM tr_esg.wrds_ref_esg
    WHERE year >= 2009 AND year < 2026 AND fieldname = 'ESGCombinedScore' AND valuescore IS NOT NULL
    GROUP BY year, orgpermid
    """
output = db.raw_sql(query)
log_esg.section("Summary statistics: distinct company-year entries")
log_esg.log(query)
log_esg.log(output)

query = """
    SELECT COUNT(DISTINCT orgpermid) as distinct_companies
    FROM tr_esg.wrds_ref_esg
    WHERE year >= 2009 AND year < 2026 AND fieldname = 'ESGCombinedScore' AND valuescore IS NOT NULL
"""
output = db.raw_sql(query)
log_esg.section("Distinct Companies")
log_esg.log(output)

query = """
    SELECT year, COUNT(*) as observations_per_year
    FROM tr_esg.wrds_ref_esg
    WHERE year >= 2009 AND year < 2026 AND fieldname = 'ESGCombinedScore' AND valuescore IS NOT NULL
    GROUP BY year
"""
output = db.raw_sql(query)
log_esg.section("Observations per year")
log_esg.log(output)








