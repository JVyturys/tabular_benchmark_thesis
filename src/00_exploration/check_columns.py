import wrds

def main():
    print("[*] Verbinde mit WRDS...")
    db = wrds.Connection()
    
    print("\n[*] Untersuche Refinitiv ESG (tr_esg.wrds_ref_esg)...")
    try:
        esg_cols = db.describe_table('tr_esg', 'wrds_ref_esg')
        # Zeige alle Spalten an, die "year", "date" oder "fy" im Namen haben
        time_cols = esg_cols[esg_cols['name'].str.contains('year|date|fy', case=False, na=False)]
        print("Mögliche Zeit-Spalten in ESG:")
        print(time_cols[['name', 'type']].to_string(index=False))
        
        # Zeige auch, wie der Identifier heißt (isin, orgid, etc.)
        id_cols = esg_cols[esg_cols['name'].str.contains('isin|cusip|orgid|id', case=False, na=False)]
        print("\nMögliche Identifier-Spalten in ESG:")
        print(id_cols[['name', 'type']].to_string(index=False))
        
    except Exception as e:
        print(f"Fehler bei ESG: {e}")

    print("\n[*] Untersuche Compustat Global (comp.g_funda)...")
    try:
        comp_cols = db.describe_table('comp', 'g_funda')
        # prüfen, welche Werte in datafmt etc. existieren
        print("Spalten existieren.")
    except Exception as e:
        print(f"Fehler bei Compustat: {e}")

if __name__ == "__main__":
    main()