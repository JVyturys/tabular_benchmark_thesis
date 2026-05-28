# Skript um herauszufinden, welche der Variablen in Refinitiv Datensatz
# ESG-Ratings sind, die ich als Zielvariable benutzen kann.

import wrds
import pandas as pd

def main():
    print("[*] Verbinde mit WRDS...")
    db = wrds.Connection()
    
    print("\n[*] Lese die Item-Namen aus...")
    try:
        
        query = """
            SELECT *
            FROM tr_esg.esgitem
            WHERE item IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
            ORDER BY item
        """
        
        df_mapping = db.raw_sql(query)
        
        print("\n[+] Refinitiv Mapping:")
        # Drucken aller Spalten, um zu sehen, wo der Klartext-Name steht
        print(df_mapping.to_string(index=False))
        
    except Exception as e:
        print(f"[-] Fehler: {e}")

if __name__ == "__main__":
    main()