import wrds
import pandas as pd
import os

def main():
    print("[*] Initiiere WRDS Verbindung...")
    db = wrds.Connection()

    years = list(range(2010, 2026))
    save_path = "data/raw_merged_full.csv"

    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"[*] Alte Datei {save_path} gelöscht für Neustart.")

    os.makedirs("data", exist_ok=True)

    print("\n[*] Starte Data-Warehouse Pipeline...")
    
    for year in years:
        print(f" [**] Verarbeite Geschäftsjahr {year}...")
        
        query = f"""
            SELECT c.*, e.*
            FROM comp.g_funda c
            INNER JOIN tr_esg.wrds_ref_esg e
            ON c.isin = e.isin AND c.fyear = e.year
            WHERE c.fyear = {year}
              AND c.indfmt = 'INDL' 
              AND c.consol = 'C'
        """
        
        try:
            chunk = db.raw_sql(query)
            
            if chunk is not None and not chunk.empty:
                # Duplikate bei Spaltennamen entfernen
                chunk = chunk.loc[:, ~chunk.columns.duplicated()]
                write_header = not os.path.exists(save_path)
                
                chunk.to_csv(save_path, mode='a', header=write_header, index=False)
                
                print(f"    [+] {year} gesichert: {chunk.shape[0]} Zeilen auf Festplatte geschrieben.")
            else:
                print(f"    [-] {year}: Keine passenden Daten gefunden.")
                
        except Exception as e:
            print(f"    [!] Fehler im Jahr {year}: {e}")

    print("\n[+] Pipeline beendet.")
    print(f"[*] Der Datensatz liegt unter: {save_path}")

if __name__ == "__main__":
    main()