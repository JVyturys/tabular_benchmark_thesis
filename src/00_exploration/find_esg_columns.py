import pandas as pd

def main():
    print("[*] Lade die Spaltennamen aus der CSV...")
    file_path = "data/raw_merged_full.csv"
    
    try:
        # nrows=0
        df_cols = pd.read_csv(file_path, nrows=0)
        alle_spalten = df_cols.columns.tolist()
        
        # Schlüsselwörter
        keywords = ['esg', 'env', 'soc', 'gov', 'cg', 'score', 'rating', 'pillar']
        
        print("\n[+] Mögliche Zielvariablen:")
        treffer = 0
        for col in alle_spalten:
            # prüfen, ob eines der Keywords im Spaltennamen steckt
            if any(kw in col.lower() for kw in keywords):
                print(f" -> {col}")
                treffer += 1
                
        print(f"\n[*] {treffer} Spalten gefunden.")
        
    except FileNotFoundError:
        print(f"[-] FEHLER: Konnte {file_path} nicht finden.")

if __name__ == "__main__":
    main()