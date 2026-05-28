import wrds
import pandas as pd
import os

def main():
    print("Starte Abfrage der Variablen-Definitionen via WRDS...")
    
    input_path = "data/03_processed/thesis_base_dataset_ftdrop.csv"
    output_dir = "results/01_desc_analysis"
    output_file = f"{output_dir}/variable_definitions_codebook.csv"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(input_path, nrows=0)
        my_columns = df.columns.tolist()
        print(f"{len(my_columns)} Variablen für den Abgleich geladen.")
    except FileNotFoundError:
        print(f"Fehler: Datei {input_path} nicht gefunden.")
        return

    db = wrds.Connection()
    
    library = 'comp'
    table = 'funda'
    
    print(f"Rufe Data Dictionary für {library}.{table} ab...")
    try:
        dict_df = db.describe_table(library, table)
    except Exception as e:
        print(f"Fehler bei WRDS-Abfrage: {e}")
        return
        
    print(f"Verfügbare Metadaten-Spalten von WRDS: {dict_df.columns.tolist()}")
    
    dict_df['name'] = dict_df['name'].str.lower()
    
    # Prüfung, welche Spalte die Beschreibung enthält
    if 'label' in dict_df.columns:
        desc_col = 'label'
    elif 'description' in dict_df.columns:
        desc_col = 'description'
    else:
        desc_col = None
        print("Warnung: Keine Beschreibungs-Spalte (label/description) von WRDS geliefert.")

    # Dictionary für das Mapping erstellen
    if desc_col:
        desc_map = dict(zip(dict_df['name'], dict_df[desc_col]))
    else:
        desc_map = {}
    
    # Eigene Spalten mit WRDS-Definitionen matchen
    codebook_data = []
    for col in my_columns:
        definition = desc_map.get(col.lower(), "Keine Definition gefunden")
        codebook_data.append({'Variable': col, 'Definition': definition})
        
    codebook_df = pd.DataFrame(codebook_data)
    codebook_df.to_csv(output_file, index=False)
    
    print(f"Codebook generiert und gespeichert unter: {output_file}")

if __name__ == "__main__":
    main()