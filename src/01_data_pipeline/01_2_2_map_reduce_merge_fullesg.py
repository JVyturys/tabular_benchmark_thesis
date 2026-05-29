import wrds
import pandas as pd
import os
import gc


def main():

    print("Verbinde mit WRDS-Datenbank...")
    db = wrds.Connection()
    
    # 1. Herunterladen und Formatieren der ESG-Scores
    query = "SELECT orgpermid, fy, item, value_ FROM tr_esg.esgscores WHERE item IN (1, 4, 5, 6)"
    df_scores = db.raw_sql(query)
    df_scores_wide = df_scores.pivot_table(index=['orgpermid', 'fy'], columns='item', values='value_').reset_index()
    df_scores_wide = df_scores_wide.rename(columns={1: 'esg_score', 4: 'env_score', 6: 'soc_score', 5: 'gov_score'})
    df_scores_wide = df_scores_wide.dropna()
    print(f"Scores geladen. Zeilen: {df_scores_wide.shape[0]}")
    esg_path = "data/thesis_Refinitiv_scores.csv"
    df_scores_wide.to_csv(esg_path, index=False)
    print(f"Scores gespeichert unter: {esg_path}")
    

    # 2. Initialisierung der Pfade und Verzeichnisse
    raw_path = "data/raw_merged_full.csv"
    chunk_dir = "data/chunks"
    final_path = "data/thesis_base_dataset_01.csv"
    os.makedirs(chunk_dir, exist_ok=True)

    print("Starte Chunk-Verarbeitung...")
    chunk_size = 1_000_000
    chunk_iterator = pd.read_csv(raw_path, chunksize=chunk_size, low_memory=False)
    chunk_files = []
    
    # 3. Map-Phase: Iterative Verarbeitung der Rohdaten
    for i, chunk in enumerate(chunk_iterator):
        merged_chunk = pd.merge(
            chunk, df_scores_wide, 
            left_on=['orgpermid', 'fyear'], right_on=['orgpermid', 'fy'], 
            how='inner'
        )
        
        if not merged_chunk.empty:
            if 'fy' in merged_chunk.columns: 
                merged_chunk = merged_chunk.drop(columns=['fy'])
                
            # Entfernen von Duplikaten innerhalb des aktuellen Chunks
            merged_chunk = merged_chunk.sort_values(by=['isin', 'fyear', 'esg_score'], ascending=[True, True, False])
            merged_chunk = merged_chunk.drop_duplicates(subset=['isin', 'fyear'], keep='first')
            
            # Speichern des bereinigten Chunks
            chunk_file = f"{chunk_dir}/clean_chunk_{i+1}.csv"
            merged_chunk.to_csv(chunk_file, index=False)
            chunk_files.append(chunk_file)
            
            print(f"Chunk {i+1} verarbeitet. Verbleibende Zeilen: {merged_chunk.shape[0]}")
        else:
            print(f"Chunk {i+1} verarbeitet. Verbleibende Zeilen: 0")

        # Manuelle Freigabe des Arbeitsspeichers
        del chunk, merged_chunk
        gc.collect()

    print("Starte Zusammenführung der Chunks...")
    
    # 4. Reduce-Phase: Zusammenführung und finale Bereinigung
    final_dfs = []
    for file in chunk_files:
        final_dfs.append(pd.read_csv(file, low_memory=False))
        
    df_final = pd.concat(final_dfs, ignore_index=True)
    
    print("Führe finale Duplikatbereinigung durch...")
    df_final = df_final.sort_values(by=['isin', 'fyear', 'esg_score'], ascending=[True, True, False])
    df_final = df_final.drop_duplicates(subset=['isin', 'fyear'], keep='first')
    
    df_final.to_csv(final_path, index=False)
    
    print(f"Verarbeitung abgeschlossen.")
    print(f"Dimensionen des finalen Datensatzes: {df_final.shape}")
    print(f"Gespeichert unter: {final_path}")

if __name__ == "__main__":
    main()