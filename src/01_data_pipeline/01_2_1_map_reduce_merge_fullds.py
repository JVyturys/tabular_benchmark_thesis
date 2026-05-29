import pandas as pd
import gc
import os

def main():
    print("Starte Map-Reduce: Extraktion des Full-Datasets...")
    
    # Pfade
    raw_path = "data/0_raw/raw_merged_full.csv"
    esg_path = "data/0_raw/thesis_Refinitiv_scores.csv"
    reference_dataset_path = "data/03_processed/thesis_base_dataset_q_ftdrop.csv"
    chunk_dir = "data/interim/chunks_fullds"
    output_path = "data/03_processed/thesis_fullds_with_nulls.csv"
    log_dir = "results/01_desc_analysis"
    log_path = f"{log_dir}/map_reduce_fullds_log.txt"
    
    os.makedirs(chunk_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. Referenzdaten laden
    try:
        df_ref = pd.read_csv(reference_dataset_path, low_memory=False)
        golden_columns = df_ref.columns.tolist()
        expected_scores_count = len(df_ref)
        print(f"Referenzdatensatz geladen. Erwartete Anzahl verfügbarer ESG-Scores: {expected_scores_count}")
        del df_ref
        gc.collect()
    except FileNotFoundError:
        print(f"Fehler: Referenzdatei {reference_dataset_path} nicht gefunden.")
        return

    # Zielvariablen definieren und aus den Compustat-Suchspalten entfernen
    esg_columns = ['esg_score', 'env_score', 'soc_score', 'gov_score']
    compustat_cols = [col for col in golden_columns if col not in esg_columns]
    
    # Sicherstellen, dass orgpermid, isin und fyear für den Merge vorhanden sind
    if 'orgpermid' not in compustat_cols: compustat_cols.append('orgpermid')
    if 'isin' not in compustat_cols: compustat_cols.append('isin')
    if 'fyear' not in compustat_cols: compustat_cols.append('fyear')

    # 2. ESG-Scores in den RAM laden und anpassen
    print("\nLade Refinitiv ESG-Scores in den Arbeitsspeicher...")
    try:
        esg_df = pd.read_csv(esg_path, low_memory=False)
        esg_df.columns = esg_df.columns.str.lower()
        
        # 'fy' zu 'fyear' umbenennen, damit es zum Compustat-Standard passt
        if 'fy' in esg_df.columns:
            esg_df = esg_df.rename(columns={'fy': 'fyear'})
            
        esg_columns_lower = [col.lower() for col in esg_columns]

        # Auf die benötigten Spalten filtern
        esg_df = esg_df[['orgpermid', 'fyear'] + esg_columns_lower]

        # Duplikate in den Scores entfernen
        esg_df = esg_df.drop_duplicates(subset=['orgpermid', 'fyear'], keep='last')
        print(f"ESG-Scores erfolgreich geladen und angepasst: {len(esg_df)} Zeilen.")
        
    except Exception as e:
        print(f"Fehler beim Laden der ESG-Daten: {e}")
        return

    # 3. Map-Phase: Rohdaten in Batches verarbeiten und mit Scores mergen
    chunk_size = 1_000_000
    chunk_files = []
    
    print("\n--- Starte Chunk-Verarbeitung ---")
    try:
        chunk_iterator = pd.read_csv(raw_path, chunksize=chunk_size, usecols=compustat_cols, low_memory=False)
        
        for i, chunk in enumerate(chunk_iterator):
            print(f"Verarbeite Batch {i+1}...")
            
            initial_len = len(chunk)
            chunk = chunk.drop_duplicates()
            
            # Duplikat-Bereinigung innerhalb des Chunks über orgpermid + fyear
            if 'orgpermid' in chunk.columns and 'fyear' in chunk.columns:
                chunk = chunk.drop_duplicates(subset=['orgpermid', 'fyear'], keep='last')
                
                # LEFT JOIN: ESG-Scores an die Compustat-Daten über orgpermid anhängen
                chunk = pd.merge(chunk, esg_df, on=['orgpermid', 'fyear'], how='left')
            
            chunk_file = f"{chunk_dir}/chunk_fullds_{i+1}.csv"
            chunk.to_csv(chunk_file, index=False)
            chunk_files.append(chunk_file)
            
            dropped = initial_len - len(chunk)
            print(f" -> Batch {i+1} gespeichert: {len(chunk)} Zeilen ({dropped} Duplikate entfernt).")
            
            del chunk
            gc.collect()
            
    except Exception as e:
        print(f"Fehler bei der Batch-Verarbeitung: {e}")
        return

    # 4. Reduce-Phase: Alle bereinigten Chunks zusammenführen
    print("\n--- Starte Merge-Phase ---")
    merged_df = pd.DataFrame()
    
    for file in chunk_files:
        print(f"Lade {file} in finalen Datensatz...")
        temp_df = pd.read_csv(file, low_memory=False)
        merged_df = pd.concat([merged_df, temp_df], ignore_index=True)
        
        del temp_df
        gc.collect()
        
    print("\nFühre finale Duplikatbereinigung über den gemergten Datensatz durch...")
    initial_final = len(merged_df)
    
    if 'orgpermid' in merged_df.columns and 'fyear' in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=['orgpermid', 'fyear'], keep='last')
        
    final_dropped = initial_final - len(merged_df)
    
    # 5. Metriken berechnen und Sanity Check durchführen
    total_rows = len(merged_df)
    actual_scores_count = merged_df['esg_score'].notna().sum()
    missing_scores_count = merged_df['esg_score'].isna().sum()
    
    sanity_check_passed = (expected_scores_count == actual_scores_count)
    sanity_status = "BESTANDEN (Korrekt)" if sanity_check_passed else "FEHLER (Abweichung vorhanden!)"
    
    # 6. Speichern des Full-Datasets
    merged_df.to_csv(output_path, index=False)
    
    # 7. TXT-Log schreiben
    log_content = [
        "==================================================\n",
        "     LOG: MAP-REDUCE FULL-DATASET EXTRAKTION      \n",
        "==================================================\n\n",
        f"Zieldatei: {output_path}\n",
        f"Gesamtanzahl Zeilen im Full-Dataset: {total_rows}\n",
        f"Spaltenanzahl: {merged_df.shape[1]}\n\n",
        "--- ESG-Score Verfügbarkeit ---\n",
        f"Beobachtungen MIT ESG-Score:    {actual_scores_count}\n",
        f"Beobachtungen OHNE ESG-Score:   {missing_scores_count}\n",
        f"Prozentuale Abdeckung:          {(actual_scores_count / total_rows * 100):.2f}%\n\n",
        "--- Bereinigungseffekte ---\n",
        f"Übergreifende Duplikate in Reduce-Phase entfernt: {final_dropped}\n\n",
        "==================================================\n",
        "            SANITY CHECK / VALIDIERUNG            \n",
        "==================================================\n",
        f"Erwartete Scores (aus gefiltertem Datensatz): {expected_scores_count}\n",
        f"Tatsächliche Scores (im neuen Full-Dataset):  {actual_scores_count}\n",
        f"Status Sanity Check:                          {sanity_status}\n",
        "==================================================\n"
    ]
    
    with open(log_path, 'w') as f:
        f.writelines(log_content)
        
    print("\n" + "".join(log_content))
    print(f"Log gespeichert unter: {log_path}")

if __name__ == "__main__":
    main()