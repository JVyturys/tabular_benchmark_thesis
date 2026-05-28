import pandas as pd
import os

def main():
    print("Starte qualitativen Feature Drop basierend auf Codebook...")
    
    # Pfade
    dataset_path = "data/03_processed/thesis_base_dataset_ftdrop.csv"
    codebook_path = "results/01_desc_analysis/variable_definitions_codebook_ann_drop.csv"
    output_path = "data/03_processed/thesis_base_dataset_q_ftdrop.csv"
    log_dir = "results/01_desc_analysis"
    log_path = f"{log_dir}/qualitative_drop_log.txt"
    
    # Pflichtvariablen, die niemals gelöscht werden dürfen
    protected_vars = ['isin', 'fyear', 'orgpermid', 'esg_score', 'env_score', 'soc_score', 'gov_score']
    
    # 1. Codebook laden
    try:
        codebook = pd.read_csv(codebook_path, encoding='latin1') # 'latin1' wegen Umlaut in "Grund für Drop"
    except FileNotFoundError:
        print(f"Fehler: Die Datei {codebook_path} wurde nicht gefunden.")
        return

    # 2. Variablen identifizieren, die gelöscht werden sollen
    drop_indicators = ['drop']
    
    # NaN-Werte in der 'Drop'-Spalte durch leere Strings ersetzen und abgleichen
    drop_mask = codebook['Drop'].fillna('').astype(str).str.strip().str.lower().isin(drop_indicators)
    vars_to_drop_requested = codebook.loc[drop_mask, 'Variable'].tolist()
    
    # 3. Datensatz laden
    try:
        df = pd.read_csv(dataset_path, low_memory=False)
    except FileNotFoundError:
        print(f"Fehler: Die Datei {dataset_path} wurde nicht gefunden.")
        return
        
    print(f"Ursprüngliche Dimensionen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten.")
    
    # 4. Filterung verifizieren
    actual_vars_to_drop = []
    for var in vars_to_drop_requested:
        if var in df.columns:
            if var not in protected_vars:
                actual_vars_to_drop.append(var)
            else:
                print(f"Warnung: Die geschützte Variable '{var}' wurde im Codebook zum Löschen markiert. Löschung ignoriert.")

    # 5. Variablen löschen
    df_cleaned = df.drop(columns=actual_vars_to_drop)
    
    # 6. Bereinigten Datensatz speichern
    df_cleaned.to_csv(output_path, index=False)
    print(f"Neue Dimensionen: {df_cleaned.shape[0]} Zeilen, {df_cleaned.shape[1]} Spalten.")
    print(f"Finaler Datensatz gespeichert unter: {output_path}")
    
    # 7. Log-Datei schreiben
    log_content = [
        "=========================================\n",
        "  DOKUMENTATION: QUALITATIVER FEATURE DROP \n",
        "=========================================\n\n",
        f"Ursprüngliche Spalten: {df.shape[1]}\n",
        f"Entfernte Spalten: {len(actual_vars_to_drop)}\n",
        f"Verbleibende Spalten: {df_cleaned.shape[1]}\n\n",
        "Liste der ENTFERNTEN Variablen:\n",
        "-----------------------------------------\n"
    ]
    
    for var in sorted(actual_vars_to_drop):
        # Den Grund für den Drop aus dem Codebook ziehen
        grund = codebook.loc[codebook['Variable'] == var, 'Grund für Keep/Drop'].values
        grund_text = grund[0] if len(grund) > 0 and pd.notna(grund[0]) else "Kein Grund angegeben"
        log_content.append(f" - {var}: {grund_text}\n")
        
    with open(log_path, 'w') as f:
        f.writelines(log_content)
        
    print(f"Dokumentation gespeichert unter: {log_path}")

if __name__ == "__main__":
    main()