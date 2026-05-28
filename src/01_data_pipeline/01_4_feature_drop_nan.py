import pandas as pd
import os

def main():
    print("Starte Bereinigung der Features (Schwellenwert: > 30% fehlende Werte)...")
    
    input_path = "data/03_processed/thesis_base_dataset_01.csv"
    output_path = "data/03_processed/thesis_base_dataset_ftdrop.csv"
    log_dir = "results/01_desc_analysis"
    log_path = f"{log_dir}/feature_drop_log.txt"
    threshold_pct = 30.0
    
    # Sicherstellen, dass der Log-Ordner existiert
    os.makedirs(log_dir, exist_ok=True)
    
    # Pflichtvariablen, die niemals gelöscht werden dürfen
    protected_vars = ['isin', 'fyear', 'orgpermid', 'esg_score', 'env_score', 'soc_score', 'gov_score']
    
    try:
        df = pd.read_csv(input_path, low_memory=False)
    except FileNotFoundError:
        print(f"Fehler: Datei {input_path} nicht gefunden.")
        return

    # Berechnung der prozentualen fehlenden Werte
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    
    # Identifikation der zu behaltenden und zu löschenden Spalten
    cols_to_keep_initial = missing_percentages[missing_percentages <= threshold_pct].index.tolist()
    
    cols_to_keep = []
    cols_to_drop = []
    
    # Sicherstellen, dass geschützte Variablen enthalten bleiben und der Rest korrekt zugeordnet wird
    for col in df.columns:
        if col in protected_vars:
            cols_to_keep.append(col)
        elif col in cols_to_keep_initial:
            cols_to_keep.append(col)
        else:
            cols_to_drop.append(col)
            
    # Datensatz filtern
    df_cleaned = df[cols_to_keep]
    
    # Ergebnisse speichern
    df_cleaned.to_csv(output_path, index=False)
    
    # ---------------------------------------------------------
    # Erstellung der Dokumentation (Log-Datei)
    # ---------------------------------------------------------
    dropped_count = len(cols_to_drop)
    kept_count = len(cols_to_keep)
    
    log_content = [
        "=========================================\n",
        "       DOKUMENTATION: FEATURE DROP       \n",
        "=========================================\n\n",
        f"Schwellenwert für fehlende Werte: > {threshold_pct}%\n\n",
        f"Ursprüngliche Dimensionen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten\n",
        f"Neue Dimensionen: {df_cleaned.shape[0]} Zeilen, {df_cleaned.shape[1]} Spalten\n",
        f"Entfernte Spalten: {dropped_count}\n",
        f"Behaltene Spalten: {kept_count}\n\n",
        "Geschützte Variablen (von der Löschung ausgeschlossen):\n"
    ]
    
    for var in protected_vars:
        log_content.append(f" - {var}\n")
        
    log_content.append("\n-----------------------------------------\n")
    log_content.append("Liste der ENTFERNTEN Variablen:\n")
    log_content.append("-----------------------------------------\n")
    
    # Sortierte Ausgabe der gelöschten Variablen inkl. ihres NaN-Anteils
    for col in sorted(cols_to_drop):
        missing_val = missing_percentages[col]
        log_content.append(f" - {col} ({missing_val:.2f}% fehlend)\n")
        
    # Log-Datei auf Festplatte schreiben
    with open(log_path, 'w') as f:
        f.writelines(log_content)
    
    # Ausgabe im Terminal
    print(f"Ursprüngliche Dimensionen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten.")
    print(f"Entfernte Spalten: {dropped_count}")
    print(f"Neue Dimensionen: {df_cleaned.shape[0]} Zeilen, {df_cleaned.shape[1]} Spalten.")
    print(f"Bereinigter Datensatz gespeichert unter: {output_path}")
    print(f"Dokumentation gespeichert unter: {log_path}")

if __name__ == "__main__":
    main()