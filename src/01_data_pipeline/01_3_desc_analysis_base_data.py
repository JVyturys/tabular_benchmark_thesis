import pandas as pd
import os

def main():
    print("Starte Analyse der fehlenden Werte...")
    
    # Pfade entsprechend der neuen Ordnerstruktur
    input_path = "data/03_processed/thesis_base_dataset_01.csv"
    output_dir = "results/01_desc_analysis"
    output_file = f"{output_dir}/missing_values_report.csv"
    summary_file = f"{output_dir}/missing_values_summary.txt"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Daten laden
    try:
        df = pd.read_csv(input_path, low_memory=False)
        print(f"Datensatz geladen. Dimensionen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten.")
    except FileNotFoundError:
        print(f"Fehler: Datei {input_path} nicht gefunden.")
        return

    # Berechnung der fehlenden Werte in Prozent
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    
    # Erstellung eines DataFrames für den Report
    missing_df = pd.DataFrame({
        'Variable': missing_percentages.index,
        'Missing_Percentage': missing_percentages.values
    })
    
    # Absteigend sortieren
    missing_df = missing_df.sort_values(by='Missing_Percentage', ascending=False).reset_index(drop=True)
    
    # Speichern des detaillierten Reports
    missing_df.to_csv(output_file, index=False)
    print(f"Detaillierter Report gespeichert unter: {output_file}")
    
    # Generierung der Zusammenfassung (Summary)
    thresholds = [0, 10, 30, 50, 80, 99]
    summary_lines = ["Zusammenfassung der fehlenden Werte:\n", "-"*40 + "\n"]
    
    total_cols = len(df.columns)
    summary_lines.append(f"Gesamtanzahl Variablen: {total_cols}\n\n")
    
    for t in thresholds:
        count = (missing_df['Missing_Percentage'] > t).sum()
        pct_of_cols = (count / total_cols) * 100
        summary_lines.append(f"Variablen mit > {t}% fehlenden Werten: {count} ({pct_of_cols:.1f}%)\n")
        
    summary_lines.append("\nBesonderer Check (Zielvariablen und Identifier):\n")
    key_vars = ['isin', 'fyear', 'orgpermid', 'esg_score', 'env_score', 'soc_score', 'gov_score']
    for kv in key_vars:
        if kv in missing_df['Variable'].values:
            val = missing_df.loc[missing_df['Variable'] == kv, 'Missing_Percentage'].values[0]
            summary_lines.append(f" - {kv}: {val:.2f}% fehlend\n")
            
    # Ausgabe im Terminal
    print("\n" + "".join(summary_lines))
    
    # Speichern der Zusammenfassung
    with open(summary_file, 'w') as f:
        f.writelines(summary_lines)
    print(f"Zusammenfassung gespeichert unter: {summary_file}")

if __name__ == "__main__":
    main()