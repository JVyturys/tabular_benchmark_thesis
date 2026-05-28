import pandas as pd
import os

def main():
    print("Starte deskriptive Analyse der Zielvariablen und Panel-Struktur...")
    
    # Pfade
    input_path = "data/03_processed/thesis_base_dataset_q_ftdrop.csv"
    output_dir = "results/02_desc_analysis_esg"
    stats_file = f"{output_dir}/esg_summary_statistics.csv"
    year_dist_file = f"{output_dir}/observations_per_year.csv"
    log_file = f"{output_dir}/descriptive_summary.txt"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Daten laden
    try:
        df = pd.read_csv(input_path, low_memory=False)
    except FileNotFoundError:
        print(f"Fehler: Datei {input_path} nicht gefunden.")
        return

    targets = ['esg_score', 'env_score', 'soc_score', 'gov_score']
    
    # 1. Summary Statistics der ESG-Scores
    print("Berechne Statistiken der ESG-Scores...")
    summary_stats = df[targets].describe().T
    summary_stats = summary_stats.round(2)
    summary_stats.to_csv(stats_file)
    
    # 2. Panel-Struktur: Beobachtungen pro Jahr
    print("Berechne Beobachtungen pro Jahr...")
    if 'fyear' in df.columns:
        year_dist = df['fyear'].value_counts().sort_index().reset_index()
        year_dist.columns = ['Geschaeftsjahr', 'Anzahl_Beobachtungen']
        
        # Prozentualer Anteil
        year_dist['Prozent'] = (year_dist['Anzahl_Beobachtungen'] / len(df) * 100).round(2)
        year_dist.to_csv(year_dist_file, index=False)
    else:
        print("Warnung: Spalte 'fyear' nicht gefunden.")
        year_dist = pd.DataFrame()

    # 3. Text-Zusammenfassung (Log) generieren
    log_content = [
        "=========================================\n",
        "  DESKRIPTIVE STATISTIK: ZIELVARIABLEN   \n",
        "=========================================\n\n",
        f"Gesamtzahl der Beobachtungen (N): {len(df)}\n",
        f"Anzahl der einzigartigen Unternehmen: {df['isin'].nunique()}\n\n",
        "--- Verteilung der Scores ---\n",
        summary_stats.to_string(),
        "\n\n",
        "--- Beobachtungen pro Jahr ---\n",
        year_dist.to_string(index=False) if not year_dist.empty else "Keine Jahresdaten verfügbar."
    ]
    
    with open(log_file, 'w') as f:
        f.writelines(log_content)
        
    print("\n" + "".join(log_content))
    print(f"\nErgebnisse gespeichert im Ordner: {output_dir}/")

if __name__ == "__main__":
    main()