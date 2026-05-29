import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Starte Analyse der Selektionsverzerrung...")
    
    # Pfade
    input_path = "data/03_processed/thesis_fullds_with_nulls.csv"
    output_dir = "results/02_desc_analysis_esg"
    plot_dir = f"{output_dir}/plots"
    log_file = f"{output_dir}/selection_bias_stats.txt"
    
    os.makedirs(plot_dir, exist_ok=True)
    
    # Daten laden
    try:
        df = pd.read_csv(input_path, low_memory=False)
    except FileNotFoundError:
        print(f"Fehler: Datei {input_path} nicht gefunden.")
        return

    # Prüfen, ob die fundamentalen Variablen 'at' (Total Assets) und 'revt' (Revenue) existieren
    required_vars = ['at', 'revt']
    missing_vars = [var for var in required_vars if var not in df.columns]
    if missing_vars:
        print(f"Fehler: Folgende Variablen fehlen im Datensatz: {missing_vars}")
        return

    # 1. Zielvariable für die Gruppierung erstellen
    df['has_score'] = df['esg_score'].notna()
    
    # --- Datenvorbereitung für Bilanzsumme (at) ---
    df_at = df.dropna(subset=['at']).copy()
    df_at = df_at[df_at['at'] > 0]
    df_at['log_at'] = np.log1p(df_at['at'])
    stats_at = df_at.groupby('has_score')['at'].describe().round(2)
    
    # --- Datenvorbereitung für Umsatz (revt) ---
    df_revt = df.dropna(subset=['revt']).copy()
    df_revt = df_revt[df_revt['revt'] > 0]
    df_revt['log_revt'] = np.log1p(df_revt['revt'])
    stats_revt = df_revt.groupby('has_score')['revt'].describe().round(2)
    
    # Score-Abdeckung nach Geschäftsjahr berechnen
    coverage_by_year = df.groupby('fyear')['has_score'].mean() * 100
    
    # ---------------------------------------------------------
    # 2. Plots generieren
    # ---------------------------------------------------------
    sns.set_theme(style="whitegrid")
    
    # Plot A: Total Assets
    print("Generiere Dichteplot für Bilanzsumme (Total Assets)...")
    plt.figure(figsize=(10, 6), dpi=300)
    sns.kdeplot(
        data=df_at, x='log_at', hue='has_score', fill=True, 
        common_norm=False, alpha=0.4, linewidth=2,
        palette={True: '#2ca02c', False: '#d62728'}
    )
    plt.title('Selection Bias: Unternehmensgröße (Total Assets) nach Score-Verfügbarkeit', pad=15, weight='bold', fontsize=14)
    plt.xlabel('Log(Total Assets)', fontsize=12)
    plt.ylabel('Dichte', fontsize=12)
    plt.legend(title='ESG-Score verfügbar?', labels=['Ja', 'Nein'], frameon=True, facecolor='white')
    plt.tight_layout()
    
    plot_path_at = f"{plot_dir}/selection_bias_assets_kde.png"
    plt.savefig(plot_path_at, bbox_inches='tight')
    plt.close()
    
    # Plot B: Revenue
    print("Generiere Dichteplot für Umsatz (Revenue)...")
    plt.figure(figsize=(10, 6), dpi=300)
    sns.kdeplot(
        data=df_revt, x='log_revt', hue='has_score', fill=True, 
        common_norm=False, alpha=0.4, linewidth=2,
        palette={True: '#1f77b4', False: '#ff7f0e'}
    )
    plt.title('Selection Bias: Unternehmensumsatz (Revenue) nach Score-Verfügbarkeit', pad=15, weight='bold', fontsize=14)
    plt.xlabel('Log(Total Revenue)', fontsize=12)
    plt.ylabel('Dichte', fontsize=12)
    plt.legend(title='ESG-Score verfügbar?', labels=['Ja', 'Nein'], frameon=True, facecolor='white')
    plt.tight_layout()
    
    plot_path_revt = f"{plot_dir}/selection_bias_revenue_kde.png"
    plt.savefig(plot_path_revt, bbox_inches='tight')
    plt.close()
    
    # ---------------------------------------------------------
    # 3. Text-Log schreiben
    # ---------------------------------------------------------
    log_content = [
        "==================================================\n",
        "        SELECTION BIAS ANALYSE (GRÖSSE & UMSATZ)  \n",
        "==================================================\n\n",
        "False = Kein Score, True = Score vorhanden\n\n",
        "--- 1. Bilanzsumme ('at') ---\n",
        stats_at.to_string(),
        "\n\n--- 2. Umsatz ('revt') ---\n",
        stats_revt.to_string(),
        "\n\n==================================================\n",
        "          SCORE-ABDECKUNG NACH GESCHÄFTSJAHR      \n",
        "==================================================\n\n",
        coverage_by_year.round(2).to_string(),
        "\n"
    ]
    
    with open(log_file, 'w') as f:
        f.writelines(log_content)
        
    print("\nVerarbeitung abgeschlossen!")
    print(f"[*] Plot Assets gespeichert unter:  {plot_path_at}")
    print(f"[*] Plot Revenue gespeichert unter: {plot_path_revt}")
    print(f"[*] Statistiken gespeichert unter:  {log_file}")

if __name__ == "__main__":
    main()