import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Starte Generierung der Dichteplots für die Zielvariablen...")
    
    # Pfade
    input_path = "data/03_processed/thesis_base_dataset_q_ftdrop.csv"
    output_dir = "results/02_desc_analysis_esg/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Daten laden
    try:
        df = pd.read_csv(input_path, low_memory=False)
        print(f"Datensatz erfolgreich geladen ({len(df)} Zeilen).")
    except FileNotFoundError:
        print(f"Fehler: Datei {input_path} nicht gefunden.")
        return

    # Definition der Zielvariablen und deren Beschriftung
    targets = {
        'esg_score': 'Gesamt-ESG-Score',
        'env_score': 'Umwelt-Score (E-Pillar)',
        'soc_score': 'Sozial-Score (S-Pillar)',
        'gov_score': 'Governance-Score (G-Pillar)'
    }
    
    # Design
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 14
    })
    
    # ---------------------------------------------------------
    # 1. Kombinierter Dichteplot (KDE Comparison)
    # ---------------------------------------------------------
    print("Erstelle kombinierten Dichteplot...")
    plt.figure(figsize=(10, 6), dpi=300)
    
    for col, label in targets.items():
        if col in df.columns:
            # alpha steuert die Transparenz der Flächen
            sns.kdeplot(data=df, x=col, label=label, fill=True, alpha=0.12, linewidth=2)
            
    plt.title('Vergleich der Verteilungsdichten (KDE) aller Refinitiv ESG-Scores', pad=15, weight='bold')
    plt.xlabel('Score-Wert (0 bis 100)')
    plt.ylabel('Wahrscheinlichkeitsdichte')
    plt.xlim(0, 1)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    
    combined_path = f"{output_dir}/esg_combined_density.png"
    plt.savefig(combined_path, bbox_inches='tight')
    plt.close()
    print(f" -> Kombinationsplot gespeichert unter: {combined_path}")
    
    # ---------------------------------------------------------
    # 2. Einzelne Detailplots (Histogramm + KDE + Metriken)
    # ---------------------------------------------------------
    for col, label in targets.items():
        if col in df.columns:
            print(f"Erstelle Einzelplot für {label}...")
            plt.figure(figsize=(8, 5), dpi=300)
            
            # Kennzahlen für Linien berechnen
            mean_val = df[col].mean()
            median_val = df[col].median()
            
            # Kombination aus dezentem Histogramm (Bins) und glatter KDE-Linie
            sns.histplot(data=df, x=col, kde=True, stat="density", fill=True, alpha=0.35, bins=40)
            
            # Vertikale Linien einzeichnen
            plt.axvline(mean_val, linestyle='--', linewidth=2, label=f'Mittelwert: {mean_val:.2f}')
            plt.axvline(median_val, linestyle=':', linewidth=2, label=f'Median: {median_val:.2f}')
            
            plt.title(f'Verteilungscharakteristik: {label}', pad=15, weight='bold')
            plt.xlabel('Score (0 bis 100)')
            plt.ylabel('Dichte')
            plt.xlim(0, 1)
            plt.legend(frameon=True, facecolor='white')
            plt.tight_layout()
            
            single_path = f"{output_dir}/density_individual_{col}.png"
            plt.savefig(single_path, bbox_inches='tight')
            plt.close()
            print(f" [*] Einzelplot gespeichert unter: {single_path}")
            
    print("\nGrafiken wurden erfolgreich in generiert.")

if __name__ == "__main__":
    main()