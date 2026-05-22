import pandas as pd
import torch
from data_loader import prepare_financial_data

def main():
    # 1. Datenpfad 
    data_path = "../data/ppreliminary_dataset_v1.csv"
    
    print(f"[*] Lade Daten von: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        # Fallback, falls das Skript aus dem Hauptordner ausgeführt wird
        df = pd.read_csv("data/preliminary_dataset_v1.csv")
        
    print(f"[*] Datensatz geladen. Shape: {df.shape}")

    # 2. Variablen definieren
    target_col = 'ESG_Combined_Score'
    
    cont_cols = [
        'total_assets', 'total_revenue', 'net_income', 
        'dltt', 'dlc', 'cash_equivalents', 'employees', 'total_debt'
    ]
    
    # IDs ('gvkey', 'conm', 'isin', 'orgpermid') lasse ich für das Feature-Mapping weg
    cat_cols = ['country', 'fyear'] 

    # 3. Fehlende Target-Werte droppen 
    initial_len = len(df)
    df = df.dropna(subset=[target_col])
    print(f"[*] {initial_len - len(df)} Zeilen mit fehlendem Target '{target_col}' entfernt. Verbleibend: {len(df)}")

    # 4. DataLoader aufrufen
    train_loader, test_loader, cat_cardinalities, d_numerical = prepare_financial_data(
        df=df,
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        target_col=target_col,
        batch_size=256 
    )

    # 5. Einen Batch inspizieren
    print("\n[*] Inspiziere den ersten Trainings-Batch:")
    for x_cont, x_cat, y in train_loader:
        print(f"    - Continuous Shape: {x_cont.shape} (Erwartet: [256, {len(cont_cols)}])")
        print(f"    - Categorical Shape: {x_cat.shape} (Erwartet: [256, {len(cat_cols)}])")
        print(f"    - Target Shape:     {y.shape} (Erwartet: [256, 1])")
        print(f"    - Berechnete Kardinalitäten (für Modell-Init): {cat_cardinalities}")
        break # Wir wollen nur den ersten Batch sehen

if __name__ == "__main__":
    main()