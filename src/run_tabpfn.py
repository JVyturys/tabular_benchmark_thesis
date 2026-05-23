import torch
import pandas as pd
import numpy as np
import math
import time
from sklearn.metrics import mean_squared_error
from tabpfn import TabPFNRegressor
from data_loader import prepare_financial_data

def extract_arrays_from_loader(loader):
    """Hilfsfunktion: Entpackt den PyTorch DataLoader in zusammenhängende Numpy-Arrays für TabPFN."""
    X_cont_list, X_cat_list, y_list = [], [], []
    for x_cont, x_cat, y in loader:
        X_cont_list.append(x_cont.numpy())
        X_cat_list.append(x_cat.numpy())
        y_list.append(y.numpy())
        
    X_cont = np.concatenate(X_cont_list, axis=0)
    X_cat = np.concatenate(X_cat_list, axis=0)
    y = np.concatenate(y_list, axis=0).ravel()
    
    # Eine gesamte Feature-Matrix für TabFN
    X_combined = np.hstack((X_cont, X_cat))
    return X_combined, y

def main():
    print("[*] Bereite Daten für TabPFN Benchmark vor...")
    
    # 1. Daten laden 
    data_path = "../data/preliminary_dataset_v1.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        df = pd.read_csv("data/preliminary_dataset_v1.csv")
        
    target_col = 'ESG_Combined_Score'
    cont_cols = ['total_assets', 'total_revenue', 'net_income', 'dltt', 'dlc', 'cash_equivalents', 'employees', 'total_debt']
    cat_cols = ['country', 'fyear']
    
    df = df.dropna(subset=[target_col])
    

    
    train_loader, test_loader, _, _ = prepare_financial_data(
        df=df, cont_cols=cont_cols, cat_cols=cat_cols, target_col=target_col, batch_size=1024
    )

    X_train, y_train = extract_arrays_from_loader(train_loader)
    X_test, y_test = extract_arrays_from_loader(test_loader)
    
    print(f"[*] Trainings-Matrix Shape: {X_train.shape}")
    print(f"[*] Test-Matrix Shape: {X_test.shape}")

    # 2. TabPFN initialisieren
    print("\n[*] Initialisiere TabPFN-Foundation Model...")
    model = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu')

    # 3. In-Context Learning - Forward Pass
    print("[*] Führe In-Context Learning durch ...")
    start_time = time.time()
    
    # speichern der Trainingsdaten als Kontext
    model.fit(X_train, y_train)
    
    # Transformer-Forward-Pass über alle Paare
    predictions = model.predict(X_test)
    
    inference_time = time.time() - start_time

    # 4. Evaluierung
    mse = mean_squared_error(y_test, predictions)
    rmse = math.sqrt(mse)
    
    print("\n" + "="*40)
    print("TabPFN BENCHMARK ERGEBNISSE")
    print("="*40)
    print(f"Test MSE:  {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Inferenzzeit: {inference_time:.2f} Sekunden")
    print("="*40)

if __name__ == "__main__":
    main()