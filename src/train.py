import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import rtdl
import math
from data_loader import prepare_financial_data

def main():
    # 1. HPC Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Nutze Device: {device}")
    
    # 2. Daten laden
    data_path = "../data/preliminary_dataset_v1.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        df = pd.read_csv("data/preliminary_dataset_v1.csv")
        
    target_col = 'ESG_Combined_Score'
    cont_cols = ['total_assets', 'total_revenue', 'net_income', 'dltt', 'dlc', 'cash_equivalents', 'employees', 'total_debt']
    cat_cols = ['country', 'fyear']
    
    df = df.dropna(subset=[target_col])
    
    train_loader, test_loader, cat_cardinalities, d_numerical = prepare_financial_data(
        df=df, cont_cols=cont_cols, cat_cols=cat_cols, target_col=target_col, batch_size=256
    )

    # 3. FT-Transformer initialisieren
    print("\n[*] Initialisiere Modell...")
    model = rtdl.FTTransformer.make_default(
        n_num_features=d_numerical,
        cat_cardinalities=cat_cardinalities,
        last_layer_query_idx=[-1],
        d_out=1
    ).to(device)

    # 4. AdamW Loss und Optimizer 
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) # Hyperparameter Optimierung?

    # 5. Trainingsschleife
    epochs = 5  # Später erweitern 
    print(f"\n[*] Starte Training für {epochs} Epochen...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for x_cont, x_cat, y in train_loader:
            x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)
            
            optimizer.zero_grad()
            predictions = model(x_cont, x_cat)
            loss = criterion(predictions, y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_cont.size(0)
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # Evaluierung auf dem Test-Set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_cont, x_cat, y in test_loader:
                x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)
                predictions = model(x_cont, x_cat)
                loss = criterion(predictions, y)
                test_loss += loss.item() * x_cont.size(0)
                
        avg_test_loss = test_loss / len(test_loader.dataset)
        test_rmse = math.sqrt(avg_test_loss)
        
        print(f"Epoche {epoch+1}/{epochs} | Train MSE: {avg_train_loss:.4f} | Test MSE: {avg_test_loss:.4f} | Test RMSE: {test_rmse:.4f}")

    print("\n[+] Training erfolgreich beendet.")

if __name__ == "__main__":
    main()