import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class FinancialDataset(Dataset):
    """PyTorch Dataset Wrapper für tabellarische Finanzdaten."""
    def __init__(self, x_cont, x_cat, y):
        # Konvertiert Numpy-Arrays in PyTorch Tensoren
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_cont[idx], self.x_cat[idx], self.y[idx]


def prepare_financial_data(df: pd.DataFrame, cont_cols: list, cat_cols: list, target_col: str, batch_size=64):
    """
    Führt Imputation, Skalierung und Encoding durch und gibt PyTorch DataLoaders zurück.
    """
    print("[*] Starte Pre-Processing der Finanzdaten...")
    
    # 1. Train/Test Split (80/20)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    y_train = df_train[target_col].values
    y_test = df_test[target_col].values
    
    # 2. Kontinuierliche Features verarbeiten (Imputation + Scaling) to do: Imputation weiterführend durch GAIN modell ersetzen?
    cont_imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_cont_train = scaler.fit_transform(cont_imputer.fit_transform(df_train[cont_cols]))
    X_cont_test = scaler.transform(cont_imputer.transform(df_test[cont_cols]))
    
    # 3. Kategoriale Features verarbeiten (Imputation + Ordinal Encoding)
    cat_imputer = SimpleImputer(strategy='most_frequent')
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    X_cat_train = encoder.fit_transform(cat_imputer.fit_transform(df_train[cat_cols]))
    X_cat_test = encoder.transform(cat_imputer.transform(df_test[cat_cols]))
    
    # Unbekannte Werte (-1) auf 0 setzen und Kardinalitäten anpassen 
    # (Kategorie 0 als "Unknown/Missing" reserviert im Modell)
    X_cat_train = np.maximum(X_cat_train, 0)
    X_cat_test = np.maximum(X_cat_test, 0)
    
    # Kardinalitäten berechnen (für den FT-Transformer)
    cat_cardinalities = [len(encoder.categories_[i]) + 1 for i in range(len(cat_cols))]
    
    # 4. PyTorch DataLoaders erstellen
    train_dataset = FinancialDataset(X_cont_train, X_cat_train, y_train)
    test_dataset = FinancialDataset(X_cont_test, X_cat_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"[+] Pre-Processing abgeschlossen. Features: {len(cont_cols)} cont, {len(cat_cols)} cat.")
    
    return train_loader, test_loader, cat_cardinalities, X_cont_train.shape[1]