import torch
import rtdl

def main():
    # 1. Hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Nutze Device: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    # 2. Dummy-Parameter für sythetischen Test-Datensatz definieren
    batch_size = 32
    n_cont_features = 5
    cat_cardinalities = [3, 10, 4]  # Drei kategoriale Spalten mit 3, 10 und 4 Ausprägungen
    d_out = 1  # Vorhersage-Dimension

    # 3. Modell initialisieren (FT-Transformer)
    print("[*] Initialisiere FT-Transformer...")
    model = rtdl.FTTransformer.make_default(
        n_num_features=n_cont_features,
        cat_cardinalities=cat_cardinalities,
        last_layer_query_idx=[-1],  # Nutzt das [CLS]-Token für die Vorhersage
        d_out=d_out
    )
    model.to(device)

    # 4. Synthetische Tabellendaten generieren
    x_cont = torch.randn(batch_size, n_cont_features).to(device)
    x_cat = torch.tensor([
        [0, 5, 2] for _ in range(batch_size) # Random Kategorien innerhalb der Kardinalitäten
    ]).to(device)

    # 5. Forward Pass
    print("[*] Führe Forward Pass durch...")
    output = model(x_cont, x_cat)
    
    print(f"[+] Erfolgreich! Output Shape: {output.shape} (Erwartet: [{batch_size}, {d_out}])")

if __name__ == "__main__":
    main()