import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import math

def serialize_row(row):
    """Übersetzt eine tabellarische Datenzeile in einen natürlichen Satz für das LLM."""
    text = (
        f"In the fiscal year {row['fyear']}, the company located in {row['country']} "
        f"reported the following financial metrics: Total Assets of {row['total_assets']:.2f}, "
        f"Total Revenue of {row['total_revenue']:.2f}, and Net Income of {row['net_income']:.2f}. "
        f"Long-term debt (dltt) was {row['dltt']:.2f}, and short-term debt (dlc) was {row['dlc']:.2f}. "
        f"The company held Cash Equivalents of {row['cash_equivalents']:.2f}, "
        f"had a Total Debt of {row['total_debt']:.2f}, and employed {row['employees']} people. "
        f"Based on these fundamentals, the predicted ESG Combined Score is:"
    )
    return text

def compute_metrics(eval_pred):
    """Berechnet MSE und RMSE für die Evaluierung."""
    predictions, labels = eval_pred
    mse = ((predictions.squeeze() - labels) ** 2).mean()
    return {"mse": mse, "rmse": math.sqrt(mse)}

def main():
    print("[*] Starte TabLLM - LoRA Regression - Pipeline - PoC Modus...")
    
    # 1. Daten laden und vorbereiten
    data_path = "../data/preliminary_dataset_v1.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        df = pd.read_csv("data/preliminary_dataset_v1.csv")
        
    df = df.dropna(subset=['ESG_Combined_Score']).copy()
    
    features = ['fyear', 'country', 'total_assets', 'total_revenue', 'net_income', 'dltt', 'dlc', 'cash_equivalents', 'total_debt', 'employees']
    df[features] = df[features].fillna(0)
    
    # TESTMODUS: Subsample n = 5000
    print("[*] Subsampling: reduziere auf 5.000 Zeilen...")
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)


    # 2. Tabellendaten in Text serialisieren
    print("[*] Übersetze Tabellendaten in Text-Prompts...")
    df['text_prompt'] = df.apply(serialize_row, axis=1)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_dataset = Dataset.from_pandas(train_df[['text_prompt', 'ESG_Combined_Score']].rename(columns={"ESG_Combined_Score": "label"}))
    test_dataset = Dataset.from_pandas(test_df[['text_prompt', 'ESG_Combined_Score']].rename(columns={"ESG_Combined_Score": "label"}))

    # 3. Modell und Tokenizer laden
    model_id = "mistralai/Mistral-7B-v0.1" 
    print(f"[*] Lade Foundation Model: {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token 
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=1, 
        problem_type="regression",
        torch_dtype=torch.float16, 
        device_map="auto" 
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # 4. LoRA-Konfiguration
    print("[*] Injiziere LoRA-Adapter für Regression...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        r=16, 
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"] 
    )
    model = get_peft_model(model, peft_config)

    # Konvertiert nur die trainierbaren LoRA-Parameter in float32
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    # 5. Daten tokenisieren
    def tokenize_function(examples):
        return tokenizer(examples["text_prompt"], padding="max_length", truncation=True, max_length=256)

    print("[*] Tokenisiere Datensatz...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # 6. Training konfigurieren
    training_args = TrainingArguments(
        output_dir="./tabllm_results",
        learning_rate=2e-4, 
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    # 7. Start
    print("[*] Starte LoRA Fine-Tuning...")
    trainer.train()
    
    print("\n[+] TabLLM Benchmark erfolgreich beendet!")
    print(trainer.evaluate())

if __name__ == "__main__":
    main()