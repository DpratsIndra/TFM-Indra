import os
import pandas as pd

def main():
    """
    Convierte el archivo cti-ate.tsv de CTIBench en un CSV estándar
    compatible con el framework de evaluación.
    """
    tsv_path = "data/eval_datasets/ctibench/cti-ate.tsv"
    csv_path = "data/eval_datasets/ctibench/ctibench_mapping.csv"
    
    if not os.path.exists(tsv_path):
        print(f"[!] Error: {tsv_path} no existe.")
        return
        
    print(f"[*] Procesando {tsv_path}...")
    df = pd.read_csv(tsv_path, sep="\t")
    
    records = []
    for _, row in df.iterrows():
        text = str(row.get("Description", "")).strip()
        gt = str(row.get("GT", "")).strip()
        
        if not text or not gt or gt.lower() == "nan":
            continue
            
        # Limpiar técnicas (ej. 'T1071, T1573' -> ['T1071', 'T1573'])
        true_labels = [t.strip() for t in gt.split(",") if t.strip()]
        
        records.append({
            "text": text,
            "true_labels": ",".join(true_labels)
        })
        
    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    out_df.to_csv(csv_path, index=False)
    print(f"[*] ¡Éxito! Generado {csv_path} con {len(out_df)} registros.")

if __name__ == "__main__":
    main()
