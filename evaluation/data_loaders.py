import os
import json
import pandas as pd

class BaseDataLoader:
    def load(self, file_path: str) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement this method")

class TramDataLoader(BaseDataLoader):
    """
    Loads MITRE TRAM dataset (multi_label.json format) into a standardized DataFrame.
    Expected output columns: 'text' (str), 'true_labels' (list of str).
    """
    def load(self, file_path: str) -> pd.DataFrame:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        records = []
        possible_keys = ['techniques', 'labels', 'technique_id', 'technique_ids']
        
        for item in data:
            # FIX: TRAM uses 'sentence' or 'text'
            text = item.get("sentence", item.get("text", "")).strip()
            if not text:
                continue
                
            raw_techs = []
            for key in possible_keys:
                if key in item:
                    raw_techs = item[key]
                    break
                    
            clean_techs = []
            for t in raw_techs:
                if isinstance(t, dict):  
                    t_id = t.get('technique_id', '')
                else:
                    t_id = str(t)
                    
                t_id = t_id.strip().upper()
                if t_id.startswith("T") and len(t_id) >= 5:
                    clean_techs.append(t_id)
            
            clean_techs = list(set(clean_techs))
            
            records.append({
                "text": text,
                "true_labels": clean_techs
            })
            
        df = pd.DataFrame(records)
        if not df.empty and 'text' in df.columns:
            df = df.dropna(subset=['text'])
        return df

class CtiHalDataLoader(BaseDataLoader):
    """
    Carga el dataset CTI-HAL.
    Lee el archivo dataset_mapping.csv que mapea PDFs con etiquetas.
    """
    def __init__(self, base_path: str = "data/eval_datasets/ctihal"):
        self.base_path = base_path
        self.csv_path = os.path.join(base_path, "dataset_mapping.csv")
        
    def load(self, file_path: str = None) -> pd.DataFrame:
        # Generate CSV if it doesn't exist
        if not os.path.exists(self.csv_path):
            print("[*] dataset_mapping.csv not found. Generating it now...")
            import subprocess
            import sys
            script_path = os.path.join(os.path.dirname(__file__), "generate_ctihal_csv.py")
            subprocess.run([sys.executable, script_path], check=True)
            
        df = pd.read_csv(self.csv_path)
        
        # Filter out PDFs without annotations
        df = df.dropna(subset=['md_path', 'true_labels'])
        
        records = []
        for _, row in df.iterrows():
            labels_str = str(row['true_labels']).strip()
            if not labels_str or labels_str == 'nan':
                continue
            
            true_labels = [l.strip() for l in labels_str.split(',') if l.strip()]
            records.append({
                "source_file": row['pdf_path'],
                "true_labels": true_labels
            })
            
            
        return pd.DataFrame(records)

class AptReportDataLoader(BaseDataLoader):
    """
    Carga el dataset APT_REPORT.
    Usa el nombre de la carpeta (nombre del grupo APT) para buscar en la base de datos STIX
    de MITRE ATT&CK todas las técnicas históricamente atribuidas a ese grupo.
    Estas técnicas se usarán como las 'true_labels' para los reportes de esa carpeta.
    """
    def __init__(self, base_path: str = "data/eval_datasets/APT_REPORT"):
        self.base_path = base_path
        self.mapping_file = os.path.join(base_path, "mitre_group_mapping.json")
        
    def load(self, file_path: str = None) -> pd.DataFrame:
        if not os.path.exists(self.mapping_file):
            print(f"[*] Mapping file not found at {self.mapping_file}. Generating it now...")
            import subprocess
            import sys
            script_path = os.path.join(os.path.dirname(__file__), "generate_apt_mapping.py")
            subprocess.run([sys.executable, script_path], check=True)
            
        with open(self.mapping_file, "r") as f:
            group_mapping = json.load(f)
            
        records = []
        
        # Recorrer todas las carpetas dentro de APT_REPORT
        for item in os.listdir(self.base_path):
            dir_path = os.path.join(self.base_path, item)
            if not os.path.isdir(dir_path) or item.startswith("."):
                continue
                
            # Normalizar el nombre de la carpeta para buscar el alias
            alias = item.lower().strip()
            
            # Buscar coincidencia exacta o parcial estricta
            matched_techniques = []
            if alias in group_mapping:
                matched_techniques = group_mapping[alias]
            else:
                # Intento de matching estricto ignorando guiones, espacios y mayúsculas
                clean_alias = alias.replace("-", "").replace(" ", "").replace("_", "")
                for map_alias, techs in group_mapping.items():
                    clean_map_alias = map_alias.replace("-", "").replace(" ", "").replace("_", "")
                    if clean_alias == clean_map_alias:
                        matched_techniques = techs
                        break
                        
            if not matched_techniques:
                continue # Saltamos las carpetas de las que no tenemos técnicas en MITRE
                
            # Robustly find all PDFs in this directory
            pdf_files = []
            for root_dir, _, files in os.walk(dir_path):
                for f in files:
                    if f.lower().endswith(".pdf"):
                        pdf_files.append(os.path.join(root_dir, f))
                        
            for pdf in pdf_files:
                records.append({
                    "source_file": pdf,
                    "true_labels": matched_techniques,
                    "group_name": item
                })
                
        df = pd.DataFrame(records)
        print(f"[*] AptReportDataLoader loaded {len(df)} PDFs with MITRE Group mappings.")
        return df

class CtibenchDataLoader(BaseDataLoader):
    """
    Carga el dataset CTIBench (cti-ate.tsv) convertido a CSV.
    A diferencia de CTI-HAL, este dataset contiene descripciones cortas (text)
    y no ficheros PDF completos.
    """
    def __init__(self, base_path: str = "data/eval_datasets/ctibench"):
        self.base_path = base_path
        self.csv_path = os.path.join(base_path, "ctibench_mapping.csv")
        
    def load(self, file_path: str = None) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            print("[*] ctibench_mapping.csv not found. Generating it now...")
            import subprocess
            import sys
            script_path = os.path.join(os.path.dirname(__file__), "generate_ctibench_csv.py")
            subprocess.run([sys.executable, script_path], check=True)
            
        df = pd.read_csv(self.csv_path)
        records = []
        for _, row in df.iterrows():
            labels_str = str(row['true_labels']).strip()
            if not labels_str or labels_str == 'nan':
                continue
                
            true_labels = [l.strip() for l in labels_str.split(',') if l.strip()]
            records.append({
                "text": row['text'],
                "true_labels": true_labels
            })
            
        return pd.DataFrame(records)
