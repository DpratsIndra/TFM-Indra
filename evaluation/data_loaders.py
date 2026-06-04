import os
import glob
import re
import difflib
import json
import pandas as pd
from typing import List

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
    Empareja reportes PDF CRUDOS (carpeta 'reports') con sus anotaciones Markdown (.md)
    de la carpeta 'annotations' (Ground Truth).
    """
    def __init__(self, base_path: str = "data/eval_datasets/ctihal"):
        self.base_path = base_path
        self.reports_dir = os.path.join(base_path, "reports")
        self.annotations_dir = os.path.join(base_path, "annotations")
        
    def load(self, file_path: str = None) -> pd.DataFrame:
        records = []
        
        # Buscar recursivamente todos los PDFs crudos
        pdf_pattern = os.path.join(self.reports_dir, "**", "*.pdf")
        pdf_files = glob.glob(pdf_pattern, recursive=True)
        
        for pdf_path in pdf_files:
            rel_path = os.path.relpath(pdf_path, self.reports_dir)
            parts = rel_path.split(os.sep)
            
            if len(parts) < 2:
                continue
                
            apt_name = parts[0]
            pdf_filename = parts[1]
            
            # Buscar el MD correspondiente en "annotator L"
            md_dir = os.path.join(self.annotations_dir, apt_name, "annotator L")
            if not os.path.exists(md_dir):
                continue
                
            md_files = glob.glob(os.path.join(md_dir, "*.md"))
            if not md_files:
                continue
                
            # Extraer nombres base para hacer matching difuso (ignorando prefijos como apt29-)
            pdf_base = os.path.splitext(pdf_filename)[0].lower()
            md_bases = [os.path.splitext(os.path.basename(m))[0].lower() for m in md_files]
            clean_md_bases = [m.replace(f"{apt_name.lower()}-", "") for m in md_bases]
            
            # Match difuso
            matches = difflib.get_close_matches(pdf_base, clean_md_bases, n=1, cutoff=0.3)
            if not matches:
                matches = difflib.get_close_matches(pdf_base, md_bases, n=1, cutoff=0.3)
                
            if matches:
                best_match = matches[0]
                match_idx = clean_md_bases.index(best_match) if best_match in clean_md_bases else md_bases.index(best_match)
                matched_md_path = md_files[match_idx]
                
                # Extraer las etiquetas verdaderas (MITRE IDs)
                try:
                    with open(matched_md_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    pattern = r'\bT\d{4}(?:\.\d{3})?\b'
                    true_labels = list(set(re.findall(pattern, content)))
                    
                    if true_labels:
                        records.append({
                            "source_file": pdf_path,
                            "true_labels": true_labels
                        })
                except Exception as e:
                    print(f"[!] Error procesando MD {matched_md_path}: {e}")
                    
        return pd.DataFrame(records)
