import os
import glob
import re
import difflib
import pandas as pd

def generate_mapping_csv(base_path: str = "data/eval_datasets/ctihal"):
    reports_dir = os.path.join(base_path, "reports")
    annotations_dir = os.path.join(base_path, "annotations")
    output_csv = os.path.join(base_path, "dataset_mapping.csv")
    
    # Manual overrides para los PDFs que no encajan por nombre
    manual_overrides = {
        "New Targeted Attack in the Middle East by APT34": "oilrig-mandiant.md",
        "Mahalo FIN7": "fin7-fireeye-1.md"
    }
    
    records = []
    
    # Buscar recursivamente todos los PDFs crudos
    pdf_pattern = os.path.join(reports_dir, "**", "*.pdf")
    pdf_files = glob.glob(pdf_pattern, recursive=True)
    
    for pdf_path in pdf_files:
        rel_path = os.path.relpath(pdf_path, reports_dir)
        parts = rel_path.split(os.sep)
        
        if len(parts) < 2:
            continue
            
        apt_name = parts[0]
        pdf_filename = parts[1]
        
        # Buscar el MD correspondiente en "annotator L"
        md_dir = os.path.join(annotations_dir, apt_name, "annotator L")
        if not os.path.exists(md_dir):
            # Fallback al directorio raíz de la APT si no hay "annotator L"
            md_dir = os.path.join(annotations_dir, apt_name)
            if not os.path.exists(md_dir):
                records.append({"pdf_path": pdf_path, "md_path": None, "apt_name": apt_name, "true_labels": ""})
                continue
            
        md_files = glob.glob(os.path.join(md_dir, "*.md"))
        if not md_files:
            records.append({"pdf_path": pdf_path, "md_path": None, "apt_name": apt_name, "true_labels": ""})
            continue
            
        # Extraer nombres base para hacer matching
        pdf_base = os.path.splitext(pdf_filename)[0].lower().replace(" ", "").replace("_", "")
        md_bases = [os.path.splitext(os.path.basename(m))[0].lower() for m in md_files]
        clean_md_bases = [m.replace(f"{apt_name.lower()}-", "").replace("-", "") for m in md_bases]
        
        matched_md_path = None
        
        # Estrategia 0: Manual Overrides
        for override_key, override_md in manual_overrides.items():
            if override_key.lower().replace(" ", "") in pdf_base:
                override_full_path = os.path.join(md_dir, override_md)
                if os.path.exists(override_full_path):
                    matched_md_path = override_full_path
                break
                
        # Estrategia 1: Substring Matching (muy efectivo aquí)
        if not matched_md_path:
            sorted_indices = sorted(range(len(clean_md_bases)), key=lambda k: len(clean_md_bases[k]), reverse=True)
            for i in sorted_indices:
                clean_base = clean_md_bases[i]
                if clean_base and clean_base in pdf_base:
                    matched_md_path = md_files[i]
                    break
                
        # Estrategia 2: Fallback a difflib difuso relajado si no hay match
        if not matched_md_path:
            matches = difflib.get_close_matches(pdf_base, clean_md_bases, n=1, cutoff=0.15)
            if matches:
                matched_md_path = md_files[clean_md_bases.index(matches[0])]

        if matched_md_path:
            # Extraer las etiquetas verdaderas (MITRE IDs)
            try:
                with open(matched_md_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                pattern = r'\bT\d{4}(?:\.\d{3})?\b'
                true_labels = list(set(re.findall(pattern, content)))
                
                records.append({
                    "pdf_path": pdf_path,
                    "md_path": matched_md_path,
                    "apt_name": apt_name,
                    "true_labels": ",".join(true_labels)
                })
            except Exception as e:
                print(f"[!] Error procesando MD {matched_md_path}: {e}")
                records.append({"pdf_path": pdf_path, "md_path": matched_md_path, "apt_name": apt_name, "true_labels": ""})
        else:
            records.append({"pdf_path": pdf_path, "md_path": None, "apt_name": apt_name, "true_labels": ""})

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"[+] CTIHAL dataset mapping generated successfully at {output_csv}")
    print(f"    Total PDFs processed: {len(pdf_files)}")
    print(f"    Successful MD matches: {len(df[df['md_path'].notnull()])}")
    return output_csv

if __name__ == "__main__":
    generate_mapping_csv()
