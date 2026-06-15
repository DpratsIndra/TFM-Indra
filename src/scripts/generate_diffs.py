import json
import sys
import os

def generate_diffs(json_path: str):
    if not os.path.exists(json_path):
        print(f"[!] File not found: {json_path}")
        return

    print(f"[*] Reading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    detailed_executions = data.get("detailed_executions", [])
    if not detailed_executions:
        print("[!] No 'detailed_executions' found in the JSON. Cannot generate diffs.")
        return

    diffs = []
    for item in detailed_executions:
        true_labels = item.get("true_labels", [])
        pred_labels = item.get("predicted_labels_normalized", [])
        
        true_set = set(true_labels)
        pred_set = set(pred_labels)
        
        if true_set != pred_set:
            diffs.append({
                "sentence_id": item.get("sentence_id", item.get("source_file", "unknown")),
                "text": item.get("text", ""),
                "human_ground_truth": list(true_set),
                "llm_prediction": list(pred_set),
                "false_negatives_missed": list(true_set - pred_set),
                "false_positives_hallucinated": list(pred_set - true_set)
            })

    # Generate output path
    dirname = os.path.dirname(json_path)
    basename = os.path.basename(json_path)
    
    # Prepend 'diff_analysis_' to the original filename
    diff_filename = f"diff_analysis_{basename}"
    output_path = os.path.join(dirname, diff_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diffs, f, indent=4)

    print(f"[+] Successfully generated diffs.")
    print(f"[+] Saved {len(diffs)} conflict cases for qualitative review in: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/scripts/generate_diffs.py <path_to_evaluation_json>")
        sys.exit(1)
        
    target_json = sys.argv[1]
    generate_diffs(target_json)
