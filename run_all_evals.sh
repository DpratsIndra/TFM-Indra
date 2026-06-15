#!/bin/bash

# Script para ejecutar todas las configuraciones de evaluación (Langchain y Langgraph)
# con las diferentes combinaciones de VLM y PROMPT REPETITION

# Puedes cambiar el script a evaluar, por defecto usará run_eval.py (TRAM)
# Otros ejemplos: evaluation/run_eval_ctihal.py, evaluation/run_eval_apt.py
EVAL_SCRIPT=${1:-"evaluation/run_eval.py"}

echo "======================================================="
echo "Iniciando suite de evaluaciones en: $EVAL_SCRIPT"
echo "======================================================="

run_eval() {
    local cmd="$@"
    eval "$cmd"
    local exit_code=$?
    if [ $exit_code -eq 429 ]; then
        echo -e "\n[❌] RATE LIMIT DETECTED (Exit code 429). Stopping suite execution to preserve checkpoint."
        exit 429
    elif [ $exit_code -ne 0 ] && [ $exit_code -ne 130 ]; then
        echo -e "\n[⚠️] Command failed with exit code $exit_code: $cmd"
    fi
}

# --- LANGCHAIN ---
echo "[*] Ejecutando LANGCHAIN: Nada (VLM=False, Repetition=False)"
run_eval "USE_VLM_EXTRACTION=False USE_PROMPT_REPETITION=False PYTHONPATH=. python \"$EVAL_SCRIPT\" --pipeline langchain"

echo "[*] Ejecutando LANGCHAIN: Solo VLM (VLM=True, Repetition=False)"
run_eval "USE_VLM_EXTRACTION=True USE_PROMPT_REPETITION=False PYTHONPATH=. python \"$EVAL_SCRIPT\" --pipeline langchain"

echo "[*] Ejecutando LANGCHAIN: Solo Repetition (VLM=False, Repetition=True)"
run_eval "USE_VLM_EXTRACTION=False USE_PROMPT_REPETITION=True PYTHONPATH=. python \"$EVAL_SCRIPT\" --pipeline langchain"

echo "[*] Ejecutando LANGCHAIN: Ambas (VLM=True, Repetition=True)"
run_eval "USE_VLM_EXTRACTION=True USE_PROMPT_REPETITION=True PYTHONPATH=. python \"$EVAL_SCRIPT\" --pipeline langchain"

# --- LANGGRAPH ---
echo "[*] Ejecutando LANGGRAPH: Nada (VLM=False, Repetition=False)"
run_eval "USE_VLM_EXTRACTION=False USE_PROMPT_REPETITION=False PYTHONPATH=. python \"$EVAL_SCRIPT\" --pipeline langgraph"

echo "[*] Ejecutando LANGGRAPH: Solo VLM (VLM=True, Repetition=False)"
run_eval "USE_VLM_EXTRACTION=True USE_PROMPT_REPETITION=False PYTHONPATH=. python \"$EVAL_SCRIPT\" --pipeline langgraph"

echo "[*] Ejecutando LANGGRAPH: Solo Repetition (VLM=False, Repetition=True)"
run_eval "USE_VLM_EXTRACTION=False USE_PROMPT_REPETITION=True PYTHONPATH=. python \"$EVAL_SCRIPT\" --pipeline langgraph"

echo "[*] Ejecutando LANGGRAPH: Ambas (VLM=True, Repetition=True)"
run_eval "USE_VLM_EXTRACTION=True USE_PROMPT_REPETITION=True PYTHONPATH=. python \"$EVAL_SCRIPT\" --pipeline langgraph"

echo "======================================================="
echo "Suite de evaluaciones finalizada."
echo "======================================================="

