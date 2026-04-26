import sys
import os

# 1. Obtenemos la ruta absoluta de la carpeta donde está este script (pruebas)
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# 2. Subimos un nivel y apuntamos a la carpeta 'src'
carpeta_src = os.path.abspath(os.path.join(directorio_actual, '..', 'src'))

# 3. Añadimos 'src' al path de Python para que encuentre nuestros módulos
if carpeta_src not in sys.path:
    sys.path.append(carpeta_src)

# --- AHORA YA PUEDES IMPORTAR TUS MÓDULOS CON NORMALIDAD ---
from report_process import CTIReportProcessor
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

def main():
    # 1. Configuración del entorno
    pdf_path = "data/APT29 attacks Embassies using CVE-2023-38831 - report en.pdf" # Asegúrate de que existe
    collection_name = "mitre_attack_techniques"
    
    # 2. Inicializar modelos y base de datos
    print("[*] Cargando modelo de embeddings y conectando a Qdrant...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    client = QdrantClient(url="http://localhost:6333")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )

    # 3. Procesar el PDF usando la nueva granularidad
    processor = CTIReportProcessor()
    report_chunks = processor.process_pdf(pdf_path)

    # 4. Búsqueda de TTPs por cada fragmento del reporte
    print(f"\n--- INICIANDO MAPEO TÁCTICO AUTOMATIZADO ---")
    
    # Para no saturar la pantalla, analizaremos los primeros 3 fragmentos
    for i, chunk in enumerate(report_chunks[:3]):
        print(f"\n>>> Analizando fragmento {i+1} (Pág. {chunk.metadata['page']}):")
        print(f"Texto: {chunk.page_content[:150]}...") # Mostramos un resumen del texto
        
        # Consultamos a Qdrant
        results = vector_store.similarity_search(chunk.page_content, k=2)
        
        print("TTPs Candidatas encontradas:")
        for res in results:
            print(f"  - [{res.metadata['id']}] {res.metadata['name']} (Similitud detectada)")

if __name__ == "__main__":
    main()