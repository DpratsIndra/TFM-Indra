import os
import torch
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings

def run_debug():
    print("[*] Conectando a Qdrant y cargando modelos...")
    qdrant_url = "http://localhost:6333"
    client = QdrantClient(url=qdrant_url)
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="mitre_attack",
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID
    )
    
    reranker = CrossEncoder('BAAI/bge-reranker-base', device='cpu')
    
    # Textos extraídos de tus logs de LangSmith
    chunk_text = """Figure.3 “DIPLOMATIC-CAR-FOR-SALE-BMW.pdf” lure document.
Figure.4 PowerShell script deploying .pdf lure and downloading next-stage payload from <IoC_DOMAIN>.
Ngrok, at its core, is an incredibly versatile and cross-platform tool designed to expose local network ports securely to the internet through a process known as tunneling. However, in the context of cyber adversaries, Ngrok has taken on a diﬀerent role. Instead of legitimate purposes, adversaries have begun leveraging Ngrok to store their next-stage PowerShell payloads and establish covert communication channels.
In this nefarious tactic, they utilize Ngrok's services by utilizing free static domains provided by Ngrok, typically in the form of a subdomain under "<IoC_DOMAIN>." These subdomains act as discrete and inconspicuous rendezvous points for their malicious payloads. This clever adaptation allows the adversaries to obfuscate their activities and communicate with compromised systems while evading detection. By exploiting Ngrok's capabilities in this manner, threat actors can further complicate cybersecurity eﬀorts and remain under the radar, making defense and attribution more challenging.
6: HKLUK"""
    
    abstract_keywords = "Spearphishing Attachment, Remote File Download, Protocol Tunneling, Command and Control, Obfuscation, Payload Delivery, Living off the Land"
    
    # Esta es la query exacta que tu Extractor está enviando a mitre_oracle
    search_query = f"Raw Text: {chunk_text}\nKeywords: {abstract_keywords}"
    
    print("\n" + "="*50)
    print("1. BÚSQUEDA HÍBRIDA EN QDRANT (TOP 15)")
    print("="*50)
    
    # Obtenemos candidatos con su score de Qdrant
    candidates_with_scores = vector_store.similarity_search_with_score(search_query, k=15)
    
    t1572_found_in_qdrant = False
    
    for i, (doc, score) in enumerate(candidates_with_scores):
        tech_id = doc.metadata.get('technique_id', 'Unknown')
        name = doc.metadata.get('name', 'Unknown')
        print(f"{i+1}. {tech_id} - {name} (Qdrant Score: {score:.4f})")
        if tech_id == 'T1572':
            t1572_found_in_qdrant = True

    if not t1572_found_in_qdrant:
        print("\n[!] ALERTA CRÍTICA: T1572 (Protocol Tunneling) NO FUE RECUPERADA POR QDRANT EN EL TOP 15.")
        print("[!] Esto significa que el fallo está en la indexación de Qdrant o que el texto de la query 'diluye' a BM25.")
        
    print("\n" + "="*50)
    print("2. EVALUACIÓN DEL CROSS-ENCODER (RERANKER)")
    print("="*50)
    
    all_pairs = [[search_query, doc.page_content] for doc, _ in candidates_with_scores]
    
    if all_pairs:
        # Extraemos logits brutos primero para ver qué está pensando el modelo matemáticamente
        raw_logits = reranker.predict(all_pairs, batch_size=16)
        # Extraemos probabilidades con Sigmoide
        probabilities = reranker.predict(all_pairs, batch_size=16, activation_fn=torch.nn.Sigmoid())
        
        results = []
        for i, (doc, _) in enumerate(candidates_with_scores):
            tech_id = doc.metadata.get('technique_id', 'Unknown')
            name = doc.metadata.get('name', 'Unknown')
            results.append({
                "id": tech_id, 
                "name": name, 
                "logit": raw_logits[i], 
                "prob": probabilities[i]
            })
            
        # Ordenar por probabilidad
        results.sort(key=lambda x: x["prob"], reverse=True)
        
        for res in results:
            marker = "--> [PASA EL CORTE >= 0.15]" if res["prob"] >= 0.15 else "    [RECHAZADO]"
            print(f"{marker} {res['id']} - {res['name']} | Logit: {res['logit']:.4f} | Prob(Sigmoid): {res['prob']:.4f}")

if __name__ == "__main__":
    run_debug()
