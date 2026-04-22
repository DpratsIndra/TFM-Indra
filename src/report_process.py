import fitz
import re
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    print(f'Extrayendo texto del PDF: {pdf_path}')
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"El archivo {pdf_path} no existe.")
    
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text('text') + "\n"
    return text
    
def safe_iocs(text):
    # Sustituye los IOCs por etiquetas genéricas para proteger la información sensible y evitar confundir al modelo de embeddings
    print('Sustituyendo IOCs por etiquetas genéricas...')
    # Expresiones regulares para detectar IPs (IPv4 e IPv6), URLs, hashes, correos electrónicos y dominios
    patterns = {
        'IP_ADDRESS': r'\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
        'URL': r'\bhttps?://[^\s]+|www\.[^\s]+',
        'HASH': r'\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b',
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'DOMAIN': r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b'
    }  
    for label, pattern in patterns.items():
        text = re.sub(pattern, f'[{label}]', text)
    return text

def text_split(text, chunk_size=1000, chunk_overlap=200):
    print(f'Segmentando el texto en fragmentos de {chunk_size} caracteres con un solapamiento de {chunk_overlap} caracteres...')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    print(f'Texto segmentado en {len(chunks)} fragmentos.')
    return chunks

def main():
    pdf_path = 'data/APT29 attacks Embassies using CVE-2023-38831 - report en.pdf'
    try:
        text = extract_text_from_pdf(pdf_path)
        text = safe_iocs(text)
        chunks = text_split(text)
        
        print(f'\nPrimer fragmento de texto:\n{chunks[0]}')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    main()