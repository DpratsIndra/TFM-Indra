import fitz  # PyMuPDF
import re
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class CTIReportProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )
        # Diccionario de IoCs
        self.patterns = {
            'IP_ADDRESS': r'\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
            'URL': r'\bhttps?://[^\s]+|www\.[^\s]+',
            'HASH': r'\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b',
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'DOMAIN': r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b'
        }

    def process_pdf(self, file_path):
        """Extrae, limpia y segmenta un PDF en documentos LangChain."""
        filename = os.path.basename(file_path)
        print(f"[*] Procesando: {filename}")
        
        doc = fitz.open(file_path)
        langchain_docs = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            
            # 1. Limpieza de IoCs (Privacidad y reducción de ruido)
            for label, pattern in self.patterns.items():
                text = re.sub(pattern, f"<{label}>", text)
            
            # 2. Segmentación (Chunking)
            chunks = self.splitter.split_text(text)
            
            # 3. Creación de Objetos Documento con Metadatos granulares
            for chunk in chunks:
                langchain_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "page": page_num + 1,
                        "type": "report_chunk"
                    }
                ))
        
        print(f"[+] {filename} segmentado en {len(langchain_docs)} fragmentos.")
        return langchain_docs