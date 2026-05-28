import os
import re
from typing import List

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

from src.core.ioc_masker import IoCMasker


class ReportIngestor:
    """
    Handles Phase 1 of the LangChain pipeline: Ingestion.
    Responsible for loading PDF reports, filtering noise (headers, footers, boilerplate),
    reconstructing structural Markdown, masking IoCs, and chunking semantically.
    """

    def __init__(self, chunk_size: int = 750, chunk_overlap: int = 150) -> None:
        """
        Initializes the ReportIngestor with chunking parameters and IoC masker.
        
        Args:
            chunk_size (int): The maximum character size for each chunk.
            chunk_overlap (int): Overlap in characters to maintain context.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ioc_masker = IoCMasker()
        
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on, strip_headers=False)
        
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        self.boilerplate_keywords = [
            "subscribe", "share on", "copyright ©", "copyright c",
            "all rights reserved", "manage cookies"
        ]

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Loads a PDF file and extracts its elements using Unstructured.
        Uses hi_res strategy for OCR and table extraction.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
            
        import logging
        logger = logging.getLogger(__name__)
        
        # Optional: fast page check using PyMuPDF to validate extraction later
        try:
            import fitz
            pdf_doc = fitz.open(file_path)
            num_pages = len(pdf_doc)
            pdf_doc.close()
        except Exception:
            num_pages = 1
            
        logger.info(f"Loading PDF '{file_path}' using Unstructured hi_res strategy (Pages detected: {num_pages})...")
            
        # Using mode='elements', hi_res strategy for OCR, and inferring tables
        loader = UnstructuredPDFLoader(
            file_path, 
            mode="elements", 
            strategy="hi_res",
            pdf_infer_table_structure=True
        )
        elements = loader.load()
        
        # Validation Logic: Suspiciously short extraction
        total_text_len = sum(len(el.page_content) for el in elements)
        if num_pages >= 3 and total_text_len < (num_pages * 50):
            logger.warning(
                f"🚨 ALERT: Extracted text is suspiciously short ({total_text_len} chars across {num_pages} pages). "
                "The PDF might be a pure scanned image requiring forced full-page OCR or it's heavily obfuscated."
            )
            
        return elements

    def process_report(self, file_path: str) -> List[Document]:
        """
        Main method to process a CTI report:
        1. Loads the PDF via unstructured elements.
        2. Filters out noise and boilerplate.
        3. Constructs a Markdown representation.
        4. Masks IoCs.
        5. Splits the text logically using Markdown headers & Recursive fallback.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        raw_elements = self.load_pdf(file_path)
        
        logger.info("[FASE 1] Limpiando ruido y reconstruyendo estructura Markdown...")
        md_lines = []
        for el in raw_elements:
            cat = el.metadata.get("category", "NarrativeText")
            text = el.page_content.strip()
            page_num = el.metadata.get("page_number", "?")
            
            if not text:
                continue
                
            # Filter Headers and Footers
            if cat in ["Header", "Footer"]:
                continue
                
            # Filter Boilerplate and short NarrativeText
            if cat == "NarrativeText":
                lower_text = text.lower()
                if any(bp in lower_text for bp in self.boilerplate_keywords):
                    continue
                if len(text.split()) < 5:
                    continue
                    
            # Handle Tables
            if cat == "Table":
                text = el.metadata.get("text_as_html", text)
                
            # Format as Markdown and inject hidden page tracker
            if cat == "Title":
                md_lines.append(f"\n## {text} <!-- Page {page_num} -->\n")
            elif cat == "ListItem":
                md_lines.append(f"- {text} <!-- Page {page_num} -->")
            else:
                md_lines.append(f"{text} <!-- Page {page_num} -->\n")
                
        full_md = "\n".join(md_lines)
        
        logger.info("[FASE 1] Enmascarando Indicadores de Compromiso (IoCs) volátiles...")
        # Mask IoCs on the combined text
        sanitized_md = self.ioc_masker.mask_text(full_md)
        
        logger.info("[FASE 1] Aplicando chunking semántico por cabeceras y tamaño...")
        # Split by Markdown Headers
        md_docs = self.md_splitter.split_text(sanitized_md)
        
        # Fallback split for huge sections
        final_docs = self.fallback_splitter.split_documents(md_docs)
        
        # Add metadata for traceability
        for i, doc in enumerate(final_docs):
            doc.metadata['chunk_index'] = i
            # Extract the first page number found in the chunk
            match = re.search(r'<!-- Page (\d+) -->', doc.page_content)
            if match:
                doc.metadata['page_number'] = match.group(1)
            else:
                doc.metadata['page_number'] = "Unknown"
                
            # Clean up the HTML comments so the LLM isn't distracted
            doc.page_content = re.sub(r' <!-- Page \d+ -->', '', doc.page_content)
            
        return final_docs
