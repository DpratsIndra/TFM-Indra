import os
from typing import List

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.ioc_masker import IoCMasker


class ReportIngestor:
    """
    Handles Phase 1 of the LangChain pipeline: Ingestion.
    Responsible for loading PDF reports, parsing their structure, masking IoCs, 
    and splitting the text into semantic chunks for vector embedding.
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300) -> None:
        """
        Initializes the ReportIngestor with chunking parameters and IoC masker.
        
        Args:
            chunk_size (int): The maximum character size for each chunk.
            chunk_overlap (int): Overlap in characters to maintain context.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.ioc_masker = IoCMasker()
        
        # Usamos RecursiveCharacterTextSplitter para garantizar un tamaño de chunk predecible.
        # SemanticChunker a veces agrupa demasiadas páginas en un solo chunk gigante,
        # lo que provoca que el Reranker (limitado a 512 tokens) trunque el texto y pierda contexto.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Loads a PDF file and extracts its content into LangChain Document objects.
        Uses UnstructuredPDFLoader to preserve paragraph structure.
        
        Args:
            file_path (str): Absolute or relative path to the PDF file.
            
        Returns:
            List[Document]: A list of unchunked Document objects with metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
            
        # Using mode='single' groups text together but retains Unstructured's smart parsing
        # Alternatively, mode='elements' returns a document per paragraph. 
        # Since we use SemanticChunker later, 'single' is preferred to provide full context.
        loader = UnstructuredPDFLoader(file_path, mode="single")
        documents = loader.load()
        return documents

    def process_report(self, file_path: str) -> List[Document]:
        """
        Main method to process a CTI report:
        1. Loads the PDF.
        2. Masks IoCs in the text.
        3. Splits the text into chunks.
        
        Args:
            file_path (str): The path to the PDF CTI report.
            
        Returns:
            List[Document]: A list of sanitized and chunked Document objects.
        """
        # 1. Load the raw documents
        raw_documents = self.load_pdf(file_path)
        
        # 2. Sanitize documents (Mask IoCs)
        sanitized_documents = []
        for doc in raw_documents:
            masked_text = self.ioc_masker.mask_text(doc.page_content)
            # Create a new Document object to preserve original metadata
            sanitized_doc = Document(
                page_content=masked_text,
                metadata=doc.metadata.copy()
            )
            sanitized_documents.append(sanitized_doc)
            
        # 3. Chunk the sanitized documents
        chunked_documents = self.text_splitter.split_documents(sanitized_documents)
        
        # Add a chunk index to the metadata for traceability
        for i, chunk in enumerate(chunked_documents):
            chunk.metadata['chunk_index'] = i
            
        return chunked_documents
