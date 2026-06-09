import os
import re
import fitz  # PyMuPDF
import base64
import logging
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from src.core.ioc_masker import IoCMasker


class ReportIngestor:
    """
    Fase 1: Ingesta y Particionado Semántico.
    Objetivo: Transformar un PDF crudo en fragmentos de texto listos para el LLM.
    - Modo Normal (Docling): Usa extracción tradicional para sacar texto y tablas.
    - Modo VLM (Gemini): Convierte las páginas a imagen y le pide al LLM multimodal que transcriba
      el texto. Muy útil para no perder info vital oculta en pantallazos negros de terminales.
    Al final, ofusca los IoCs (IPs, hashes) para evitar sesgos y trocea el texto.
    """

    def __init__(
        self, chunk_size: int = 3500, chunk_overlap: int = 500, use_vlm: bool = False
    ) -> None:
        """
        Initializes the ReportIngestor with chunking parameters, IoC masker, and extraction mode.

        Args:
            chunk_size (int): The maximum character size for each chunk.
            chunk_overlap (int): Overlap in characters to maintain context.
            use_vlm (bool): If True, uses Gemini Flash VLM for extraction instead of traditional OCR.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_vlm = use_vlm
        self.ioc_masker = IoCMasker()

        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on, strip_headers=False
        )

        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        self.boilerplate_keywords = [
            "subscribe",
            "share on",
            "copyright ©",
            "copyright c",
            "all rights reserved",
            "manage cookies",
        ]

    def _load_pdf_vlm(self, file_path: str) -> List[Document]:
        """
        Loads a PDF file by converting each page to an image and using a Vision-Language Model
        (Gemini 1.5 Flash) to transcribe it. This solves the issue of traditional OCR failing
        on dark terminal screenshots and code blocks.
        """
        logger = logging.getLogger(__name__)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        logger.info(
            f"Loading PDF '{file_path}' using Multimodal VLM extraction (Gemini Flash)..."
        )
        # Por ahora forzamos Gemini siempre para VLM, ya que el cluster remoto vLLM no tiene modelo visual.
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.0, max_retries=3)

        doc = fitz.open(file_path)
        documents = []

        for page_num in range(len(doc)):
            logger.info(f"[VLM] Processing page {page_num + 1}/{len(doc)}...")
            page = doc.load_page(page_num)
            # Render page to image at 150 DPI (good balance of quality/size)
            pix = page.get_pixmap(dpi=150)
            img_data = pix.tobytes("png")
            img_base64 = base64.b64encode(img_data).decode("utf-8")

            prompt_text = (
                "You are a specialized Vision-Language Model for cybersecurity.\n"
                "Analyze this document page. Your ONLY task is to extract text that appears INSIDE images, diagrams, terminal windows (black backgrounds), or code blocks.\n"
                "Do NOT transcribe the normal paragraphs of the document. Ignore standard text.\n"
                "If you find terminal logs, commands, or code in an image, transcribe them exactly and wrap them in markdown code blocks.\n"
                "If the page has no images, diagrams, or terminal blocks, reply EXACTLY with 'NO_IMAGE_DATA'."
            )

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                ]
            )

            try:
                response = llm.invoke([message])

                # Extracción robusta del contenido
                if isinstance(response.content, str):
                    page_text = response.content
                elif isinstance(response.content, list):
                    text_parts = []
                    for item in response.content:
                        if isinstance(item, str):
                            text_parts.append(item)
                        elif isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                    page_text = "\n".join(text_parts)
                else:
                    page_text = str(response.content)

            except Exception as e:
                logger.error(
                    f"[VLM] Error processing page {page_num + 1}: {e}. Falling back to basic text extraction."
                )
                page_text = page.get_text()

            # Forzar cast a string por si acaso (Pydantic ValidationError protection)
            if not isinstance(page_text, str):
                page_text = str(page_text)

            # Simulate the Unstructured format so the rest of the pipeline works seamlessly
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={"page_number": page_num + 1, "category": "NarrativeText"},
                )
            )

        doc.close()
        return documents

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Loads a PDF file and extracts its elements using Docling.
        Uses Docling's advanced document layout analysis.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        logger = logging.getLogger(__name__)

        # Optional: fast page check using PyMuPDF to validate extraction later
        try:
            pdf_doc = fitz.open(file_path)
            num_pages = len(pdf_doc)
            pdf_doc.close()
        except Exception:
            num_pages = 1

        logger.info(
            f"Loading PDF '{file_path}' using Docling (Pages detected: {num_pages})..."
        )

        try:
            # Quitamos EasyOCR por completo porque es demasiado restrictivo.
            # Docling tiene integrado RapidOCR por defecto que procesa de forma universal
            # ruso, chino, árabe, japonés e inglés simultáneamente con un único modelo global sin crashear.
            pipeline_options = PdfPipelineOptions(do_ocr=True)

            converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                },
            )
        except ImportError:
            logger.warning(
                "[WARNING] EasyOCR options failed to import. Falling back to default Docling converter."
            )
            converter = DocumentConverter()

        result = converter.convert(file_path)
        doc = result.document

        elements = []
        for item, level in doc.iterate_items():
            text = ""
            cat = "NarrativeText"
            html = None

            # Extract the string representation of the enum label
            label = getattr(item, "label", "text")
            if hasattr(label, "name"):
                label_str = label.name.lower()
            elif isinstance(label, str):
                label_str = label.lower()
            else:
                label_str = "text"

            # Map Docling labels to our expected categories
            if label_str in ["title", "section_header", "page_header"]:
                cat = "Title"
                if label_str == "page_header":
                    cat = "Header"
            elif label_str in ["page_footer"]:
                cat = "Footer"
            elif label_str == "list_item":
                cat = "ListItem"
            elif label_str == "table":
                cat = "Table"
                if hasattr(item, "export_to_html"):
                    html = item.export_to_html()

            if hasattr(item, "text"):
                text = item.text

            # Ignore empty elements or images without text/html
            if not text and not html:
                continue

            # Try to fetch provenance for page number
            page_num = "?"
            if hasattr(item, "prov") and item.prov:
                page_num = str(item.prov[0].page_no)

            metadata = {"category": cat, "page_number": page_num}
            if html:
                metadata["text_as_html"] = html
                text = html  # Use HTML representation for the chunk content

            elements.append(Document(page_content=text, metadata=metadata))

        # Validation Logic: Suspiciously short extraction
        total_text_len = sum(len(el.page_content) for el in elements)
        if num_pages >= 3 and total_text_len < (num_pages * 50):
            logger.warning(
                f"[WARNING] Extracted text is suspiciously short ({total_text_len} chars across {num_pages} pages). "
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
        logger = logging.getLogger(__name__)

        # 1. SIEMPRE usamos Docling para extraer el texto base perfecto
        logger.info("[INFO] Extracting base text using Docling (OCR)...")
        raw_elements = self.load_pdf(file_path)

        # VLM integration moved to the end as independent chunks.

        logger.info(
            "[INFO] Phase 1: Cleaning noise and reconstructing Markdown structure..."
        )
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

        logger.info(
            "[INFO] Phase 1: Masking volatile Indicators of Compromise (IoCs)..."
        )
        sanitized_md = self.ioc_masker.mask_text(full_md)

        logger.info("[INFO] Phase 1: Applying semantic chunking by headers and size...")
        md_docs = self.md_splitter.split_text(sanitized_md)

        # Fallback split for huge sections
        final_docs = self.fallback_splitter.split_documents(md_docs)

        # =====================================================================
        # 2. ENRIQUECIMIENTO VLM: CHUNKS INDEPENDIENTES (NUEVO)
        # =====================================================================
        if self.use_vlm:
            logger.info("[INFO] VLM is active. Appending independent visual chunks...")
            vlm_elements = self._load_pdf_vlm(file_path)
            for el in vlm_elements:
                # Solo si encontró algo y no es ruido de 2 letras
                if (
                    "NO_IMAGE_DATA" not in el.page_content
                    and len(el.page_content.strip()) > 10
                ):
                    vlm_text = f"### [Visual Evidence extracted from Image/Terminal] ###\n{el.page_content}"
                    # Sanitizamos los IoCs que el VLM haya podido leer de la imagen
                    vlm_text = self.ioc_masker.mask_text(vlm_text)

                    final_docs.append(
                        Document(
                            page_content=vlm_text,
                            metadata={
                                "category": "VLM_Extraction",
                                "page_number": el.metadata.get("page_number", "?"),
                            },
                        )
                    )
        # =====================================================================

        # Add metadata for traceability
        for i, doc in enumerate(final_docs):
            doc.metadata["chunk_index"] = i
            # Extract the first page number found in the chunk
            match = re.search(r"<!-- Page (\d+) -->", doc.page_content)
            if match:
                doc.metadata["page_number"] = match.group(1)
            else:
                # Fallback para los chunks del VLM que ya traen el page_number
                doc.metadata["page_number"] = doc.metadata.get("page_number", "Unknown")

            # Clean up the HTML comments so the LLM isn't distracted
            doc.page_content = re.sub(r" <!-- Page \d+ -->", "", doc.page_content)

        return final_docs
