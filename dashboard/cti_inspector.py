# To run: conda activate tfm_env 
# streamlit run dashboard/cti_inspector.py 
import sys
import os
import tempfile
import streamlit as st

# Add the project root to the python path so imports from src work smoothly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.ioc_masker import IoCMasker
from src.langchain_pipeline.phase1_ingestion import ReportIngestor

# Setup Page Configuration
st.set_page_config(page_title="CTI Preprocessing Inspector", layout="wide")
st.title("CTI Preprocessing Inspector")
st.markdown("Upload a CTI report (PDF) to visualize the behavior of **Phase 1: Ingestion, IoC Sanitization, and Semantic Chunking**.")

uploaded_file = st.file_uploader("Upload a CTI report (PDF)", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    st.success(f"File uploaded successfully: {uploaded_file.name}")

    with st.spinner("Processing and splitting the report (Phase 1)..."):
        ingestor = ReportIngestor(chunk_size=3500, chunk_overlap=500)
        masker = IoCMasker()
        
        try:
            raw_docs = ingestor.load_pdf(tmp_file_path)
            raw_text = "\n\n".join([doc.page_content for doc in raw_docs])
            
            chunked_docs = ingestor.process_report(tmp_file_path)
            
            st.header("1. Sanitization Visualization (IoC Masking)")
            st.markdown("Compare the original text extracted from the PDF with the text processed by `IoCMasker`.")
            
            col1, col2 = st.columns(2)
            
            # Tomamos una muestra generosa del inicio del documento (ej. primeros 3000 caracteres)
            sample_raw_text = raw_text[:3000] + ("..." if len(raw_text) > 3000 else "")
            sample_masked_text = masker.mask_text(sample_raw_text)
            
            # Función para resaltar los tags en el markdown de Streamlit
            def highlight_tags(text: str) -> str:
                tags = ["<IoC_URL>", "<IoC_EMAIL>", "<IoC_IPv6>", "<IoC_IPv4>", "<IoC_HASH>", "<IoC_DOMAIN>"]
                highlighted = text
                for tag in tags:
                    # Streamlit Markdown soporta colores mediante HTML inline
                    highlighted = highlighted.replace(tag, f"**<span style='color:#FF4B4B;'>{tag}</span>**")
                return highlighted

            with col1:
                st.subheader("Original Text Extract")
                st.text_area("Raw Text", value=sample_raw_text, height=400, disabled=True, label_visibility="collapsed")
                
            with col2:
                st.subheader("Sanitized Text")
                st.markdown(
                    f"<div style='height: 400px; overflow-y: scroll; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f9f9f9; color: black; font-family: monospace; white-space: pre-wrap;'>"
                    f"{highlight_tags(sample_masked_text)}"
                    f"</div>", 
                    unsafe_allow_html=True
                )
                
            st.markdown("---")
            
            st.header("2. Semantic Division (Chunking)")
            st.markdown("Review how the `RecursiveCharacterTextSplitter` has divided the document.")
            
            with st.expander("View Generated Chunks"):
                total_chunks = len(chunked_docs)
                st.write(f"**Total Chunks:** {total_chunks}")
                
                for i, doc in enumerate(chunked_docs):
                    chunk_length = len(doc.page_content)
                    st.markdown(f"#### Chunk {i + 1}/{total_chunks} (Length: {chunk_length} characters)")
                    st.info(doc.page_content)
                    
        except Exception as e:
            st.error(f"Error processing the file: {e}")
            
        finally:
            # Limpiar archivo temporal
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
