import streamlit as st
from utils import extract_text_from_pdf, chunk_text, summarize_text
import time

# Configure app
st.set_page_config(page_title="Mistral PDF Summarizer", layout="wide")
st.title("📄 PDF Summarizer with Mistral 7B")

# Sidebar controls
with st.sidebar:
    st.markdown("## Settings")
    chunk_size = st.slider("Chunk Size (characters)", 500, 3000, 1000)
    st.markdown("""
    **Note:**  
    - Uses Mistral 7B via HuggingFace  
    - Free tier has ~500 requests/day  
    [Get API Key](https://huggingface.co/settings/tokens)
    """)

# Main interface
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Get API key from secrets
    if "HF_API_KEY" not in st.secrets:
        st.error("Add your HuggingFace API key in `.streamlit/secrets.toml`")
        st.stop()
    
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(uploaded_file)
        if not text.strip():
            st.error("No text found in PDF")
            st.stop()
    
    with st.spinner("Processing..."):
        chunks = chunk_text(text, chunk_size=chunk_size)
        
        summaries = []
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            summary = summarize_text(chunk, st.secrets["HF_API_KEY"])
            summaries.append(summary)
            progress_bar.progress((i + 1) / len(chunks))
            time.sleep(1)  # Avoid rate limiting
        
        final_summary = "\n\n".join(summaries)
        
    st.subheader("Summary")
    st.write(final_summary)
    
    # Show raw text (collapsible)
    with st.expander("View extracted text"):
        st.text(text[:2000] + "...")