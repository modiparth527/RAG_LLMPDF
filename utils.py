import os
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using PyPDF2"""
    reader = PdfReader(pdf_file)
    return " ".join([page.extract_text() or "" for page in reader.pages])

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into manageable chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def call_mistral_api(prompt, api_key):
    """Call Mistral 7B via HuggingFace Inference API"""
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"inputs": prompt},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        elif response.status_code == 503:
            # Model loading - wait and retry
            time.sleep(30)
            return call_mistral_api(prompt, api_key)
        else:
            return f"⚠️ API Error: {response.text}"
            
    except Exception as e:
        return f"⚠️ Connection Error: {str(e)}"

def summarize_text(text, api_key, instruction="Summarize concisely:"):
    """Generate summary using Mistral"""
    prompt = f"{instruction}\n\n{text}"
    return call_mistral_api(prompt, api_key)