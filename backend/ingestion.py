import os
import pickle
from dotenv import load_dotenv

# Force PyTorch to use CPU and avoid MPS serialization issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_DEVICE"] = ""

import torch
torch.set_default_device("cpu")

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def ingest_docs():
    # Load PDFs
    pdf_loader = DirectoryLoader("data/pdfs", glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()

    # Load text files
    text_loader = DirectoryLoader("data/web_content", glob="*.txt", loader_cls=TextLoader)
    text_docs = text_loader.load()

    all_docs = pdf_docs + text_docs

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    print(f"Total chunks: {len(chunks)}")

    # Create embeddings on CPU only
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # Force CPU
    )

    # Build FAISS index
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    # Save to disk using CPU-safe format
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    print("AISS index saved to faiss_index.pkl (CPU-safe)")

if __name__ == "__main__":
    ingest_docs()
