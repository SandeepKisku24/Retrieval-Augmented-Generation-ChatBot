# ingestion.py

import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def ingest_docs():
    # Load PDFs
    pdf_loader = DirectoryLoader("data/pdfs", glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()

    # Load text articles
    text_loader = DirectoryLoader("data/web_content", glob="*.txt", loader_cls=TextLoader)
    text_docs = text_loader.load()

    all_docs = pdf_docs + text_docs

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    # Embedding
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embedding)

    # Save to disk
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    print(f"✅ Ingested {len(chunks)} chunks into FAISS and saved as faiss_index.pkl.")

if __name__ == "__main__":
    ingest_docs()
