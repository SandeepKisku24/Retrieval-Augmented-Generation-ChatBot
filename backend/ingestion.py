import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def ingest_docs():
    # Load PDFs
    pdf_loader = DirectoryLoader("data/pdfs", glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()

    # Load text articles
    text_loader = DirectoryLoader("data/web_content", glob="*.txt", loader_cls=TextLoader)
    text_docs = text_loader.load()

    all_docs = pdf_docs + text_docs

    # Chunk the documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    print(f"✅ Total chunks: {len(chunks)}")

    # Embedding model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    # Save using CPU (avoids mps/cuda serialization issues)
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    print("✅ FAISS index saved to faiss_index.pkl")

if __name__ == "__main__":
    ingest_docs()
