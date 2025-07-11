import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

load_dotenv()

def ingest_docs():
    # Load PDFs
    pdf_loader = DirectoryLoader("data/pdfs", glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()

    # Load support articles
    text_loader = DirectoryLoader("data/web_content", glob="*.txt", loader_cls=TextLoader)
    text_docs = text_loader.load()

    all_docs = pdf_docs + text_docs

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    # Use open-source embedding
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Save to Chroma vector store
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="chroma_db")
    vectorstore.persist()

    print(f"Ingested {len(chunks)} chunks into vector store.")
