import os
import google.generativeai as genai
from dotenv import load_dotenv
from functools import lru_cache
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-1.5-flash")

@lru_cache(maxsize=1)
def get_retriever():
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)
    return vectorstore.as_retriever(search_kwargs={"k": 2})  # Low memory footprint

def get_rag_response(question: str) -> str:
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    if not context.strip():
        return "I don't know"

    prompt = f"""Answer the question based only on the context below. If you don't know the answer, say "I don't know".

Context:
{context}

Question: {question}
"""
    response = model.generate_content(prompt)
    return response.text.strip()
