import os
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()
HF_TOKEN = os.getenv("HF_API_KEY")  # Put this in Render secrets panel

# Use a light-weight embedding model (free-tier safe)
embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

def call_huggingface_model(prompt: str):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 300
        }
    }

    response = requests.post(
        "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",  # this one works
        headers=headers,
        json=payload
    )

    print("STATUS CODE:", response.status_code)
    print("RESPONSE TEXT:", response.text)

    try:
        result = response.json()
    except Exception as e:
        return f"❌ Failed to parse response JSON: {str(e)}"

    if isinstance(result, dict) and "error" in result:
        return "❌ HuggingFace Error: " + result["error"]

    if isinstance(result, list) and "summary_text" in result[0]:
        return result[0]["summary_text"]

    return "❌ Unexpected format from HuggingFace"

def get_rag_response(question: str) -> str:
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    if not context.strip():
        return "🤔 I don't know"

    prompt = f"""Answer the question based only on the context below. If you don't know the answer, say "I don't know".

Context:
{context}

Question: {question}
"""
    return call_huggingface_model(prompt)
