import os
import pickle
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()
HF_TOKEN = os.getenv("HF_API_KEY")

# Do NOT load embedding model in Render backend — it will cause OOM
# Just load FAISS index (with precomputed embeddings)
with open("faiss_index.pkl", "rb") as f:
    vectorstore = pickle.load(f)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

def call_huggingface_model(prompt: str) -> str:
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
        "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
        headers=headers,
        json=payload
    )

    try:
        result = response.json()
    except Exception as e:
        return f"Failed to parse response JSON: {str(e)}"

    if isinstance(result, dict) and "error" in result:
        return "HuggingFace Error: " + result["error"]

    if isinstance(result, list) and "summary_text" in result[0]:
        return result[0]["summary_text"]

    return "Unexpected HuggingFace result format."

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
