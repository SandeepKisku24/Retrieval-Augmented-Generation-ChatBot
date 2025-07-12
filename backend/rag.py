import os
import pickle
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()
HF_TOKEN = os.getenv("HF_API_KEY")

def download_faiss_from_gdrive():
    file_id = "1D71dmUfIoG99BMVYlP9dEBgcQjl469lh"
    destination = "faiss_index.pkl"

    if not os.path.exists(destination):
        print("Downloading FAISS index from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Download complete.")
        else:
            print(f"Download failed. Status code: {response.status_code}")

download_faiss_from_gdrive()

def call_huggingface_model(prompt: str) -> str:
    if not HF_TOKEN:
        return "Hugging Face API key not set."

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

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6",
            headers=headers,
            json=payload,
            timeout=20
        )

        if response.status_code != 200:
            return f"HuggingFace Error {response.status_code}: {response.text}"

        result = response.json()
        print("HF Response:", result)

        if isinstance(result, list) and "summary_text" in result[0]:
            return result[0]["summary_text"]

        return "Unexpected HuggingFace result format."

    except requests.exceptions.Timeout:
        return "HuggingFace request timed out."
    except Exception as e:
        return f"Request failed: {str(e)}"

def get_rag_response(question: str) -> str:
    try:
        with open("faiss_index.pkl", "rb") as f:
            vectorstore = pickle.load(f)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    except Exception as e:
        return f"Failed to load FAISS index: {str(e)}"

    try:
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"

    if not context.strip():
        return "I don't know"

    prompt = f"""Answer the question based only on the context below. If you don't know the answer, say "I don't know".

Context:
{context}

Question: {question}
"""
    return call_huggingface_model(prompt)
