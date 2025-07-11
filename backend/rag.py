import os
import pickle
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()
HF_TOKEN = os.getenv("HF_API_KEY")

# Google Drive FAISS download
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

# Download FAISS index only if needed
download_faiss_from_gdrive()

# Global variable for retriever
retriever = None

def get_retriever():
    global retriever
    if retriever is None:
        try:
            with open("faiss_index.pkl", "rb") as f:
                vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        except Exception as e:
            print(f"Failed to load FAISS index: {str(e)}")
            retriever = None
    return retriever

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

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
            headers=headers,
            json=payload
        )
        result = response.json()
    except Exception as e:
        return f"Failed to call Hugging Face: {str(e)}"

    if isinstance(result, dict) and "error" in result:
        return "HuggingFace Error: " + result["error"]
    if isinstance(result, list) and "summary_text" in result[0]:
        return result[0]["summary_text"]

    return "Unexpected HuggingFace result format."

def get_rag_response(question: str) -> str:
    retriever_obj = get_retriever()
    if retriever_obj is None:
        return "Service is temporarily unavailable due to memory/load issues."

    docs = retriever_obj.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    if not context.strip():
        return "I don't know"

    prompt = f"""Answer the question based only on the context below. If you don't know the answer, say "I don't know".

Context:
{context}

Question: {question}
"""
    return call_huggingface_model(prompt)
