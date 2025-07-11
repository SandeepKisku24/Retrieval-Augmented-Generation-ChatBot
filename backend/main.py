from fastapi import FastAPI
from pydantic import BaseModel
from rag import get_rag_response

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    response = get_rag_response(query.question)
    return {"answer": response}
