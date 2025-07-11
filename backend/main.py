from fastapi import FastAPI
from pydantic import BaseModel
from rag import get_rag_response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS for any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    answer = get_rag_response(query.question)
    return {"answer": answer}

# Optional: health check endpoint
@app.get("/")
def health_check():
    return {"status": "ok"}
