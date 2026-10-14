from fastapi import FastAPI, HTTPException            
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

app = FastAPI(title="Onboarding Chatbot API", description="API for the onboarding chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="onboarding_faq")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class QueryRequest(BaseModel):
    query: str
    
@app.post("/ask")
def ask_chatbot(req: QueryRequest):
    # Vectorize the question and query ChromaDB for the top relevant chunks.
    q_embedding = embedding_model.encode([req.query]).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=3,
    )
    retrieved_chunks = results.get("documents", [[]])[0]
    context = "\n\n".join(retrieved_chunks)

    # Strict anti-hallucination prompt.
    system_prompt = (
        "You are a strict onboarding assistant. Answer the user's question "
        "using ONLY the provided context below. If the answer cannot be found "
        "in the context, you must state: 'I'm sorry, but that information is not "
        "contained in the onboarding documentation.' Do not make anything up."
    )

    prompt = f"Context:\n{context}\n\nQuestion: {req.query}\n\nAnswer:"

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response["message"]["content"]
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error querying the model: {e}")

    return {
        "question": req.query,
        "answer": answer,
        "sources": retrieved_chunks,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)