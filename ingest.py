import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Load the PDF document
pdf_path = "Onboarding_FAQ.pdf"
reader = PdfReader(pdf_path)

print(f"Successfully loaded {pdf_path} with {len(reader.pages)} pages.")

# 2. Extract text page by page
raw_text = ""
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += f"\n--- Page {i+1} ---\n" + text
# 3. Better chunking strategy: Split text into smaller chunks of ~500 characters
chunk_size = 500
chunk_overlap = 50

chunks = []
for i in range(0, len(raw_text), chunk_size - chunk_overlap):
    chunk = raw_text[i:i + chunk_size].strip()
    if len(chunk) > 30:
        chunks.append(chunk)

print(f"Split document into {len(chunks)} smart chunks.")

# 4. Initialize ChromaDB (local persistent storage)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="onboarding_faq")

# 5. Initialize the lightweight embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 6. Generate embeddings and store chunks in ChromaDB
ids = [str(i) for i in range(len(chunks))]
embeddings = embedding_model.encode(chunks).tolist()    

collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)

print("Document chunks successfully embedded and stored in ChromaDB!")

