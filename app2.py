from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
import re
import time
from numpy.linalg import norm
import spacy

app = FastAPI()

# ---- Storage ----
vectorstores_dir = "vectorstores"
history_store = {}
os.makedirs(vectorstores_dir, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
DEEPINFRA_API_KEY = "tSpRKL5Nm8c9pt48Z6fcqWXqDZte2xjk"  # <-- Directly set here
DEEPINFRA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load SpaCy NER model
nlp = spacy.load("en_core_web_lg")

# ---- Helper Functions ----

def fetch_url_content(url, retries=3, delay=5):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/113.0.0.0 Safari/537.36"
        )
    }
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=60, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.get_text(separator="\n")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise HTTPException(status_code=500, detail=f"Failed to fetch URL: {url}. Error: {e}")

def split_text(text, chunk_size=500):
    """
    Paragraph-based chunking. Each chunk is a group of paragraphs up to chunk_size words.
    """
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current = []
    count = 0
    for para in paras:
        words = para.split()
        count += len(words)
        current.append(para)
        if count >= chunk_size:
            chunks.append('\n\n'.join(current))
            current = []
            count = 0
    if current:
        chunks.append('\n\n'.join(current))
    return chunks

def vectorize_and_store(session_id, texts):
    chunks = split_text(texts)
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    index_path = os.path.join(vectorstores_dir, f"{session_id}.index")
    metadata_path = os.path.join(vectorstores_dir, f"{session_id}.pkl")

    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)

    history_store[session_id] = {
        "chat": [],
        "current_model": None,
        "pending_ambiguity": None
    }

def load_vectorstore(session_id):
    index_path = os.path.join(vectorstores_dir, f"{session_id}.index")
    metadata_path = os.path.join(vectorstores_dir, f"{session_id}.pkl")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None, None

    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def call_deepinfra(context, question, history, model_context=None):
    history_prompt = ""
    for interaction in history[-3:]:
        history_prompt += f"Q: {interaction['q']}\nA: {interaction['a']}\n"

    if model_context:
        question += f" (regarding {model_context})"

    prompt = f"""
Use ONLY the following content to answer the question accurately and concisely. If ambiguous, ask for clarification.

Context:
{context}

{history_prompt}
Question: {question}
""".strip()

    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPINFRA_MODEL,
        "messages": [
            {"role": "system", "content": "You're a precise assistant. Always give short, specific answers using only the given context. Ask for clarification if the question is ambiguous."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post("https://api.deepinfra.com/v1/openai/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"DeepInfra API error: {e}")
        return "Sorry, I couldn't process your request due to an internal error. Please try again later."

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2) + 1e-10)

def extract_entities(text):
    doc = nlp(text)
    return set(ent.text.lower() for ent in doc.ents if ent.label_ in {"ORG", "PRODUCT", "GPE", "PERSON", "EVENT", "WORK_OF_ART", "LOC"})

def detect_ambiguity(question, chunks):
    # Use SpaCy to extract entities from the question
    question_entities = extract_entities(question)
    if not question_entities:
        return True, [], {"type": "no_entity"}

    # Extract all entities from the context chunks
    chunk_entities = set()
    for chunk in chunks:
        chunk_entities.update(extract_entities(chunk))

    # If none of the question entities are in the context, ask for clarification
    if not question_entities.intersection(chunk_entities):
        return True, [], {"type": "entity_mismatch", "entities": question_entities}

    # If multiple entities in question, ask for clarification
    if len(question_entities) > 1:
        return True, list(question_entities), {"type": "multi_entity", "entities": question_entities}

    return False, [], {}

def find_relevant_chunks(question, chunks, index, top_k=8):
    question_entities = extract_entities(question)
    if not question_entities:
        # fallback to embedding search
        query_embedding = model.encode([question])
        distances, indices = index.search(np.array(query_embedding), k=top_k)
        return [chunks[i] for i in indices[0] if i < len(chunks)]

    # Prioritize chunks containing the entities
    relevant = []
    for i, chunk in enumerate(chunks):
        chunk_entities = extract_entities(chunk)
        if question_entities.intersection(chunk_entities):
            relevant.append(chunk)
            if len(relevant) >= top_k:
                break
    # If not enough, fill with embedding search
    if len(relevant) < top_k:
        query_embedding = model.encode([question])
        distances, indices = index.search(np.array(query_embedding), k=top_k)
        for i in indices[0]:
            if i < len(chunks) and chunks[i] not in relevant:
                relevant.append(chunks[i])
            if len(relevant) >= top_k:
                break
    return relevant[:top_k]

# ---- API Schemas ----

class URLPayload(BaseModel):
    urls: list[str]

class Question(BaseModel):
    question: str
    session_id: str

# ---- API Endpoints ----

@app.post("/upload_urls")
def upload_urls(payload: URLPayload):
    session_id = str(uuid.uuid4())
    combined_text = ""

    for url in payload.urls:
        content = fetch_url_content(url)
        combined_text += content + "\n\n"

    vectorize_and_store(session_id, combined_text)
    return {"session_id": session_id, "message": "URLs processed and vectorized successfully."}

@app.post("/chat")
def chat(query: Question):
    session_id = query.session_id
    index, chunks = load_vectorstore(session_id)

    if not index or not chunks:
        return {"error": "No data found for this session. Please upload URLs first."}

    state = history_store.get(session_id, {"chat": [], "current_model": None, "pending_ambiguity": None})
    chat_history = state["chat"]
    pending_ambiguity = state.get("pending_ambiguity", None)

    question = query.question.strip()

    # Use full text for ambiguity detection
    ambiguous, entities, ambiguity_info = detect_ambiguity(question, chunks)

    if ambiguous:
        if ambiguity_info.get("type") == "no_entity":
            clarification = "Your question seems general. Could you specify a product, organization, location, or person?"
        elif ambiguity_info.get("type") == "entity_mismatch":
            clarification = f"The context doesn't mention the entities in your question ({', '.join(ambiguity_info.get('entities', []))}). Could you clarify or rephrase?"
        elif ambiguity_info.get("type") == "multi_entity":
            clarification = f"Your question mentions multiple entities ({', '.join(ambiguity_info.get('entities', []))}). Please specify which one you are referring to."
        else:
            clarification = "Could you clarify your question?"
        history_store[session_id]["pending_ambiguity"] = {
            "original_question": question,
            "entities": entities
        }
        return {"response": clarification}

    # Find relevant chunks using NER and embeddings
    relevant_chunks = find_relevant_chunks(question, chunks, index, top_k=8)
    context = "\n\n".join(relevant_chunks)

    use_history = False
    if chat_history:
        last_question = chat_history[-1]["q"]
        last_embedding = model.encode([last_question])
        query_embedding = model.encode([question])
        sim = cosine_similarity(query_embedding[0], last_embedding[0])
        if sim > 0.65:
            use_history = True

    answer = call_deepinfra(context, question, chat_history if use_history else [])
    chat_history.append({"q": question, "a": answer})

    history_store[session_id] = {
        "chat": chat_history,
        "current_model": None,
        "pending_ambiguity": None
    }

    return {"response": answer}

# ---- Run FastAPI App ----

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
