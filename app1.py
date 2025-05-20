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

app = FastAPI()

# ---- Storage ----
vectorstores_dir = "vectorstores"
history_store = {}
os.makedirs(vectorstores_dir, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
DEEPINFRA_API_KEY = "tSpRKL5Nm8c9pt48Z6fcqWXqDZte2xjk"
DEEPINFRA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# ---- Helper Functions ----

def fetch_url_content(url, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.get_text(separator="\n")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise HTTPException(status_code=500, detail=f"Failed to fetch URL: {url}. Error: {e}")

def split_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
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

    history_store[session_id] = {"chat": [], "current_model": None, "pending_ambiguity": None}

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
    for turn in history[-3:]:
        history_prompt += f"Q: {turn['q']}\nA: {turn['a']}\n"

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

    response = requests.post("https://api.deepinfra.com/v1/openai/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"DeepInfra Error: {response.status_code} - {response.text}")
    return response.json()["choices"][0]["message"]["content"].strip()

def detect_ambiguity(question, context, session_id):
    matches = re.findall(r'RS485-[A-Za-z0-9]+', context, flags=re.IGNORECASE)
    unique = set([m.upper() for m in matches])
    if len(unique) > 1 and not any(m.lower() in question.lower() for m in unique):
        return True, list(unique), unique
    return False, [], unique

def update_model_from_clarification(user_input, known_models):
    user_input = user_input.strip().lower()
    known_models_lower = {model.lower(): model for model in known_models}

    # If full model mentioned
    if user_input in known_models_lower:
        return known_models_lower[user_input]

    # If short form e.g. CB -> RS485-CB
    for full_model in known_models:
        suffix = full_model.split("-")[-1].lower()
        if suffix == user_input:
            return full_model
    return None

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

    if not index:
        return {"error": "Invalid session_id. Please upload URLs first."}

    state = history_store.get(session_id, {"chat": [], "current_model": None, "pending_ambiguity": None})
    chat_history = state["chat"]
    current_model = state["current_model"]
    pending_ambiguity = state.get("pending_ambiguity", None)

    question = query.question.strip()
    query_embedding = model.encode([question])
    distances, indices = index.search(np.array(query_embedding), k=3)
    relevant = [chunks[i] for i in indices[0] if i < len(chunks)]
    context = "\n\n".join(relevant)

    # If user is clarifying a model
    if pending_ambiguity:
        clarification = update_model_from_clarification(question, pending_ambiguity["models"])
        if clarification:
            current_model = clarification
            reconstructed_question = f"{pending_ambiguity['original_question']} (regarding {clarification})"
            answer = call_deepinfra(context, reconstructed_question, chat_history, clarification)

            chat_history.append({"q": reconstructed_question, "a": answer})
            history_store[session_id] = {
                "chat": chat_history,
                "current_model": clarification,
                "pending_ambiguity": None
            }
            return {"response": answer}

    # Detect ambiguity
    ambiguous, models_found, all_known_models = detect_ambiguity(question, context, session_id)
    if ambiguous:
        history_store[session_id]["pending_ambiguity"] = {
            "original_question": question,
            "models": list(all_known_models)
        }
        return {
            "response": f"Your question may refer to multiple models such as {', '.join(sorted(all_known_models))}. Could you please specify the exact model you're referring to?"
        }

    # Try to resolve model name from full/short form
    resolved_model = update_model_from_clarification(question, all_known_models)
    if resolved_model:
        current_model = resolved_model

    # Final response
    answer = call_deepinfra(context, question, chat_history, current_model)
    chat_history.append({"q": question, "a": answer})

    history_store[session_id] = {
        "chat": chat_history,
        "current_model": current_model,
        "pending_ambiguity": None
    }

    return {"response": answer}

# ---- Run App ----

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
