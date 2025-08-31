"""
HTTP wrapper around the chatbot logic.
Run locally with:
    uvicorn chat_server:app --reload
"""

from __future__ import annotations
import os
import asyncio
import datetime
import re
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

load_dotenv()

# ── project imports ─────────────────────────────────────────────────────────
import chat_test1 as bot

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="Medical Chatbot API")

# ── CORS middleware for web/mobile access ──────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Request and Response Models ---
class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    diabetes: Optional[bool] = False
    hypertension: Optional[bool] = False
    conditions: Optional[List[str]] = []

class SignupResponse(BaseModel):
    message: str

class ChatRequest(BaseModel):
    uid: str
    message: str

class ChatResponse(BaseModel):
    reply: str

# In-memory cache for user conversations
_conversations: dict[str, list[tuple[str, str]]] = {}

# --- Signup Endpoint ---
@app.post("/signup", response_model=SignupResponse)
async def signup(req: SignupRequest):
    try:
        # 1. Create the user in Supabase Auth
        auth_response = bot.db.supabase.auth.sign_up({
            "email": req.email,
            "password": req.password,
        })
        
        new_user = auth_response.user
        if not new_user:
            raise HTTPException(status_code=400, detail="Could not create user in Supabase Auth.")

        # 2. Create the corresponding profile in the 'patients' table with all data
        profile_data = {
            "id": new_user.id,
            "name": req.name,
            "age": req.age,
            "gender": req.gender,
            "weight": req.weight,
            "height": req.height,
            "diabetes": req.diabetes,
            "hypertension": req.hypertension,
            "conditions": req.conditions,
        }
        
        profile_response = bot.db.supabase.table("patients").insert(profile_data).execute()

        if profile_response.data:
            return SignupResponse(message="Signup successful! Please log in.")
        else:
            raise HTTPException(status_code=500, detail="Could not create user profile in database.")

    except Exception as e:
        error_message = str(e)
        if "User already registered" in error_message:
            raise HTTPException(status_code=400, detail="A user with this email already exists.")
        if "Email signups are disabled" in error_message:
             raise HTTPException(status_code=400, detail="Email signups are currently disabled in the Supabase project.")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {error_message}")


def rank_passages(docs, query: str, top_k: int = bot.TOP_PASSAGES):
    # This function remains the same
    q_words = set(bot.TOKEN_RE.findall(query.lower()))
    def score(doc):
        return sum(1 for w in bot.TOKEN_RE.findall(doc.page_content.lower()) if w in q_words)
    return sorted(docs, key=score, reverse=True)[:top_k]

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # This function remains the same
    uid = req.uid.strip()
    if not uid:
        raise HTTPException(status_code=400, detail="uid is required")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is empty")

    memory       = _conversations.setdefault(uid, bot._load_chat(uid))
    question_log = bot._load_questions(uid)
    note_memory  = bot._load_notes(uid)

    retriever = bot.CleanRetriever(
        retriever=bot.kb_store.as_retriever(search_kwargs={"k": 20})
    )
    
    raw_docs = await asyncio.get_event_loop().run_in_executor(
        None, lambda: retriever._get_relevant_documents(req.message)
    )
    top_docs = rank_passages(raw_docs, req.message)
    excerpts = "\n".join(d.page_content for d in top_docs)

    # --- THIS IS THE LINE THAT WAS FIXED ---
    profile = bot.db.get_user_data(uid) or {}
    
    profile_txt = bot.build_patient_profile(profile)
    notes_block = "\n".join(note_memory[-bot.NOTE_LIMIT:])
    hist = "\n".join(
        f"Patient: {u}\nAssistant: {a}"
        for u, a in memory[-bot.MEMORY_TURNS:]
    )

    prompt = bot.build_prompt(profile_txt, notes_block, excerpts, hist, req.message)
    raw    = await bot.generate(prompt)
    answer = bot.clean_answer(raw)

    memory.append((req.message, answer))
    bot._save_chat(uid, memory)

    question_log.append(req.message)
    bot._save_questions(uid, question_log)

    note = (await bot.derive_note(req.message, answer)).strip()
    if note.lower() != "none" and note not in note_memory:
        timestamp = datetime.date.today().isoformat()
        note_memory.append(f"{timestamp}: {note}")
        bot._save_notes(uid, note_memory)

    bot.PineconeVectorStore.from_existing_index(
        index_name="user-chat-memory",
        embedding=bot.embeddings,
        namespace=uid,
    ).add_documents([
            bot.Document(page_content=f"Patient: {req.message}")
    ])

    return ChatResponse(reply=answer)