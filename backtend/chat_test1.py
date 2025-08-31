"""
Dolphin-3.0-Mistral-24B conversational RAG medical chatbot
with long-term factual memory.

Run:  python chat_test1.py <user_id>
"""

from __future__ import annotations
import os, sys, asyncio, re, logging, warnings, importlib.util
import json, pathlib, datetime
from typing import List, ClassVar
from dotenv import load_dotenv

load_dotenv()

logging.getLogger("openai").setLevel(logging.ERROR)
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass

# ── dependency guard ───────────────────────────────────────────────────────
REQ = (
    "firebase_admin", "pinecone", "langchain", "openai",
    "langchain_huggingface", "torch", "sentence_transformers"
)
if miss := [m for m in REQ if importlib.util.find_spec(m) is None]:
    sys.exit("❌ Missing libs: " + ", ".join(miss))

# ── imports ────────────────────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document, BaseRetriever
from pydantic import BaseModel
import pinecone

import db
from model_loader2 import generate

# ── settings ─────────────────────────────────────────────────────────────
MEMORY_TURNS  = 6
TOP_PASSAGES  = 8
NOTE_LIMIT    = 12
STORAGE_DIR   = pathlib.Path("session_memory")
STORAGE_DIR.mkdir(exist_ok=True)

# ── helpers: local persistence ─────────────────────────────────────────────
def _load_json(path: pathlib.Path, default):
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return default

def _save_json(path: pathlib.Path, data) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)

def _load_chat(uid: str) -> list[tuple[str, str]]:
    return _load_json(STORAGE_DIR / f"chat_{uid}.json", [])

def _save_chat(uid: str, chat: list[tuple[str, str]]) -> None:
    _save_json(STORAGE_DIR / f"chat_{uid}.json", chat)

def _load_notes(uid: str) -> list[str]:
    return _load_json(STORAGE_DIR / f"notes_{uid}.json", [])

def _save_notes(uid: str, notes: list[str]) -> None:
    _save_json(STORAGE_DIR / f"notes_{uid}.json", notes)

def _load_questions(uid: str) -> list[str]:
    return _load_json(STORAGE_DIR / f"questions_{uid}.json", [])

def _save_questions(uid: str, qs: list[str]) -> None:
    _save_json(STORAGE_DIR / f"questions_{uid}.json", qs)

# ── Pinecone & embeddings ──────────────────────────────────────────────────
if "PINECONE_API_KEY" not in os.environ:
    sys.exit("❌  PINECONE_API_KEY env var not set")
pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
kb_store = PineconeVectorStore.from_existing_index(
    index_name="medicalbot",
    embedding=embeddings,
)

# ── medical-question guard ─────────────────────────────────────────────────
def is_medical_question(q: str) -> bool:
    """Return True for medical queries (or allowed commands), False otherwise."""
    q_lower = q.lower().strip()
    # allow simple commands
    if q_lower in ("help", "exit", "quit"):
        return True
    # basic medical keywords
    medical_keywords = [
        "pain","fever","cough","symptom","treatment","dose","prescribe",
        "diagnosis","headache","infection","blood","pressure","cold",
        "virus","bacteria","medicine","medication","allergy","rash",
        "nausea","vomit","diabetes","hypertension","heart","scan","lab",
        "surgery","mental","health","x-ray","mri","ultrasound"
    ]
    return any(kw in q_lower for kw in medical_keywords)

# ── NEW: richer patient-profile builder ────────────────────────────────────
def build_patient_profile(p: dict) -> str:
    """
    Craft the PATIENT PROFILE block used in the system prompt.
    Only includes a field if it exists, so the output is concise.
    """
    lines: list[str] = []
    name = p.get("name", "there")
    lines.append(f"Name: {name}")

    if p.get("age"):        lines.append(f"Age: {p['age']}")
    if p.get("gender"):     lines.append(f"Gender: {p['gender']}")
    if p.get("blood_type"):   lines.append(f"Blood type: {p['blood_type']}")

    # weight / height may come as scalar or dict
    wt = p.get("weight")
    if wt:
        if isinstance(wt, dict):
            lines.append(f"Weight: {wt.get('weight')} {wt.get('unit', '')}".strip())
        else:
            lines.append(f"Weight: {wt}")
    ht = p.get("height")
    if ht:
        if isinstance(ht, dict):
            lines.append(f"Height: {ht.get('height')} {ht.get('unit', '')}".strip())
        else:
            lines.append(f"Height: {ht}")

    chronic: list[str] = []
    if p.get("diabetes"):     chronic.append("diabetes")
    if p.get("hypertension"):  chronic.append("hypertension")
    if chronic:
        lines.append("Chronic conditions: " + ", ".join(chronic))

    conds = p.get("conditions", [])
    if conds:
        lines.append("Current symptoms: " + ", ".join(conds))

    return "\n".join(lines)

# ── Clean retriever with disclaimer stripping ──────────────────────────────
class CleanRetriever(BaseRetriever, BaseModel):
    """Return passages stripped of chat junk, timers, and generic disclaimers."""
    model_config = {"arbitrary_types_allowed": True}

    retriever: BaseRetriever

    banned_prefixes: ClassVar[tuple[str, ...]] = (
        "user:", "assistant:", "original", "rephrased"
    )
    timer_tokens: ClassVar[tuple[str, ...]] = (
        "time elapsed", "remaining turns", "seconds", "minutes"
    )
    disclaimer_tokens: ClassVar[tuple[str, ...]] = (
        "consult with a healthcare provider",
        "consult your healthcare provider",
        "consult your doctor",
        "see a doctor",
        "seek medical attention",
        "important to get evaluated",
        "not a substitute for",
        "just information",
    )

    def _strip(self, txt: str) -> str:
        keep: list[str] = []
        for ln in txt.splitlines():
            t = ln.strip()
            if not t or t.endswith("?"):
                continue
            l = t.lower()
            if l.startswith(self.banned_prefixes):
                continue
            if any(tok in l for tok in self.timer_tokens):
                continue
            if any(tok in l for tok in self.disclaimer_tokens):
                continue
            keep.append(t)
        return "\n".join(keep)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.retriever.invoke(query)
        return [
            Document(page_content=self._strip(d.page_content), metadata=d.metadata)
            for d in docs if self._strip(d.page_content)
        ]

# ── prompt & cleaner ───────────────────────────────────────────────────────
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
BULLET     = re.compile(r"^(?:[-•*]|\d+\.)\s*", re.UNICODE)
TOKEN_RE   = re.compile(r"\b\w+\b")
TAG_RE     = re.compile(r"<[^>]+>")

def build_prompt(profile: str, notes: str, excerpts: str,
                 history: str, question: str) -> str:
    return f"""
You are speaking as a board-certified physician with prescribing authority.

PATIENT PROFILE
---------------
{profile}

PATIENT NOTES  (long-term memory)
--------------
{notes or '[no lasting notes]'}

EXCERPTS  (cite when relevant)
--------
{excerpts or '[NO MATCHING PASSAGES]'}

RECENT DIALOGUE (last {MEMORY_TURNS} turns)
-------------------------------------------
{history}

GUIDELINES
----------
• Ask follow-up questions until you have all necessary clinical details:
  – Onset, duration, severity of symptoms  
  – Past medical history, allergies, current meds  
  – Any exam or lab findings  
• Then give a clear prescription plan:
  – Drug name, dosage, route, duration  
  – Key monitoring parameters and side effects  
• Be decisive and confident. Avoid generic disclaimers.
• Provide concise answers in 1–2 sentences only. No paragraphs.
• never answer to non medical qustions exept if the user inquires about his info 

Patient question: {question}

Physician:
""".strip()

def clean_answer(raw: str) -> str:
    raw = TAG_RE.sub("", raw)
    lines: list[str] = []
    for ln in raw.splitlines():
        t = ln.strip()
        if not t or t.lower().startswith(("patient statement", "assistant:", "user:", "turn")):
            continue
        if BULLET.match(t):
            continue
        lines.append(t)
    # take only first two sentences
    joined = " ".join(lines)
    sentences = SENT_SPLIT.split(joined)
    return " ".join(sentences[:2]).strip()

# ── derive one-line memory note ────────────────────────────────────────────
async def derive_note(question: str, answer: str) -> str:
    prompt = f"""
Summarise any new, durable fact learned about the patient from this exchange,
max 20 words. If nothing worth remembering, output "None".

Patient said: "{question}"
Physician replied: "{answer}"
""".strip()
    raw = await generate(prompt)
    return raw.strip().splitlines()[0]

# ── chat loop ──────────────────────────────────────────────────────────────
async def chat_loop(uid: str):
    profile_data = db.get_user_data(uid) or {}
    print("✅ profile:", profile_data or "[none]")

    profile_txt = build_patient_profile(profile_data)

    chat_memory   = _load_chat(uid)
    note_memory   = _load_notes(uid)
    question_log  = _load_questions(uid)

    name = profile_data.get("name", "there")
    print(f"Hello {name}! How can I help you today?\n")

    retriever = CleanRetriever(
        retriever=kb_store.as_retriever(search_kwargs={"k": 20})
    )

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 goodbye!")
            break
        if not question:
            continue

        # ── refuse non-medical queries ─────────────────────────────────────
        if not is_medical_question(question):
            print("Sorry, I can only answer medical questions. Please ask about medical conditions or treatments.\n")
            continue

        # ── log raw prompt ────────────────────────────────────────────────
        question_log.append(question)
        _save_questions(uid, question_log)

        # ── retrieve context passages ─────────────────────────────────────
        raw_docs = await asyncio.get_event_loop().run_in_executor(
            None, lambda: retriever._get_relevant_documents(question)
        )
        # simple keyword overlap ranking
        def _overlap_score(doc: Document) -> int:
            q_words = set(TOKEN_RE.findall(question.lower()))
            return sum(1 for w in TOKEN_RE.findall(doc.page_content.lower()) if w in q_words)

        top_docs = sorted(raw_docs, key=_overlap_score, reverse=True)[:TOP_PASSAGES]
        excerpts = "\n".join(d.page_content for d in top_docs)

        # ── build and send prompt ─────────────────────────────────────────
        hist = "\n".join(
            f"Patient: {u}\nPhysician: {a}"
            for u, a in chat_memory[-MEMORY_TURNS:]
        )
        notes_block = "\n".join(note_memory[-NOTE_LIMIT:])

        prompt = build_prompt(profile_txt, notes_block, excerpts, hist, question)
        raw    = await generate(prompt)
        answer = clean_answer(raw)

        print(answer, "\n")

        # ── update chat memory ────────────────────────────────────────────
        chat_memory.append((question, answer))
        _save_chat(uid, chat_memory)

        # ── derive & store factual note ──────────────────────────────────
        note = (await derive_note(question, answer)).strip()
        if note.lower() != "none" and note not in note_memory:
            timestamp = datetime.date.today().isoformat()
            note_memory.append(f"{timestamp}: {note}")
            _save_notes(uid, note_memory)

        # ── persist ONLY the patient's text to Pinecone ──────────────────
        PineconeVectorStore.from_existing_index(
            index_name="user-chat-memory",
            embedding=embeddings,
            namespace=uid,
        ).add_documents([
            Document(page_content=f"Patient: {question}"),
        ])

# ── entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python chat_test1.py <user_id>") 
        sys.exit(1) 
    asyncio.run(chat_loop(sys.argv[1]))