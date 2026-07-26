"""
pipeline.py — Backend logic untuk PT Retailindo Sejahtera Multi-Agent Assistant

Modul ini adalah hasil refactor dari 00_master_notebook.ipynb (Part 1-5),
supaya bisa di-import langsung oleh app.py (Streamlit dashboard) tanpa
perlu menjalankan ulang notebook.

Cara pakai:
    from pipeline import get_orchestrator, evaluate_orchestrator_run

Environment variables yang dibutuhkan:
    GROQ_API_KEY   -> https://console.groq.com/keys (gratis)

Struktur folder yang diharapkan (hasil dari notebook Part 1 & 2):
    ./chroma_db/                          -> vector DB hasil Part 2
    ./clean_customer_service.csv          -> opsional, dipakai jika chroma_db belum ada
    ./clean_billing_returns.csv
    ./clean_sales_presales.csv
    ./clean_inventory.csv
    ./clean_marketing.csv
"""

import os
import time
import json
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

CHROMA_PATH = "./chroma_db"
FINETUNED_MODEL_PATH = "./lora_finetuned_model"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
COLLECTION_NAMES = {
    "customer_service": "customer_service_kb",
    "billing_returns": "billing_returns_kb",
    "sales_presales": "sales_presales_kb",
    "inventory": "inventory_kb",
    "marketing": "marketing_kb",
}


# ---------------------------------------------------------------------------
# 1. Setup embedder, LLM, dan koneksi ChromaDB (dipanggil sekali, di-cache
#    oleh app.py lewat st.cache_resource)
# ---------------------------------------------------------------------------

def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def load_llm():
    if "GROQ_API_KEY" not in os.environ:
        raise RuntimeError(
            "GROQ_API_KEY belum di-set. Set env var ini sebelum menjalankan app.py, "
            "contoh: export GROQ_API_KEY=xxxx"
        )
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)


def load_finetuned_pipeline():
    """Load model hasil fine-tuning LoRA (Part 4/4.5 notebook) sebagai HF text-generation
    pipeline. Dipakai khusus oleh agent Customer Service & Billing/Retur, karena data
    training fine-tuning berasal dari 2 divisi ini.

    Mengembalikan (None, None) kalau folder model belum ada, supaya app.py bisa fallback
    ke Groq untuk semua agent tanpa crash (misal saat development sebelum fine-tuning selesai).
    """
    if not os.path.isdir(FINETUNED_MODEL_PATH):
        return None, None

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, device_map="auto", torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
    model = model.merge_and_unload()

    text_pipeline = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.3,
    )
    return text_pipeline, tokenizer


def load_collections():
    """Ambil 5 collection dari ChromaDB yang sudah di-build di notebook Part 2.

    Jika folder chroma_db/ belum ada sama sekali, lempar error yang jelas
    supaya user tahu harus menjalankan notebook Part 1-2 dulu.
    """
    if not os.path.isdir(CHROMA_PATH):
        raise RuntimeError(
            f"Folder '{CHROMA_PATH}' tidak ditemukan. Jalankan Part 1 & 2 di "
            "00_master_notebook.ipynb dulu, lalu copy folder chroma_db/ ke "
            "direktori yang sama dengan app.py dan pipeline.py."
        )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collections = {}
    for key, name in COLLECTION_NAMES.items():
        collections[key] = client.get_collection(name)
    return collections


# ---------------------------------------------------------------------------
# 2. Retrieval (RAG) helper
# ---------------------------------------------------------------------------

def retrieve_context(embedder, collection, query, n_results=3):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    docs = results["documents"][0]
    return "\n\n".join(docs)


# ---------------------------------------------------------------------------
# 3. Agent per divisi
# ---------------------------------------------------------------------------

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Kamu adalah {agent_name} di perusahaan retail PT Retailindo Sejahtera. "
     "Jawab pertanyaan HANYA berdasarkan konteks berikut. "
     "Jika informasi tidak ada di konteks, katakan dengan jujur bahwa kamu tidak punya data untuk itu "
     "daripada mengarang jawaban.\n\nKonteks:\n{context}"),
    ("human", "{question}")
])


def make_agent(agent_name, collection, embedder, llm):
    def agent_fn(question: str) -> dict:
        context = retrieve_context(embedder, collection, question)
        chain = AGENT_PROMPT | llm
        response = chain.invoke({"agent_name": agent_name, "context": context, "question": question})
        return {"agent": agent_name, "answer": response.content, "retrieved_context": context}
    return agent_fn


def make_finetuned_agent(agent_name, collection, embedder, text_pipeline, tokenizer):
    """Agent yang menjawab pakai model hasil fine-tuning lokal (Part 4/4.5), bukan Groq."""
    def agent_fn(question: str) -> dict:
        context = retrieve_context(embedder, collection, question)
        messages = [
            {"role": "system", "content": (
                f"Kamu adalah {agent_name} di PT Retailindo Sejahtera. "
                f"Jawab pertanyaan berdasarkan konteks berikut dengan sopan, jelas, dan solutif. "
                f"Jika informasi tidak ada di konteks, katakan dengan jujur.\n\nKonteks:\n{context}"
            )},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        output = text_pipeline(prompt)[0]["generated_text"]
        answer = output[len(prompt):].strip()
        return {"agent": agent_name, "answer": answer, "retrieved_context": context}
    return agent_fn


def build_agents(collections, embedder, llm, ft_pipeline=None, ft_tokenizer=None):
    """Bangun 4 agent divisi.

    Kalau model fine-tuned tersedia (ft_pipeline tidak None), agent Customer Service
    dan Billing/Retur pakai model fine-tuned lokal. Kalau tidak (misal fine-tuning
    belum dijalankan), fallback ke Groq untuk semua agent supaya dashboard tetap
    bisa dipakai selama development.
    """
    if ft_pipeline is not None and ft_tokenizer is not None:
        customer_service_agent = make_finetuned_agent(
            "Agent Customer Service (fine-tuned)", collections["customer_service"], embedder, ft_pipeline, ft_tokenizer
        )
        billing_returns_agent = make_finetuned_agent(
            "Agent Finance & Retur (fine-tuned)", collections["billing_returns"], embedder, ft_pipeline, ft_tokenizer
        )
    else:
        customer_service_agent = make_agent("Agent Customer Service", collections["customer_service"], embedder, llm)
        billing_returns_agent = make_agent("Agent Finance & Retur", collections["billing_returns"], embedder, llm)

    return {
        "customer_service": customer_service_agent,
        "billing_returns": billing_returns_agent,
        "inventory": make_agent("Agent Inventory & Store Operations", collections["inventory"], embedder, llm),
        "marketing": make_agent("Agent Marketing", collections["marketing"], embedder, llm),
    }


# ---------------------------------------------------------------------------
# 4. Orchestrator (LangGraph)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    route: List[str]
    responses: List[dict]
    final_answer: str


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Kamu adalah router/supervisor di PT Retailindo Sejahtera. Ada 4 agent divisi:\n"
     "- customer_service: pertanyaan umum, keluhan, akun\n"
     "- billing_returns: pertanyaan soal tagihan, pembayaran, retur, refund\n"
     "- inventory: pertanyaan soal stok, penjualan per toko/departemen\n"
     "- marketing: pertanyaan soal promosi, markdown, tren\n\n"
     "Balas HANYA dengan daftar nama agent yang relevan, dipisah koma (bisa lebih dari satu jika "
     "pertanyaan lintas divisi). Contoh output: customer_service,billing_returns"),
    ("human", "{question}")
])

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Gabungkan jawaban dari beberapa agent berikut menjadi satu jawaban akhir yang koheren untuk user. "
     "Jangan sebutkan proses internal, langsung jawab seperti satu asisten terpadu."),
    ("human", "Pertanyaan: {question}\n\nJawaban tiap agent:\n{agent_answers}")
])

VALID_ROUTES = ["customer_service", "billing_returns", "inventory", "marketing"]


def build_orchestrator(agents, llm):
    def router_node(state: AgentState) -> AgentState:
        chain = ROUTER_PROMPT | llm
        result = chain.invoke({"question": state["question"]})
        route = [r.strip() for r in result.content.split(",") if r.strip() in VALID_ROUTES]
        if not route:
            route = ["customer_service"]
        return {**state, "route": route}

    def agent_execution_node(state: AgentState) -> AgentState:
        responses = [agents[name](state["question"]) for name in state["route"]]
        return {**state, "responses": responses}

    def synthesis_node(state: AgentState) -> AgentState:
        agent_answers = "\n\n".join([f"[{r['agent']}]: {r['answer']}" for r in state["responses"]])
        chain = SYNTHESIS_PROMPT | llm
        result = chain.invoke({"question": state["question"], "agent_answers": agent_answers})
        return {**state, "final_answer": result.content}

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("execute_agents", agent_execution_node)
    graph.add_node("synthesize", synthesis_node)
    graph.set_entry_point("router")
    graph.add_edge("router", "execute_agents")
    graph.add_edge("execute_agents", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def get_orchestrator():
    """Entry point utama dipanggil dari app.py. Mengembalikan (orchestrator, llm).

    Otomatis memakai model fine-tuned untuk agent Customer Service & Billing/Retur
    kalau folder ./lora_finetuned_model ada; kalau tidak, fallback ke Groq untuk
    semua agent (misal saat dashboard dijalankan sebelum Part 4 notebook selesai).
    """
    embedder = load_embedder()
    llm = load_llm()
    collections = load_collections()
    ft_pipeline, ft_tokenizer = load_finetuned_pipeline()
    agents = build_agents(collections, embedder, llm, ft_pipeline, ft_tokenizer)
    orchestrator = build_orchestrator(agents, llm)
    return orchestrator, llm


# ---------------------------------------------------------------------------
# 5. Evaluator
# ---------------------------------------------------------------------------

EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Kamu adalah evaluator independen untuk sistem AI customer service. "
     "Nilai jawaban AI berikut berdasarkan konteks yang tersedia, dengan skor 1-5 (5=terbaik) untuk tiap dimensi:\n"
     "- accuracy: seberapa akurat jawaban dibanding konteks\n"
     "- effectiveness: seberapa efektif jawaban menyelesaikan masalah user\n"
     "- explainability: seberapa jelas jawaban bisa dilacak sumbernya dari konteks\n"
     "- hallucination: seberapa RENDAH tingkat karangan/informasi di luar konteks (5=tidak ada halusinasi sama sekali)\n\n"
     "Balas HANYA dalam format JSON valid, tanpa teks tambahan, contoh:\n"
     '{{"accuracy": 4, "effectiveness": 5, "explainability": 3, "hallucination": 5, "reasoning": "penjelasan singkat"}}'),
    ("human", "Pertanyaan: {question}\n\nKonteks yang tersedia:\n{context}\n\nJawaban AI:\n{answer}")
])


def latency_to_efficiency_score(latency_sec: float) -> int:
    """Ubah waktu respons agent jadi skor 1-5 (5=paling efisien), supaya Efficiency
    jadi dimensi skor eksplisit sejajar dengan 4 dimensi lain sesuai soal UAS poin 4.
    """
    if latency_sec < 2:
        return 5
    elif latency_sec < 4:
        return 4
    elif latency_sec < 6:
        return 3
    elif latency_sec < 8:
        return 2
    else:
        return 1


def evaluate_response(llm, question, context, answer, response_latency_sec):
    start = time.time()
    chain = EVAL_PROMPT | llm
    result = chain.invoke({"question": question, "context": context, "answer": answer})
    eval_latency = time.time() - start

    try:
        scores = json.loads(result.content)
    except json.JSONDecodeError:
        scores = {"accuracy": None, "effectiveness": None, "explainability": None,
                   "hallucination": None, "reasoning": "gagal parse JSON dari evaluator"}

    # Efficiency dihitung dari waktu respons AGENT (bukan waktu evaluator menilai)
    scores["efficiency"] = latency_to_efficiency_score(response_latency_sec)
    scores["response_latency_sec"] = round(response_latency_sec, 2)
    scores["eval_latency_sec"] = round(eval_latency, 2)
    scores["estimated_tokens"] = len(answer.split())
    return scores


def evaluate_orchestrator_run(orchestrator, llm, question: str) -> dict:
    """Jalankan orchestrator untuk satu pertanyaan lalu evaluasi hasilnya.
    Ini fungsi utama yang dipanggil dari tab 'Chat' di app.py supaya user
    langsung lihat skor evaluasi di samping jawaban.
    """
    start = time.time()
    result = orchestrator.invoke({"question": question})
    total_latency = time.time() - start

    combined_context = "\n\n".join([r["retrieved_context"] for r in result["responses"]])
    eval_scores = evaluate_response(llm, question, combined_context, result["final_answer"], total_latency)
    eval_scores["route"] = result["route"]
    eval_scores["total_latency_sec"] = round(total_latency, 2)
    eval_scores["final_answer"] = result["final_answer"]
    return eval_scores


def run_batch_evaluation(orchestrator, llm, questions: list) -> pd.DataFrame:
    """Dipakai untuk generate evaluation_results.csv yang ditampilkan di tab Evaluasi."""
    records = []
    for q in questions:
        r = evaluate_orchestrator_run(orchestrator, llm, q)
        records.append({
            "question": q,
            "route": ", ".join(r.get("route", [])),
            "accuracy": r.get("accuracy"),
            "effectiveness": r.get("effectiveness"),
            "efficiency": r.get("efficiency"),
            "explainability": r.get("explainability"),
            "hallucination": r.get("hallucination"),
            "latency_sec": r.get("total_latency_sec"),
        })
    return pd.DataFrame(records)
