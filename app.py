"""
app.py — Dashboard Streamlit untuk PT Retailindo Sejahtera Multi-Agent Assistant
(Versi redesain — tema navy modern, agent color-coding)

Jalankan dengan:
    streamlit run app.py

Pastikan sebelum menjalankan:
1. Folder chroma_db/ (hasil Part 0-2 notebook) ada di direktori yang sama
2. Environment variable GROQ_API_KEY sudah di-set
3. pipeline.py ada di direktori yang sama dengan file ini
4. (Opsional) lora_finetuned_model/ ada di direktori yang sama, untuk agent CS & Billing
"""

import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import get_orchestrator, evaluate_orchestrator_run, run_batch_evaluation

st.set_page_config(
    page_title="PT Retailindo Sejahtera — Multi-Agent Assistant",
    page_icon="🏬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design tokens — satu tempat untuk semua warna/font, biar konsisten
# ---------------------------------------------------------------------------

COLORS = {
    "bg": "#0B0F1A",
    "surface": "#141B2E",
    "surface_alt": "#1B2440",
    "border": "#2A3350",
    "text": "#EDEFF5",
    "text_muted": "#8B93A8",
    "brand": "#4C7EFF",
}

AGENT_COLORS = {
    "customer_service": "#45D6C4",   # teal
    "billing_returns": "#F0A83D",    # amber
    "inventory": "#5FD98A",          # green
    "marketing": "#B18AF0",          # violet
}

AGENT_LABELS = {
    "customer_service": "Customer Service",
    "billing_returns": "Finance & Retur",
    "inventory": "Inventory",
    "marketing": "Marketing",
}


def agent_color(route_list):
    """Ambil warna dari agent pertama yang menjawab, untuk aksen bubble/card."""
    if not route_list:
        return COLORS["brand"]
    return AGENT_COLORS.get(route_list[0], COLORS["brand"])


# ---------------------------------------------------------------------------
# Inject custom CSS
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}}

/* Header banner */
.rs-header {{
    padding: 1.75rem 2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['surface_alt']} 100%);
    border: 1px solid {COLORS['border']};
    margin-bottom: 1.25rem;
}}
.rs-header h1 {{
    margin: 0;
    font-size: 1.9rem;
    color: {COLORS['text']};
}}
.rs-header p {{
    margin: 0.35rem 0 0 0;
    color: {COLORS['text_muted']};
    font-size: 0.95rem;
}}

/* Agent status strip — signature element */
.rs-agent-strip {{
    display: flex;
    gap: 0.6rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}}
.rs-chip {{
    display: flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    font-size: 0.82rem;
    color: {COLORS['text']};
}}
.rs-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}}

/* Chat bubble with agent color-coded left border */
.rs-chat-answer {{
    border-left: 3px solid var(--agent-color, {COLORS['brand']});
    background: {COLORS['surface']};
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    margin: 0.4rem 0 0.6rem 0;
}}
.rs-chat-meta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: {COLORS['text_muted']};
    margin-top: 0.5rem;
}}

/* Score mini-cards */
.rs-score-row {{
    display: flex;
    gap: 0.6rem;
    margin-top: 0.75rem;
    flex-wrap: wrap;
}}
.rs-score-card {{
    flex: 1;
    min-width: 100px;
    background: {COLORS['surface_alt']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    text-align: center;
}}
.rs-score-label {{
    font-size: 0.72rem;
    color: {COLORS['text_muted']};
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.rs-score-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: {COLORS['text']};
    margin-top: 0.15rem;
}}

/* Streamlit native element overrides */
.stButton>button {{
    background: {COLORS['brand']};
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
}}
.stButton>button:hover {{
    background: #3D68DB;
    color: white;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace;
}}

/* Architecture diagram (Tab 3) */
.rs-flow {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 1.5rem 0;
}}
.rs-flow-box {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    text-align: center;
    min-width: 140px;
    font-size: 0.85rem;
}}
.rs-flow-arrow {{
    color: {COLORS['text_muted']};
    font-size: 1.3rem;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Setup (di-cache supaya orchestrator/LLM/vector DB hanya di-load sekali)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Menyiapkan multi-agent system (embedder, LLM, vector DB)...")
def init_system():
    orchestrator, llm = get_orchestrator()
    return orchestrator, llm


try:
    orchestrator, llm = init_system()
    system_ready = True
    init_error = None
except Exception as e:
    system_ready = False
    init_error = str(e)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

status_chips = ""
for key, label in AGENT_LABELS.items():
    color = AGENT_COLORS[key]
    dot_color = color if system_ready else "#555F7A"
    status_chips += f"""
    <div class="rs-chip">
        <span class="rs-dot" style="background:{dot_color};"></span>
        {label}
    </div>"""

st.markdown(f"""
<div class="rs-header">
    <h1>🏬 PT Retailindo Sejahtera</h1>
    <p>Multi-Agent Enterprise Assistant — Proyek Data Mining (ST167)</p>
    <div class="rs-agent-strip">{status_chips}</div>
</div>
""", unsafe_allow_html=True)

if not system_ready:
    st.error(f"Sistem belum siap: {init_error}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ℹ️ Tentang Sistem")
    st.markdown("""
    **4 Agent Divisi:**
    - 🟢 Customer Service
    - 🟠 Finance & Retur
    - 🟢 Inventory
    - 🟣 Marketing

    **Orchestrator** merutekan pertanyaan ke agent yang relevan, bisa lebih dari satu jika lintas divisi.
    """)
    st.markdown("---")
    st.markdown("### 🧪 Coba Pertanyaan Ini")
    sample_questions = [
        "Bagaimana cara reset password akun saya?",
        "Toko mana yang penjualannya paling tinggi?",
        "Toko mana yang markdown-nya paling tinggi?",
    ]
    for q in sample_questions:
        st.caption(f"• {q}")


tab1, tab2, tab3 = st.tabs(["💬 Chat Multi-Agent", "📊 Evaluasi", "🧩 Arsitektur"])

# ---------------------------------------------------------------------------
# TAB 1 — Chat
# ---------------------------------------------------------------------------
with tab1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col_input, col_toggle = st.columns([4, 1])
    with col_input:
        user_question = st.text_input(
            "Pertanyaan Anda",
            placeholder="Contoh: Saya mau retur produk yang cacat, bagaimana prosesnya?",
            disabled=not system_ready,
            label_visibility="collapsed",
        )
    with col_toggle:
        show_eval = st.checkbox("Sertakan skor", value=True)

    send = st.button("Kirim →", disabled=not system_ready)

    if send and user_question:
        with st.spinner("Agent sedang memproses..."):
            if show_eval:
                result = evaluate_orchestrator_run(orchestrator, llm, user_question)
                answer = result["final_answer"]
                route = result["route"]
                latency = result["total_latency_sec"]
                scores = {
                    "accuracy": result.get("accuracy"),
                    "effectiveness": result.get("effectiveness"),
                    "efficiency": result.get("efficiency"),
                    "explainability": result.get("explainability"),
                    "hallucination": result.get("hallucination"),
                }
            else:
                start = time.time()
                raw = orchestrator.invoke({"question": user_question})
                answer = raw["final_answer"]
                route = raw["route"]
                latency = round(time.time() - start, 2)
                scores = None

        st.session_state.chat_history.append({
            "question": user_question,
            "answer": answer,
            "route": route,
            "latency": latency,
            "scores": scores,
        })

    for chat in reversed(st.session_state.chat_history):
        color = agent_color(chat["route"])
        route_label = ", ".join(AGENT_LABELS.get(r, r) for r in chat["route"])

        st.markdown(f"**Anda:** {chat['question']}")
        st.markdown(f"""
        <div class="rs-chat-answer" style="--agent-color:{color};">
            {chat['answer']}
            <div class="rs-chat-meta">🤖 {route_label} · ⏱ {chat['latency']}s</div>
        </div>
        """, unsafe_allow_html=True)

        if chat["scores"]:
            cards = ""
            for label, key in [("Accuracy", "accuracy"), ("Effective.", "effectiveness"),
                                ("Efficient.", "efficiency"), ("Explain.", "explainability"),
                                ("Halluc.", "hallucination")]:
                val = chat["scores"].get(key)
                cards += f"""
                <div class="rs-score-card">
                    <div class="rs-score-label">{label}</div>
                    <div class="rs-score-value">{val if val is not None else '–'}/5</div>
                </div>"""
            st.markdown(f'<div class="rs-score-row">{cards}</div>', unsafe_allow_html=True)

        st.markdown("&nbsp;", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 2 — Evaluasi
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("#### Evaluasi Sistem (Batch)")

    default_questions = [
        "Bagaimana cara reset password akun saya?",
        "Saya mau retur produk yang cacat, bagaimana prosesnya?",
        "Toko mana yang penjualannya paling tinggi bulan ini?",
        "Kenapa promosi minggu ini kurang efektif?",
        "Berapa lama proses refund biasanya cair?",
    ]

    questions_text = st.text_area(
        "Daftar pertanyaan uji (satu per baris):",
        value="\n".join(default_questions),
        height=140,
        disabled=not system_ready,
    )

    if st.button("▶ Jalankan Evaluasi Batch", disabled=not system_ready):
        questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
        with st.spinner(f"Mengevaluasi {len(questions)} pertanyaan..."):
            eval_df = run_batch_evaluation(orchestrator, llm, questions)
            eval_df.to_csv("evaluation_results.csv", index=False)
            st.session_state.eval_df = eval_df

    if "eval_df" in st.session_state:
        eval_df = st.session_state.eval_df
    else:
        try:
            eval_df = pd.read_csv("evaluation_results.csv")
        except FileNotFoundError:
            eval_df = None

    if eval_df is not None:
        metrics = ["accuracy", "effectiveness", "efficiency", "explainability", "hallucination"]
        metric_colors = ["#4C7EFF", "#45D6C4", "#F0A83D", "#5FD98A", "#B18AF0"]

        cols = st.columns(5)
        for col, m in zip(cols, metrics):
            col.metric(m.capitalize(), f"{eval_df[m].mean():.2f}/5")

        fig = go.Figure()
        for m, c in zip(metrics, metric_colors):
            fig.add_trace(go.Bar(
                name=m.capitalize(),
                x=[f"Q{i+1}" for i in range(len(eval_df))],
                y=eval_df[m],
                marker_color=c,
            ))
        fig.update_layout(
            barmode="group",
            paper_bgcolor=COLORS["bg"],
            plot_bgcolor=COLORS["bg"],
            font=dict(color=COLORS["text"], family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=40, l=10, r=10, b=10),
            yaxis=dict(range=[0, 5], gridcolor=COLORS["border"]),
            xaxis=dict(gridcolor=COLORS["border"]),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Lihat tabel detail & pertanyaan"):
            st.dataframe(eval_df, use_container_width=True)
    else:
        st.info("Belum ada hasil evaluasi. Klik 'Jalankan Evaluasi Batch' di atas.")

# ---------------------------------------------------------------------------
# TAB 3 — Arsitektur
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("#### Alur Sistem Multi-Agent")

    st.markdown(f"""
    <div class="rs-flow">
        <div class="rs-flow-box">👤 User Question</div>
        <div class="rs-flow-arrow">→</div>
        <div class="rs-flow-box" style="border-color:{COLORS['brand']};">🧭 Orchestrator<br><span style="color:{COLORS['text_muted']};font-size:0.75rem;">(LangGraph router)</span></div>
        <div class="rs-flow-arrow">→</div>
        <div class="rs-flow-box" style="border-color:{AGENT_COLORS['customer_service']};">🟢 Customer Service<br><span style="font-size:0.7rem;">fine-tuned LoRA</span></div>
    </div>
    <div class="rs-flow">
        <div class="rs-flow-box" style="border-color:{AGENT_COLORS['billing_returns']};">🟠 Finance & Retur<br><span style="font-size:0.7rem;">fine-tuned LoRA</span></div>
        <div class="rs-flow-box" style="border-color:{AGENT_COLORS['inventory']};">🟢 Inventory<br><span style="font-size:0.7rem;">Groq + RAG</span></div>
        <div class="rs-flow-box" style="border-color:{AGENT_COLORS['marketing']};">🟣 Marketing<br><span style="font-size:0.7rem;">Groq + RAG</span></div>
    </div>
    <div class="rs-flow">
        <div class="rs-flow-box">💬 Jawaban Tersintesis</div>
        <div class="rs-flow-arrow">←</div>
        <div class="rs-flow-box">📊 Evaluator<br><span style="font-size:0.7rem;">5 dimensi skor</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Teknologi yang dipakai:**
        - LangChain + LangGraph (orkestrasi)
        - ChromaDB (vector DB)
        - sentence-transformers (embedding)
        - LoRA fine-tuning — Qwen2.5-0.5B (agent CS & Billing)
        - Groq (Llama 3.3) — agent Inventory & Marketing
        """)
    with col_b:
        st.markdown(f"""
        **Status koneksi backend:**
        {'✅ Terhubung ke pipeline.py' if system_ready else '❌ Belum terhubung — lihat pesan error di atas'}

        **Evaluasi mencakup 5 dimensi:**
        Accuracy, Effectiveness, Efficiency, Explainability, Hallucination
        """)
