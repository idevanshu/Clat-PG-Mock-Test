# main.py
import streamlit as st
import pdfplumber
import re
import glob
import os
import random
import time
import json
from datetime import datetime, timedelta

st.set_page_config(
    page_title="CLAT PG Mock Test",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Custom CSS - Fixed text visibility ----------------
st.markdown("""
<style>
    /* Force light theme background for question area */
    .main {
        background-color: #f8f9fa;
    }
    
    .block-container {
        padding: 1rem !important;
        max-width: 1200px !important;
    }
    
    /* Buttons */
    .stButton button {
        width: 100%;
        border-radius: 10px;
        padding: 12px 20px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Radio buttons - FIXED VISIBILITY */
    .stRadio > label {
        color: #1a202c !important;
        font-weight: 600;
        font-size: 16px;
    }
    
    .stRadio > div {
        gap: 12px;
    }
    
    .stRadio [data-baseweb="radio"] {
        background-color: white !important;
        padding: 16px !important;
        border: 2px solid #cbd5e0 !important;
        border-radius: 10px !important;
        margin-bottom: 8px !important;
    }
    
    .stRadio [data-baseweb="radio"]:hover {
        border-color: #667eea !important;
        background-color: #f7fafc !important;
    }
    
    /* Radio label text - HIGH CONTRAST */
    .stRadio label div {
        color: #1a202c !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        font-weight: 500 !important;
    }
    
    .stRadio label span {
        color: #1a202c !important;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: white !important;
        font-weight: 700;
    }
    
    /* Question box - WHITE background with BLACK text */
    .question-box {
        background-color: #ffffff !important;
        padding: 28px;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        margin: 20px 0;
        border-left: 5px solid #667eea;
    }
    
    .question-number {
        color: #667eea !important;
        font-weight: 700;
        font-size: 22px;
        margin-bottom: 18px;
        display: block;
    }
    
    .question-text {
        color: #000000 !important;
        font-size: 19px !important;
        line-height: 1.8 !important;
        font-weight: 500 !important;
        margin-bottom: 24px !important;
    }
    
    /* Instructions */
    .instructions-panel {
        background: #ffffff;
        padding: 28px;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        margin: 20px 0;
    }
    
    .instructions-panel h3 {
        color: #1a202c !important;
    }
    
    .instructions-panel p, .instructions-panel li {
        color: #2d3748 !important;
        font-size: 15px;
        line-height: 1.8;
    }
    
    /* Info/warning boxes */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Mobile optimization */
    @media (max-width: 768px) {
        .block-container {
            padding: 0.5rem !important;
        }
        
        .question-box {
            padding: 20px;
        }
        
        .question-number {
            font-size: 19px;
        }
        
        .question-text {
            font-size: 17px !important;
        }
        
        .stRadio label div {
            font-size: 15px !important;
        }
    }
    
    /* Desktop optimization */
    @media (min-width: 769px) {
        .block-container {
            padding: 2rem 3rem !important;
        }
        
        .question-box {
            padding: 32px;
        }
        
        .question-text {
            font-size: 20px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Config ----------------
FOLDER = "questions"
LETTERS = ["A", "B", "C", "D"]
DEFAULT_DURATION_MIN = 120
MARKS_CORRECT = 1.0
MARKS_NEGATIVE = 0.25
TARGET_COUNT = 120

CLAT_INSTRUCTIONS = """
### 📋 CLAT PG 2026 - Exam Instructions

**General Instructions:**

1. **Duration:** The test is of 2 hours (120 minutes)
2. **Questions:** The test contains 120 multiple-choice questions
3. **Marking Scheme:** 
   - Each correct answer: +1 mark
   - Each incorrect answer: -0.25 marks (Negative marking)
   - Unanswered questions: No marks deducted
4. **Question Type:** All questions are Multiple Choice Questions (MCQs) with four options (A, B, C, D)

**During the Test:**

- You can navigate between questions using Previous/Next buttons
- You can skip questions and return to them later
- Mark your answers carefully - only one option per question
- Keep track of time using the timer displayed at the top
- Submit the test when you're done or it will auto-submit when time is up

**Important Rules:**

- Do not refresh the page during the test
- Ensure stable internet connection
- No external materials or calculators allowed
- Focus on accuracy - negative marking applies
- Manage your time wisely across all 120 questions

**Test Sections (CLAT PG):**

Constitutional Law • Jurisprudence • Contract Law • Tort Law • Criminal Law • Property Law • Administrative Law • International Law

**Good Luck! 🎓**
"""

# ------------- Parsing helpers (same) -------------
def normalize_lines(text: str):
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"-\n", "", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return [ln for ln in lines if ln]

def is_option_line(s: str, letter: str):
    return re.match(rf"^\(?{letter}\)?[\.\:\-\)]\s+", s, flags=re.IGNORECASE) is not None

def strip_option_prefix(s: str, letter: str):
    return re.sub(rf"^\(?{letter}\)?[\.\:\-\)]\s+", "", s, flags=re.IGNORECASE).strip()

def is_question_start(s: str):
    return re.match(r"^(?:Q(?:uestion)?\.?\s*)?\d{1,3}[\)\.\:\-]\s+|^(?:Q(?:uestion)?\.?\s*)\d{1,3}\s+", s, flags=re.IGNORECASE) is not None

def strip_question_prefix(s: str):
    return re.sub(r"^(?:Q(?:uestion)?\.?\s*)?\d{1,3}[\)\.\:\-]?\s*", "", s, flags=re.IGNORECASE).strip()

def find_answer_in_window(lines, start, window=12):
    chunk = "\n".join(lines[start : start + window])
    m = re.search(r"(?:CORRECT\s*OPTION|Correct\s*Option|Answer|Ans\.?)\s*[:\-]?\s*([ABCD])", chunk, re.IGNORECASE)
    if m:
        return LETTERS.index(m.group(1).upper())
    return None

def parse_questions_from_text(text: str):
    lines = normalize_lines(text)
    res = []
    i = 0
    n = len(lines)
    while i < n:
        if not is_question_start(lines[i]):
            i += 1
            continue

        stem_parts = [strip_question_prefix(lines[i])]
        j = i + 1
        while j < n and not any(is_option_line(lines[j], L) for L in LETTERS):
            if is_question_start(lines[j]):
                break
            stem_parts.append(lines[j])
            j += 1

        options = []
        ok = True
        for L in LETTERS:
            if j < n and is_option_line(lines[j], L):
                options.append(strip_option_prefix(lines[j], L))
                j += 1
            else:
                ok = False
                break
        if not ok or len(options) != 4:
            i = max(i + 1, j)
            continue

        ans_idx = find_answer_in_window(lines, j, window=12)
        q = {
            "question": " ".join(stem_parts).strip(),
            "options": options,
            "answer_index": ans_idx,
        }
        if len(q["question"]) > 10:
            res.append(q)
        i = max(i + 1, j)
    return res

@st.cache_data(show_spinner=False, ttl=3600)
def extract_pool_from_folder(folder: str):
    pdf_paths = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    pool = []
    for p in pdf_paths:
        try:
            with pdfplumber.open(p) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception:
            continue
        if not text:
            continue
        qs = parse_questions_from_text(text)
        for q in qs:
            q["source"] = os.path.basename(p)
        pool.extend(qs)

    seen = set()
    uniq = []
    for q in pool:
        key = (q["question"][:160].lower(), q["options"][0][:80].lower() if q["options"] else "")
        if key not in seen:
            seen.add(key)
            uniq.append(q)
    return uniq

def prepare_exam_set(pool, count=TARGET_COUNT, seed=42):
    rng = random.Random(seed)
    if len(pool) == 0:
        return []
    if len(pool) <= count:
        subset = pool[:]
        rng.shuffle(subset)
        return subset
    return rng.sample(pool, count)

# ---------------- Session state ----------------
def ensure_state():
    st.session_state.setdefault("stage", "idle")
    st.session_state.setdefault("questions", [])
    st.session_state.setdefault("answers", {})
    st.session_state.setdefault("current", 0)
    st.session_state.setdefault("end_ts", None)
    st.session_state.setdefault("duration_min", DEFAULT_DURATION_MIN)
    st.session_state.setdefault("show_palette", False)

ensure_state()

# ---------------- Header ----------------
st.title("⚖️ CLAT PG Mock Test 2026")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### ⚙️ Test Settings")
    folder = st.text_input("📁 PDF Folder", value=FOLDER)
    duration_min = st.number_input("⏱️ Duration (minutes)", 10, 240, st.session_state["duration_min"], 5)
    seed = st.number_input("🎲 Random Seed", 0, 9999, 42, 1)
    
    st.markdown("---")
    prep_clicked = st.button("🔄 Prepare 120 Questions", use_container_width=True, 
                             disabled=st.session_state["stage"] in ["started", "instructions"])
    reset_clicked = st.button("🔁 Reset Everything", use_container_width=True)

if reset_clicked:
    st.session_state.update(stage="idle", questions=[], answers={}, current=0, end_ts=None, 
                           duration_min=DEFAULT_DURATION_MIN, show_palette=False)
    st.rerun()

if prep_clicked:
    st.session_state["duration_min"] = int(duration_min)
    with st.status("🔍 Preparing 120 questions...", expanded=True) as status:
        pool = extract_pool_from_folder(folder)
        subset = prepare_exam_set(pool, count=TARGET_COUNT, seed=int(seed))
        if not subset:
            st.error("❌ No questions found in PDFs")
            status.update(label="❌ Failed", state="error")
        else:
            st.session_state["questions"] = subset
            st.session_state["answers"] = {}
            st.session_state["current"] = 0
            st.session_state["stage"] = "instructions"
            st.success(f"✅ Loaded {len(subset)} questions!")
            status.update(label=f"✅ Ready: {len(subset)} questions", state="complete")
    st.rerun()

# --------------- Metrics ---------------
def remaining_seconds():
    if not st.session_state.get("end_ts"):
        return None
    return max(0, int(st.session_state["end_ts"] - time.time()))

rem = remaining_seconds() if st.session_state["stage"] == "started" else None

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Status", st.session_state["stage"].title())
with col2:
    st.metric("📝 Questions", f"{len(st.session_state.get('questions', []))}")
with col3:
    st.metric("⏱️ Time", f"{(rem or 0)//60:02d}:{(rem or 0)%60:02d}")
with col4:
    attempted = sum(1 for v in st.session_state["answers"].values() if v is not None)
    st.metric("✓ Done", f"{attempted}")

st.markdown("---")

# --------------- Stages ---------------
if st.session_state["stage"] == "idle":
    st.info("👋 Click **'Prepare 120 Questions'** in the sidebar to start")

elif st.session_state["stage"] == "instructions":
    st.markdown('<div class="instructions-panel">', unsafe_allow_html=True)
    st.markdown(CLAT_INSTRUCTIONS)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("▶️ Start Test Now", type="primary", use_container_width=True):
            st.session_state["end_ts"] = (datetime.now() + timedelta(minutes=st.session_state["duration_min"])).timestamp()
            st.session_state["stage"] = "started"
            st.rerun()

elif st.session_state["stage"] == "started":
    if rem is not None and rem <= 0:
        st.warning("⏰ Time's up! Auto-submitting...")
        st.session_state["stage"] = "submitted"
        st.rerun()

    qs = st.session_state["questions"]
    idx = st.session_state["current"]
    q = qs[idx]

    # Question with HIGH CONTRAST
    st.markdown(f"""
    <div class="question-box">
        <span class="question-number">Question {idx+1} of {len(qs)}</span>
        <div class="question-text">{q["question"]}</div>
    </div>
    """, unsafe_allow_html=True)

    # Options
    opts = [f"{LETTERS[i]}) {txt}" for i, txt in enumerate(q["options"])]
    saved = st.session_state["answers"].get(idx, None)
    default = 0 if saved is None else (saved + 1)
    
    choice = st.radio(
        "Select your answer:",
        ["⊗ Skip"] + opts,
        index=default,
        key=f"q_{idx}"
    )
    
    st.session_state["answers"][idx] = None if choice == "⊗ Skip" else ["A", "B", "C", "D"].index(choice[0])

    st.markdown("---")
    
    nav = st.columns([2, 2, 2])
    with nav[0]:
        if st.button("⬅️ Previous", disabled=idx == 0, use_container_width=True):
            st.session_state["current"] = idx - 1
            st.rerun()
    with nav[1]:
        if st.button("🔢 Palette", use_container_width=True):
            st.session_state["show_palette"] = not st.session_state.get("show_palette", False)
            st.rerun()
    with nav[2]:
        if st.button("Next ➡️", disabled=idx >= len(qs) - 1, use_container_width=True):
            st.session_state["current"] = idx + 1
            st.rerun()
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("✓ Submit Test", type="primary", use_container_width=True):
            st.session_state["stage"] = "submitted"
            st.rerun()

    if st.session_state.get("show_palette", False):
        st.markdown("### 🔢 Question Navigator")
        per_row = 10
        rows = (len(qs) + per_row - 1) // per_row
        for r in range(rows):
            cols = st.columns(per_row)
            for c in range(per_row):
                k = r * per_row + c
                if k >= len(qs):
                    continue
                attempted = st.session_state["answers"].get(k, None) is not None
                label = f"{'✓' if attempted else ''}{k+1}"
                if cols[c].button(label, key=f"j_{k}", use_container_width=True):
                    st.session_state["current"] = k
                    st.rerun()

elif st.session_state["stage"] == "submitted":
    qs = st.session_state["questions"]
    ans = st.session_state["answers"]
    correct = wrong = unattempt = unscored = 0
    score = 0.0
    
    for i, q in enumerate(qs):
        chosen = ans.get(i, None)
        true_idx = q.get("answer_index", None)
        if chosen is None:
            unattempt += 1
        elif true_idx is None:
            unscored += 1
        elif chosen == true_idx:
            correct += 1
            score += MARKS_CORRECT
        else:
            wrong += 1
            score -= MARKS_NEGATIVE

    st.balloons()
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 3rem; border-radius: 16px; color: white; text-align: center;">
        <h1 style="margin: 0; font-size: 56px;">{score:.2f}</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 24px;">Final Score</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Correct", correct)
    with col2:
        st.metric("❌ Wrong", wrong)
    with col3:
        st.metric("⊗ Skipped", unattempt)
    with col4:
        st.metric("⚠️ Unscored", unscored)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🔄 New Test", type="primary", use_container_width=True):
            st.session_state.update(stage="idle", questions=[], answers={}, current=0, end_ts=None, show_palette=False)
            st.rerun()
