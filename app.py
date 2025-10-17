# main.py - CLAT PG Mock Test with proper passage detection
import streamlit as st
import pdfplumber
import re
import glob
import os
import random
import time
from datetime import datetime, timedelta
from collections import defaultdict

st.set_page_config(
    page_title="CLAT PG Mock Test",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CSS Styling ----------------
st.markdown("""
<style>
    .main { background-color: #0a0e27; }
    .block-container { padding: 1rem !important; max-width: 1200px !important; }
    .stButton button { 
        width: 100%; border-radius: 10px; padding: 12px 20px; 
        font-weight: 600; font-size: 16px; transition: all 0.3s ease; border: none; 
    }
    .stButton button[kind="primary"] { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; 
    }
    .stRadio > label { color: #e5e7eb !important; font-weight: 600; font-size: 16px; }
    .stRadio > div { gap: 12px; }
    .stRadio [data-baseweb="radio"] { 
        background-color: #1a1f3a !important; padding: 16px !important; 
        border: 2px solid #374151 !important; border-radius: 10px !important; margin-bottom: 8px !important; 
    }
    .stRadio [data-baseweb="radio"]:hover { border-color: #667eea !important; background-color: #0d1128 !important; }
    .stRadio label div { color: #e5e7eb !important; font-size: 16px !important; line-height: 1.6 !important; font-weight: 500 !important; }
    .stRadio label span { color: #e5e7eb !important; }
    [data-testid="stMetric"] { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 16px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: white !important; font-weight: 700; }
    .passage-box { 
        background-color: #fffbf0 !important; padding: 26px; border-radius: 12px; 
        border: 3px solid #f59e0b; border-left: 6px solid #d97706; 
        margin: 20px 0; font-size: 16px; line-height: 1.85; color: #1f2937; 
        max-height: 500px; overflow-y: auto; box-shadow: 0 4px 14px rgba(245, 158, 11, 0.2); 
    }
    .passage-box h4 { 
        color: #c2410c !important; margin-top: 0; margin-bottom: 16px; 
        font-size: 19px; font-weight: 800; border-bottom: 2px solid #fdba74; padding-bottom: 8px; 
    }
    .question-box { 
        background-color: #141829 !important; padding: 28px; border-radius: 12px; 
        box-shadow: 0 2px 12px rgba(0,0,0,0.4); margin: 20px 0; border-left: 5px solid #667eea; 
    }
    .question-number { 
        color: #a5b4fc !important; font-weight: 700; font-size: 22px; margin-bottom: 18px; display: block; 
    }
    .question-text { 
        color: #f3f4f6 !important; font-size: 19px !important; line-height: 1.8 !important; 
        font-weight: 500 !important; margin-bottom: 24px !important; 
    }
    .subject-badge { 
        display: inline-block; background: #1e293b; color: #e0e7ff; 
        padding: 7px 15px; border-radius: 20px; font-size: 14px; 
        font-weight: 600; margin-bottom: 12px; border: 1px solid #475569; 
    }
    .subject-header { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        color: white; padding: 12px 20px; border-radius: 8px; 
        margin: 20px 0 10px 0; font-weight: 700; font-size: 16px; 
    }
    .instructions-panel { 
        background: #141829; padding: 28px; border-radius: 12px; 
        border: 2px solid #334155; margin: 20px 0; 
    }
    .instructions-panel h3 { color: #e5e7eb !important; }
    .instructions-panel p, .instructions-panel li { color: #cbd5e1 !important; font-size: 15px; line-height: 1.8; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    
    @media (max-width: 768px) {
        .block-container { padding: 0.5rem !important; }
        .passage-box { padding: 18px; font-size: 15px; max-height: 350px; }
        .question-box { padding: 20px; }
        .question-number { font-size: 19px; }
        .question-text { font-size: 17px !important; }
        .stRadio label div { font-size: 15px !important; }
    }
    @media (min-width: 769px) {
        .block-container { padding: 2rem 3rem !important; }
        .question-box { padding: 32px; }
        .question-text { font-size: 20px !important; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Configuration ----------------
FOLDER = "questions"
LETTERS = ["A", "B", "C", "D"]
DEFAULT_DURATION_MIN = 120
MARKS_CORRECT = 1.0
MARKS_NEGATIVE = 0.25
TARGET_COUNT = 120

SUBJECT_KEYWORDS = {
    "Constitutional Law": ["constitution", "fundamental rights", "dpsp", "article", "amendment", "judicial review", "preamble", "citizenship"],
    "Jurisprudence": ["jurisprudence", "legal theory", "natural law", "positivism", "austin", "kelsen", "hart", "dworkin", "pure theory"],
    "Contract Law": ["contract", "agreement", "offer", "acceptance", "consideration", "breach", "damages", "specific performance"],
    "Criminal Law": ["ipc", "criminal", "murder", "theft", "cheating", "culpable homicide", "penal code", "offence", "punishment"],
    "Tort Law": ["tort", "negligence", "nuisance", "defamation", "trespass", "strict liability", "vicarious liability"],
    "Property Law": ["property", "transfer", "easement", "mortgage", "lease", "possession", "ownership", "immovable", "article 300a"],
    "Administrative Law": ["administrative", "delegated legislation", "natural justice", "tribunal", "ombudsman", "rule of law"],
    "Company Law": ["company", "director", "shareholder", "corporate", "board", "companies act", "memorandum", "articles"],
    "Family Law": ["marriage", "divorce", "maintenance", "custody", "adoption", "hindu marriage", "succession"],
    "International Law": ["international", "treaty", "sovereignty", "united nations", "icc", "geneva convention", "bilateral"],
}

SUBJECT_DISTRIBUTION = {
    "Constitutional Law": 32,
    "Jurisprudence": 22,
    "Contract Law": 11,
    "Criminal Law": 11,
    "Tort Law": 9,
    "Property Law": 9,
    "Administrative Law": 9,
    "Company Law": 6,
    "International Law": 5,
    "Family Law": 6,
}

CLAT_INSTRUCTIONS = """
### 📋 CLAT PG 2026 - Exam Instructions

**General Instructions:**

1. **Duration:** 2 hours (120 minutes)
2. **Questions:** 120 MCQs with passage-based comprehension
3. **Marking Scheme:** +1 for correct, -0.25 for wrong, 0 for unattempted
4. **Question Format:**
   - Passages followed by 3–8 questions per passage
   - Read passages carefully before attempting questions

**Subject-wise Navigation:**

Questions are organized by subjects. Use the palette to jump between subjects and questions.

**Good Luck! 🎓**
"""

# ---------------- Parsing Helpers ----------------
PASSAGE_KEYWORDS = [
    "article", "section", "constitution", "supreme court", "high court", "judgment",
    "held that", "observed", "scc", "air", "v.", "rights", "liberty", "bail",
    "property", "article 300a", "criminal", "jurisdiction", "statute", "hindu", "contract"
]

def normalize_lines(text: str):
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"-\n", "", text)
    lines = [ln.strip() for ln in text.split("\n")]
    return [ln for ln in lines if ln]

def is_noise(line: str) -> bool:
    if not line or len(line) < 3:
        return True
    return bool(re.match(
        r"^(Page\s+\d+|CLAT\s+PG|Official\s+Question|Detailed\s+Solutions?|"
        r"CONSORTIUM|National\s+Law|Question\s+Paper|\*+|_+|-+|©)$",
        line, re.IGNORECASE
    ))

def is_option_line(s: str, letter: str):
    return bool(re.match(rf"^\(?{letter}\)?[\.\:\-\)]\s+", s, flags=re.IGNORECASE))

def strip_option_prefix(s: str, letter: str):
    return re.sub(rf"^\(?{letter}\)?[\.\:\-\)]\s+", "", s, flags=re.IGNORECASE).strip()

def is_question_start(s: str):
    return bool(re.match(r"^\d{1,3}[\)\.\:\-]\s+", s.strip()))

def strip_question_prefix(s: str):
    return re.sub(r"^\d{1,3}[\)\.\:\-]?\s*", "", s, flags=re.IGNORECASE).strip()

def is_passage_marker(line: str):
    s = line.strip()
    return bool(re.match(
        r"^(Passage\s+)?(I{1,3}|IV|V|VI{0,3}|IX|X{1,3}|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX|XXI|XXII)\.?\s*$",
        s, re.IGNORECASE
    ))

def is_passage_opener(line: str):
    s = line.strip().lower()
    if s.startswith("the supreme court of india") or s.startswith("the high court"):
        return True
    if len(line) > 100 and not is_question_start(line):
        hits = sum(kw in s for kw in PASSAGE_KEYWORDS)
        return hits >= 2
    return False

def clean_passage_text(txt: str) -> str:
    """Remove solution artifacts from passage text"""
    # Remove leading "CORRECT OPTION: X" or "Answer: X"
    txt = re.sub(
        r"^\s*(CORRECT\s*OPTION|Answer|Ans\.?)\s*[:\-]?\s*\(?[ABCD]\)?\s*",
        "", txt, flags=re.IGNORECASE
    )
    return txt.strip()

def is_potential_passage_block(lines, start, max_lookahead=40):
    """Probe forward to capture prose passage ending before a numbered question"""
    n = len(lines)
    j = start
    buf = []
    saw_content = False
    
    while j < n and j < start + max_lookahead:
        s = lines[j].strip()
        
        # Stop at question start
        if is_question_start(s):
            break
        
        # Stop at another passage marker
        if is_passage_marker(s) and j != start:
            break
        
        # Stop if options appear (means this isn't a passage)
        if any(is_option_line(s, L) for L in LETTERS):
            return None
        
        if not is_noise(s) and len(s) > 2:
            buf.append(s)
            saw_content = True
        
        j += 1
    
    if not saw_content:
        return None
    
    text = " ".join(buf).strip()
    text = clean_passage_text(text)  # Clean solution artifacts
    
    hits = sum(kw in text.lower() for kw in PASSAGE_KEYWORDS)
    
    # Valid passage: >200 chars with legal keywords
    if len(text) >= 200 and hits >= 2:
        return {"text": text, "next_i": j}
    
    return None

def find_answer_after_options(lines, start, max_lines=8, prose_len=120):
    """
    Scan only a few short lines after the 4 options.
    Stop if we hit a new question, passage marker, or long prose.
    """
    patterns = [
        r"(?:Correct\s*Option|CORRECT\s*OPTION|Answer|Ans\.?)\s*[:\-]?\s*\(?([ABCD])\)?",
        r"(?:correct\s+answer\s+is)\s*[:\-]?\s*\(?([ABCD])\)?",
    ]
    
    nonempty = 0
    k = start
    n = len(lines)
    
    while k < n and nonempty < max_lines:
        s = (lines[k] or "").strip()
        k += 1
        
        if not s:
            continue
        
        # Hard boundaries - stop searching
        if is_question_start(s) or is_passage_marker(s) or is_passage_opener(s):
            return None
        
        # If long prose starts (likely passage/explanation), stop
        if len(s) > prose_len and not s.lower().startswith("explanation"):
            return None
        
        # Try to match answer patterns
        for pat in patterns:
            m = re.search(pat, s, re.IGNORECASE)
            if m:
                return LETTERS.index(m.group(1).upper())
        
        nonempty += 1
    
    return None

def classify_subject(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for subject, keywords in SUBJECT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[subject] = score
    if scores:
        return max(scores, key=scores.get)
    return "Other Laws"

def parse_questions_with_passages(text: str):
    """
    Robustly parse CLAT-style PDFs:
    - Detect Roman numeral passages or 'Passage I/II'
    - Detect 'Supreme Court...' legal narration passages
    - Detect generic long legal prose immediately preceding questions
    - Attach each passage to next 3–8 questions, then reset
    - Clean solution artifacts from passages
    """
    lines = normalize_lines(text)
    results = []
    i = 0
    n = len(lines)

    current_passage = None
    passage_number = None
    questions_after_passage = 0
    PASSAGE_MIN_Q = 3
    PASSAGE_MAX_Q = 8

    while i < n:
        line = lines[i].strip()

        # Skip boilerplate noise
        if is_noise(line):
            i += 1
            continue

        # 1) Explicit Roman numeral/Passage marker
        if is_passage_marker(line):
            passage_number = line.strip()
            i += 1
            passage_lines = []
            questions_after_passage = 0
            
            while i < n:
                s = lines[i].strip()
                if is_passage_marker(s) or is_question_start(s):
                    break
                if not is_noise(s):
                    passage_lines.append(s)
                i += 1
            
            text_block = " ".join(passage_lines).strip()
            text_block = clean_passage_text(text_block)
            current_passage = text_block if len(text_block) >= 120 else None
            continue

        # 2) Recognizable legal opener
        if is_passage_opener(line):
            probe = is_potential_passage_block(lines, i, max_lookahead=50)
            if probe:
                current_passage = probe["text"]
                passage_number = None
                questions_after_passage = 0
                i = probe["next_i"]
                continue

        # 3) Generic prose block immediately before questions
        probe = is_potential_passage_block(lines, i, max_lookahead=40)
        if probe and not is_question_start(line):
            current_passage = probe["text"]
            passage_number = None
            questions_after_passage = 0
            i = probe["next_i"]
            continue

        # 4) Question parsing
        if is_question_start(line):
            stem_parts = [strip_question_prefix(line)]
            j = i + 1

            # Accumulate multi-line question stem
            while j < n:
                nxt = lines[j].strip()
                
                if (any(is_option_line(nxt, L) for L in LETTERS) or 
                    is_question_start(nxt) or 
                    is_passage_marker(nxt) or 
                    is_passage_opener(nxt)):
                    break
                
                if not is_noise(nxt):
                    stem_parts.append(nxt)
                
                j += 1

            # Collect 4 options
            options = []
            for L in LETTERS:
                if j < n and is_option_line(lines[j], L):
                    options.append(strip_option_prefix(lines[j], L))
                    j += 1
                else:
                    break

            # Only process if we have all 4 options
            if len(options) == 4:
                q_text = " ".join(stem_parts).strip()
                
                # Use strict answer finder
                ans_idx = find_answer_after_options(lines, j, max_lines=8, prose_len=120)

                # Attach passage for next 3–8 questions
                if current_passage:
                    questions_after_passage += 1
                    use_passage = current_passage
                    use_passage_number = passage_number
                    
                    # Reset after max questions to avoid leakage
                    if questions_after_passage >= PASSAGE_MAX_Q:
                        current_passage = None
                        passage_number = None
                        questions_after_passage = 0
                else:
                    use_passage = None
                    use_passage_number = None

                combined = (use_passage or "") + " " + q_text
                
                q = {
                    "passage": use_passage,
                    "passage_number": use_passage_number,
                    "question": q_text,
                    "options": options,
                    "answer_index": ans_idx,
                    "subject": classify_subject(combined)
                }
                
                if len(q["question"]) > 10:
                    results.append(q)

            i = max(i + 1, j)
        else:
            i += 1

    return results

# ---------------- PDF Extraction ----------------
@st.cache_data(show_spinner=False, ttl=3600)
def extract_pool_from_folder(folder: str):
    pdf_paths = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    pool = []
    
    for p in pdf_paths:
        try:
            with pdfplumber.open(p) as pdf:
                text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        except Exception:
            continue
        
        if not text:
            continue
        
        qs = parse_questions_with_passages(text)
        for q in qs:
            q["source"] = os.path.basename(p)
        pool.extend(qs)

    # Deduplicate
    seen = set()
    uniq = []
    for q in pool:
        key = (q["question"][:160].lower(), q["options"][0][:80].lower() if q["options"] else "")
        if key not in seen:
            seen.add(key)
            uniq.append(q)
    
    return uniq

def prepare_exam_set_subjectwise(pool, total_count=TARGET_COUNT):
    by_subject = defaultdict(list)
    for q in pool:
        by_subject[q["subject"]].append(q)

    selected = []
    for subject, target in SUBJECT_DISTRIBUTION.items():
        available = by_subject.get(subject, [])
        if len(available) >= target:
            selected.extend(random.sample(available, target))
        else:
            selected.extend(available)

    if len(selected) < total_count:
        remaining = [q for q in pool if q not in selected]
        need = total_count - len(selected)
        if remaining:
            selected.extend(random.sample(remaining, min(need, len(remaining))))
    
    return selected[:total_count]

# ---------------- Session State ----------------
def ensure_state():
    st.session_state.setdefault("stage", "idle")
    st.session_state.setdefault("questions", [])
    st.session_state.setdefault("answers", {})
    st.session_state.setdefault("current", 0)
    st.session_state.setdefault("end_ts", None)
    st.session_state.setdefault("duration_min", DEFAULT_DURATION_MIN)
    st.session_state.setdefault("show_palette", False)

ensure_state()

# ---------------- Main UI ----------------
st.title("⚖️ CLAT PG Mock Test 2026")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Test Settings")
    folder = st.text_input("📁 PDF Folder", value=FOLDER)
    duration_min = st.number_input("⏱️ Duration (minutes)", 10, 240, st.session_state["duration_min"], 5)
    
    st.markdown("---")
    st.markdown("### 🎯 Actions")
    prep_clicked = st.button(
        "🔄 Prepare Subject-Wise Test", 
        use_container_width=True,
        disabled=st.session_state["stage"] in ["started", "instructions"]
    )
    reset_clicked = st.button("🔁 Reset", use_container_width=True)

if reset_clicked:
    st.session_state.update(
        stage="idle", questions=[], answers={}, current=0, 
        end_ts=None, duration_min=DEFAULT_DURATION_MIN, show_palette=False
    )
    st.rerun()

if prep_clicked:
    st.session_state["duration_min"] = int(duration_min)
    with st.status("🔍 Preparing subject-wise test with passages...", expanded=True) as status:
        pool = extract_pool_from_folder(folder)
        subset = prepare_exam_set_subjectwise(pool, total_count=TARGET_COUNT)
        
        if not subset:
            st.error("❌ No questions found")
            status.update(label="❌ Failed", state="error")
        else:
            st.session_state["questions"] = subset
            st.session_state["answers"] = {}
            st.session_state["current"] = 0
            st.session_state["stage"] = "instructions"
            
            by_subj = defaultdict(int)
            for q in subset:
                by_subj[q["subject"]] += 1
            
            breakdown = ", ".join([f"{s}: {c}" for s, c in sorted(by_subj.items())])
            with_passages = sum(1 for q in subset if q.get("passage"))
            
            st.success(f"✅ Loaded {len(subset)} questions")
            st.info(f"📖 Questions with passages: {with_passages}")
            st.info(f"📊 {breakdown}")
            status.update(label=f"✅ Ready: {len(subset)} questions", state="complete")
    st.rerun()

# Metrics
def remaining_seconds():
    if not st.session_state.get("end_ts"):
        return None
    return max(0, int(st.session_state["end_ts"] - time.time()))

if st.session_state["stage"] == "started":
    rem = remaining_seconds()
    if rem is not None and rem <= 0:
        st.warning("⏰ Time's up! Auto-submitting...")
        st.session_state["stage"] = "submitted"
        st.rerun()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Status", st.session_state["stage"].title())
with col2:
    st.metric("📝 Questions", f"{len(st.session_state.get('questions', []))}")
with col3:
    if st.session_state["stage"] == "started":
        rem = remaining_seconds()
        if rem is not None:
            mins, secs = divmod(rem, 60)
            timer_icon = "🔴" if rem < 600 else "⏱️"
            st.metric(f"{timer_icon} Time Left", f"{mins:02d}:{secs:02d}")
        else:
            st.metric("⏱️ Time Left", "120:00")
    else:
        st.metric("⏱️ Time Left", "--:--")
with col4:
    attempted = sum(1 for v in st.session_state["answers"].values() if v is not None)
    st.metric("✓ Done", f"{attempted}")

st.markdown("---")

# Stages
if st.session_state["stage"] == "idle":
    st.info("👋 Click **'Prepare Subject-Wise Test'** to load a test with passage-based questions")

elif st.session_state["stage"] == "instructions":
    st.markdown('<div class="instructions-panel">', unsafe_allow_html=True)
    st.markdown(CLAT_INSTRUCTIONS)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶️ Start Test Now", type="primary", use_container_width=True):
            st.session_state["end_ts"] = (
                datetime.now() + timedelta(minutes=st.session_state["duration_min"])
            ).timestamp()
            st.session_state["stage"] = "started"
            st.rerun()

elif st.session_state["stage"] == "started":
    qs = st.session_state["questions"]
    idx = st.session_state["current"]
    q = qs[idx]

    st.markdown(f'<div class="subject-badge">📚 {q["subject"]}</div>', unsafe_allow_html=True)

    # Show passage if present
    if q.get("passage") and len(q["passage"]) > 80:
        title = f"📖 {q.get('passage_number', 'Passage')}"
        st.markdown(f'''
        <div class="passage-box">
            <h4>{title}</h4>
            <div>{q["passage"]}</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="question-box">
        <span class="question-number">Question {idx+1} of {len(qs)}</span>
        <div class="question-text">{q["question"]}</div>
    </div>
    """, unsafe_allow_html=True)

    opts = [f"{LETTERS[i]}) {txt}" for i, txt in enumerate(q["options"])]
    saved = st.session_state["answers"].get(idx, None)
    default = 0 if saved is None else (saved + 1)

    choice = st.radio(
        "Select your answer:",
        ["⊗ Skip"] + opts,
        index=default,
        key=f"q_{idx}"
    )
    
    st.session_state["answers"][idx] = None if choice == "⊗ Skip" else LETTERS.index(choice[0])

    st.markdown("---")
    
    nav = st.columns([2, 2, 2])
    with nav[0]:
        if st.button("⬅️ Previous", disabled=idx == 0, use_container_width=True):
            st.session_state["current"] = idx - 1
            st.rerun()
    with nav[1]:
        if st.button("🔢 Subject Palette", use_container_width=True):
            st.session_state["show_palette"] = not st.session_state.get("show_palette", False)
            st.rerun()
    with nav[2]:
        if st.button("Next ➡️", disabled=idx >= len(qs) - 1, use_container_width=True):
            st.session_state["current"] = idx + 1
            st.rerun()
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✓ Submit Test", type="primary", use_container_width=True):
            st.session_state["stage"] = "submitted"
            st.rerun()

    if st.session_state.get("show_palette", False):
        st.markdown("### 🔢 Question Palette (Subject-wise)")
        
        by_subject = defaultdict(list)
        for i, qx in enumerate(qs):
            by_subject[qx["subject"]].append(i)
        
        for subject in sorted(by_subject.keys()):
            indices = by_subject[subject]
            st.markdown(
                f'<div class="subject-header">{subject} ({len(indices)} questions)</div>',
                unsafe_allow_html=True
            )
            
            per_row = 10
            rows = (len(indices) + per_row - 1) // per_row
            for r in range(rows):
                cols = st.columns(per_row)
                for c in range(per_row):
                    pos = r * per_row + c
                    if pos >= len(indices):
                        continue
                    k = indices[pos]
                    attempted = st.session_state["answers"].get(k, None) is not None
                    is_current = (k == idx)
                    label = f"{'✓' if attempted else ''}{k+1}{'←' if is_current else ''}"
                    if cols[c].button(label, key=f"j_{k}", use_container_width=True):
                        st.session_state["current"] = k
                        st.rerun()

elif st.session_state["stage"] == "submitted":
    qs = st.session_state["questions"]
    ans = st.session_state["answers"]
    correct = wrong = unattempt = unscored = 0
    score = 0.0

    subject_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "unattempt": 0, "total": 0})
    
    for i, q in enumerate(qs):
        chosen = ans.get(i, None)
        true_idx = q.get("answer_index", None)
        subj = q["subject"]
        subject_stats[subj]["total"] += 1
        
        if chosen is None:
            unattempt += 1
            subject_stats[subj]["unattempt"] += 1
        elif true_idx is None:
            unscored += 1
        elif chosen == true_idx:
            correct += 1
            score += MARKS_CORRECT
            subject_stats[subj]["correct"] += 1
        else:
            wrong += 1
            score -= MARKS_NEGATIVE
            subject_stats[subj]["wrong"] += 1

    st.balloons()
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 3rem; border-radius: 16px; color: white; text-align: center;">
        <h1 style="margin: 0; font-size: 56px;">{score:.2f}</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 24px;">Final Score</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Overall Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Correct", correct)
    with col2:
        st.metric("❌ Wrong", wrong)
    with col3:
        st.metric("⊗ Skipped", unattempt)
    with col4:
        st.metric("⚠️ Unscored", unscored)

    st.markdown("### 📚 Subject-wise Performance")
    for subj in sorted(subject_stats.keys()):
        stats = subject_stats[subj]
        with st.expander(f"{subj} - {stats['correct']}/{stats['total']} correct"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Correct", stats["correct"])
            with c2:
                st.metric("Wrong", stats["wrong"])
            with c3:
                st.metric("Skipped", stats["unattempt"])
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("🔄 Take Another Test", type="primary", use_container_width=True):
            st.session_state.update(
                stage="idle", questions=[], answers={}, 
                current=0, end_ts=None, show_palette=False
            )
            st.rerun()

