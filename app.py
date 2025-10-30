# app.py — CLAT PG Mock Test - User provides OpenAI key in UI
import streamlit as st
import pdfplumber
import re
import glob
import os
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
from openai import OpenAI

st.set_page_config(page_title="CLAT PG Mock Test", page_icon="⚖️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main { background-color: #0a0e27; }
    .block-container { padding: 1rem !important; max-width: 1200px !important; }
    .stButton button { width: 100%; border-radius: 10px; padding: 12px 20px; font-weight: 600; font-size: 16px; border: none; }
    .stButton button[kind="primary"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .stRadio > label { color: #e5e7eb !important; font-weight: 600; font-size: 16px; }
    .stRadio [data-baseweb="radio"] { background-color: #1a1f3a !important; padding: 16px !important; border: 2px solid #374151 !important; border-radius: 10px !important; margin-bottom: 8px !important; }
    .stRadio [data-baseweb="radio"]:hover { border-color: #667eea !important; background-color: #0d1128 !important; }
    .stRadio label div { color: #e5e7eb !important; font-size: 16px !important; line-height: 1.6 !important; font-weight: 500 !important; }
    [data-testid="stMetric"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 16px; border-radius: 12px; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: white !important; font-weight: 700; }
    .passage-box { background-color: #fffbf0 !important; padding: 26px; border-radius: 12px; border: 3px solid #f59e0b; border-left: 6px solid #d97706; margin: 20px 0; font-size: 16px; line-height: 1.85; color: #1f2937; max-height: 550px; overflow-y: auto; }
    .passage-box h4 { color: #c2410c !important; margin: 0 0 16px 0; font-size: 19px; font-weight: 800; border-bottom: 2px solid #fdba74; padding-bottom: 8px; }
    .question-box { background-color: #141829 !important; padding: 28px; border-radius: 12px; margin: 20px 0; border-left: 5px solid #667eea; }
    .question-number { color: #a5b4fc !important; font-weight: 700; font-size: 22px; margin-bottom: 18px; display: block; }
    .question-text { color: #f3f4f6 !important; font-size: 19px !important; line-height: 1.8 !important; font-weight: 500 !important; }
    .subject-badge { display: inline-block; background: #1e293b; color: #e0e7ff; padding: 7px 15px; border-radius: 20px; font-size: 14px; font-weight: 600; border: 1px solid #475569; margin-bottom: 12px; }
    .subject-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 20px; border-radius: 8px; margin: 20px 0 10px 0; font-weight: 700; }
    .instructions-panel { background: #141829; padding: 28px; border-radius: 12px; border: 2px solid #334155; margin: 20px 0; }
    .instructions-panel h3, .instructions-panel p, .instructions-panel li { color: #e5e7eb !important; line-height: 1.8; }
    #MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

FOLDER = "questions"
LETTERS = ["A", "B", "C", "D"]
DEFAULT_DURATION_MIN = 120
MARKS_CORRECT = 1.0
MARKS_NEGATIVE = 0.25
MODEL = "gpt-4o-mini"

SUBJECT_KEYWORDS = {
    "Constitutional Law": ["constitution","fundamental rights","dpsp","article","amendment","judicial review","preamble","citizenship"],
    "Jurisprudence": ["jurisprudence","legal theory","natural law","positivism","austin","kelsen","hart","dworkin"],
    "Contract Law": ["contract","agreement","offer","acceptance","consideration","breach","damages"],
    "Criminal Law": ["ipc","criminal","murder","theft","cheating","culpable homicide","penal code"],
    "Tort Law": ["tort","negligence","nuisance","defamation","trespass","strict liability"],
    "Property Law": ["property","transfer","easement","mortgage","lease","possession","ownership"],
    "Administrative Law": ["administrative","delegated legislation","natural justice","tribunal","ombudsman"],
    "Company Law": ["company","director","shareholder","corporate","board","companies act"],
    "Family Law": ["marriage","divorce","maintenance","custody","adoption","hindu marriage"],
    "International Law": ["international","treaty","sovereignty","united nations","icc","geneva convention"],
}

CLAT_INSTRUCTIONS = """
### 📋 CLAT PG Mock Test - Instructions

1. **Duration:** Configurable (default 2 hours)
2. **Questions:** All questions from PDFs in exact sequence
3. **Marking Scheme:** +1 for correct, -0.25 for wrong, 0 for unattempted
4. **Instant Feedback:** After you answer, see the correct option (from PDF key) and an AI-generated explanation

**Good Luck! 🎓**
"""

PASSAGE_KEYWORDS = ["article","section","constitution","supreme court","high court","judgment","held that","observed","scc","air","v.","rights","liberty","bail","property","criminal","jurisdiction","statute","contract","doctrine","estoppel","government","equity"]

def normalize_lines(text):
    text = text.replace("\r","\n")
    text = re.sub(r"[ \t]+"," ",text)
    text = re.sub(r"-\n","",text)
    lines = [ln.strip() for ln in text.split("\n")]
    return [ln for ln in lines if ln]

def is_noise(line):
    if not line or len(line)<3: return True
    return bool(re.match(r"^(Page\s+\d+|CLAT\s+PG|Official\s+Question|Detailed\s+Solutions?|CONSORTIUM|National\s+Law|Question\s+Paper|\*+|_+|-+|©)$",line,re.I))

def is_option_line(s,letter):
    return bool(re.match(rf"^\(?{letter}\)?[\.\:\-\)]\s+",s,re.I))

def strip_option_prefix(s,letter):
    return re.sub(rf"^\(?{letter}\)?[\.\:\-\)]\s+","",s,re.I).strip()

def is_question_start(s):
    return bool(re.match(r"^\d{1,3}[\)\.\:\-]\s+",s.strip()))

def strip_question_prefix(s):
    return re.sub(r"^\d{1,3}[\)\.\:\-]?\s*","",s,re.I).strip()

def is_passage_marker(line):
    s=line.strip()
    return bool(re.match(r"^(Passage\s+)?(I{1,3}|IV|V|VI{0,3}|IX|X{1,3}|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX|XXI|XXII)\.?\s*$",s,re.I))

def is_passage_opener(line):
    s=line.strip().lower()
    openers=["the supreme court of india","the high court","we may note that","the doctrine of","it is well settled","it is settled law"]
    for opener in openers:
        if s.startswith(opener): return True
    if len(line)>100 and not is_question_start(line):
        hits=sum(kw in s for kw in PASSAGE_KEYWORDS)
        return hits>=3
    return False

def clean_passage_text(txt):
    txt=re.sub(r"^\s*(CORRECT\s*OPTION|Answer|Ans\.?)\s*[:\-]?\s*\(?[ABCD]\)?\s*","",txt,re.I)
    return txt.strip()

def contains_cid_artifacts(text):
    return bool(re.search(r"\(cid:\d+\)",text))

def strip_cid_artifacts(text):
    return re.sub(r"\(cid:\d+\)","",text)

def is_potential_passage_block(lines, start, max_lookahead=120):
    n = len(lines); j = start; buf = []; saw_content = False; consecutive_short = 0; consecutive_empty = 0
    while j < n and j < start + max_lookahead:
        s = lines[j].strip()
        if is_question_start(s) and j > start: break
        if is_passage_marker(s) and j != start: break
        if any(is_option_line(s, L) for L in LETTERS): break
        if is_noise(s): j += 1; continue
        if not s:
            consecutive_empty += 1
            if consecutive_empty > 5 and saw_content: break
            j += 1; continue
        else: consecutive_empty = 0
        if len(s) > 2: buf.append(s); saw_content = True; consecutive_short = 0
        elif len(s) > 0: consecutive_short += 1; 
        if consecutive_short > 5: break
        j += 1
    if not saw_content: return None
    text = " ".join(buf).strip(); text = clean_passage_text(text)
    hits = sum(kw in text.lower() for kw in PASSAGE_KEYWORDS)
    if len(text) >= 150 and hits >= 1: return {"text": text, "next_i": j}
    return None

def extract_global_answer_keys(text):
    keys = {}; lines = normalize_lines(text); in_key_section = False
    for line in lines:
        if re.match(r"^(Answer\s+Key|ANSWER\s+KEY|Correct\s+Answers)[\s:]*$", line, re.I): in_key_section = True; continue
        if in_key_section and (is_passage_marker(line) or re.match(r"^(Passage|Question|Section)\s+", line, re.I)): in_key_section = False
        if in_key_section:
            matches = re.findall(r"(?:Q\.?|Question\s+)?(\d+)[\.\:\)]\s*([ABCD])", line, re.I)
            for qnum_str, letter in matches:
                qnum = int(qnum_str)
                if letter.upper() in LETTERS: keys[qnum] = LETTERS.index(letter.upper())
    return keys

def debug_answer_search(lines, start, max_debug_lines=30):
    debug_text = []; k = start; n = len(lines); count = 0
    while k < n and count < max_debug_lines:
        s = (lines[k] or "").strip()
        if is_question_start(s) or is_passage_marker(s): break
        if s: debug_text.append(s)
        k += 1; count += 1
    return "\n".join(debug_text)

def find_answer_after_options(lines, start, max_lines=80, prose_len=500):
    patterns = [
        r"(?:Correct\s*Option|CORRECT\s*OPTION|Answer|Ans\.?|Correct\s*Answer)\s*[:\-]?\s*\(?([ABCD])\)?",
        r"(?:correct\s+answer\s+is)\s*[:\-]?\s*\(?([ABCD])\)?",
        r"^([ABCD])\s*is\s+(?:correct|right|the\s+answer)",
        r"Option\s+\(?([ABCD])\)?\s+is\s+(?:correct|right)",
        r"The\s+correct\s+option\s+is\s+\(?([ABCD])\)?",
        r"Correct:\s*\(?([ABCD])\)?",
        r"Answer\s+Key:\s*\(?([ABCD])\)?",
        r"\bOption\s+([ABCD])\b",
        r"^\(?([ABCD])\)?\s*$",
    ]
    nonempty = 0; k = start; n = len(lines)
    while k < n and nonempty < max_lines:
        s = (lines[k] or "").strip(); k += 1
        if not s: continue
        if is_question_start(s): break
        if is_passage_marker(s): break
        if len(s) > prose_len:
            for pat in patterns:
                m = re.search(pat, s, re.I)
                if m:
                    letter = m.group(1).upper()
                    if letter in LETTERS: return LETTERS.index(letter)
            if not re.match(r"^(Explanation|Solution|Detailed\s+Solution|Answer|Rationale|Correct)", s, re.I): break
        for pat in patterns:
            m = re.search(pat, s, re.I)
            if m:
                letter = m.group(1).upper()
                if letter in LETTERS: return LETTERS.index(letter)
        nonempty += 1
    remaining_block = []; k2 = start; line_count = 0; max_block_lines = max_lines * 2
    while k2 < n and line_count < max_block_lines:
        s = (lines[k2] or "").strip(); k2 += 1; line_count += 1
        if is_question_start(s) or is_passage_marker(s): break
        if s: remaining_block.append(s)
    full_text = " ".join(remaining_block)
    for pat in patterns:
        m = re.search(pat, full_text, re.I)
        if m:
            letter = m.group(1).upper()
            if letter in LETTERS: return LETTERS.index(letter)
    k3 = start; line_limit = 200; lines_checked = 0
    while k3 < n and lines_checked < line_limit:
        s = (lines[k3] or "").strip(); k3 += 1; lines_checked += 1
        if is_question_start(s) or is_passage_marker(s): break
        m = re.search(r"^\(?([ABCD])\)?\.?\s*$", s)
        if m:
            letter = m.group(1).upper()
            return LETTERS.index(letter)
        m = re.match(r"^([ABCD])\.\s+", s)
        if m and len(s) > 10:
            letter = m.group(1).upper()
            return LETTERS.index(letter)
    return None

def classify_subject(text):
    text_lower=text.lower(); scores={}
    for subject,keywords in SUBJECT_KEYWORDS.items():
        score=sum(1 for kw in keywords if kw in text_lower)
        if score>0: scores[subject]=score
    if scores: return max(scores,key=scores.get)
    return "Other Laws"

def parse_questions_with_passages(text):
    lines=normalize_lines(text); results=[]; i=0; n=len(lines)
    current_passage=None; passage_number=None; questions_after_passage=0; PASSAGE_MAX_Q=10
    global_keys = extract_global_answer_keys(text)
    while i<n:
        line=lines[i].strip()
        if is_noise(line): i+=1; continue
        if is_passage_marker(line):
            passage_number=line.strip(); i+=1; passage_lines=[]; questions_after_passage=0
            while i<n:
                s=lines[i].strip()
                if is_passage_marker(s) or is_question_start(s): break
                if not is_noise(s) and len(s)>2: passage_lines.append(s)
                i+=1
            text_block=" ".join(passage_lines).strip(); text_block=clean_passage_text(text_block)
            if contains_cid_artifacts(text_block): text_block=strip_cid_artifacts(text_block)
            current_passage=text_block if len(text_block)>=150 else None
            continue
        if is_passage_opener(line):
            probe=is_potential_passage_block(lines,i,max_lookahead=120)
            if probe:
                txt=probe["text"]
                if contains_cid_artifacts(txt): txt=strip_cid_artifacts(txt)
                current_passage=txt; passage_number=None; questions_after_passage=0; i=probe["next_i"]; continue
        probe=is_potential_passage_block(lines,i,max_lookahead=120)
        if probe and not is_question_start(line):
            txt=probe["text"]
            if contains_cid_artifacts(txt): txt=strip_cid_artifacts(txt)
            current_passage=txt; passage_number=None; questions_after_passage=0; i=probe["next_i"]; continue
        if is_question_start(line):
            stem_parts=[strip_question_prefix(line)]; j=i+1
            original_qnum = None
            qnum_match = re.search(r"^(\d+)", line.strip())
            if qnum_match: original_qnum = int(qnum_match.group(1))
            while j<n:
                nxt=lines[j].strip()
                if any(is_option_line(nxt,L) for L in LETTERS) or is_question_start(nxt) or is_passage_marker(nxt) or is_passage_opener(nxt): break
                if not is_noise(nxt) and len(nxt)>1: stem_parts.append(nxt)
                j+=1
            options=[]
            for L in LETTERS:
                if j<n and is_option_line(lines[j],L): options.append(strip_option_prefix(lines[j],L)); j+=1
                else: break
            if len(options)==4:
                q_text=" ".join(stem_parts).strip()
                debug_text_after_options = debug_answer_search(lines, j, max_debug_lines=30)
                ans_idx=find_answer_after_options(lines,j,max_lines=80,prose_len=500)
                if ans_idx is None and global_keys and original_qnum: ans_idx = global_keys.get(original_qnum)
                if current_passage:
                    questions_after_passage+=1; use_passage=current_passage; use_passage_number=passage_number
                    if questions_after_passage>=PASSAGE_MAX_Q: current_passage=None; passage_number=None; questions_after_passage=0
                else: use_passage=None; use_passage_number=None
                combined=(use_passage or "")+" "+q_text
                q={"passage":use_passage,"passage_number":use_passage_number,"question":q_text,"options":options,"answer_index":ans_idx,"subject":classify_subject(combined),"debug_text":debug_text_after_options if ans_idx is None else None}
                if len(q["question"])>10: results.append(q)
            i=max(i+1,j)
        else: i+=1
    return results

def extract_text_from_pdf_smart(path,use_ocr,ocr_pages_limit=20):
    text_parts=[]
    try:
        with pdfplumber.open(path) as pdf:
            for pno,page in enumerate(pdf.pages):
                t=page.extract_text() or ""
                if contains_cid_artifacts(t) and use_ocr and pno<ocr_pages_limit:
                    try:
                        import pypdfium2 as pdfium
                        import pytesseract
                        pdf_doc=pdfium.PdfDocument(path); pg=pdf_doc[pno]
                        pil_img=pg.render(scale=2.0).to_pil()
                        ocr_txt=pytesseract.image_to_string(pil_img,lang="eng")
                        t=ocr_txt or strip_cid_artifacts(t)
                    except: t=strip_cid_artifacts(t)
                elif contains_cid_artifacts(t): t=strip_cid_artifacts(t)
                text_parts.append(t)
    except: return ""
    return "\n".join(text_parts)

@st.cache_data(show_spinner=False,ttl=3600)
def extract_pool_from_folder(folder,use_ocr):
    pdf_paths=sorted(glob.glob(os.path.join(folder,"*.pdf"))); pool=[]
    for p in pdf_paths:
        text=extract_text_from_pdf_smart(p,use_ocr=use_ocr)
        if not text: continue
        qs=parse_questions_with_passages(text)
        for q in qs: q["source"]=os.path.basename(p)
        pool.extend(qs)
    seen=set(); uniq=[]
    for q in pool:
        key=(q["question"][:160].lower(),q["options"][0][:80].lower() if q["options"] else "")
        if key not in seen: seen.add(key); uniq.append(q)
    return uniq

def prepare_exam_set_sequential(pool):
    return pool

def llm_explain_after_answer(q, api_key):
    if not api_key or not api_key.strip():
        raise RuntimeError("OpenAI API key not provided")
    client = OpenAI(api_key=api_key.strip())
    passage = (q.get("passage") or "").strip()
    stem = q["question"].strip()
    opts = q["options"]
    correct_idx = q.get("answer_index")
    if correct_idx is None: raise RuntimeError("No PDF answer key")
    correct_letter = LETTERS[correct_idx]
    response_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "mcq_explanation",
            "schema": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "why_correct": {"type": "string"},
                    "why_others": {"type":"object","properties":{"A":{"type":"string"},"B":{"type":"string"},"C":{"type":"string"},"D":{"type":"string"}},"required":["A","B","C","D"],"additionalProperties":False}
                },
                "required": ["explanation", "why_correct", "why_others"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    user_prompt = f"""Passage (if any):
{passage}

Question:
{stem}

Options:
A) {opts[0]}
B) {opts[1]}
C) {opts[2]}
D) {opts[3]}

Correct option (from official key): {correct_letter}

Task: In 4–7 sentences, explain the legal reasoning/doctrines/case-law that make the correct option right, then give one-line reasons for why each of the other options is not best. Output JSON only."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a precise CLAT PG law tutor. Respect the provided correct option and explain concisely."},
                {"role": "user", "content": user_prompt}
            ],
            response_format=response_schema,
            temperature=0.7
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return data
    except Exception as e:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise CLAT PG law tutor. Always respond with valid JSON only."},
                    {"role": "user", "content": user_prompt + "\n\nRespond with valid JSON only in this exact format: {\"explanation\": \"...\", \"why_correct\": \"...\", \"why_others\": {\"A\": \"...\", \"B\": \"...\", \"C\": \"...\", \"D\": \"...\"}}"}
                ],
                temperature=0.7
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            return data
        except Exception as e2:
            raise RuntimeError(f"OpenAI API error: {str(e2)}")

def ensure_state():
    st.session_state.setdefault("stage","idle")
    st.session_state.setdefault("questions",[])
    st.session_state.setdefault("answers",{})
    st.session_state.setdefault("current",0)
    st.session_state.setdefault("end_ts",None)
    st.session_state.setdefault("duration_min",DEFAULT_DURATION_MIN)
    st.session_state.setdefault("show_palette",False)
    st.session_state.setdefault("explanations",{})
    st.session_state.setdefault("use_ocr_fallback",True)
    st.session_state.setdefault("show_debug",False)
    st.session_state.setdefault("openai_api_key","")

ensure_state()

st.title("⚖️ CLAT PG Mock Test 2026")

with st.sidebar:
    st.markdown("### ⚙️ Test Settings")
    folder=st.text_input("📁 PDF Folder",value=FOLDER)
    duration_min=st.number_input("⏱️ Duration (minutes)",10,480,st.session_state["duration_min"],5)
    st.markdown("---")
    st.markdown("### 🔐 OpenAI API Key")
    api_key_input = st.text_input(
        "Enter your OpenAI API key",
        type="password",
        value=st.session_state["openai_api_key"],
        help="Get your key from https://platform.openai.com/api-keys",
        placeholder="sk-..."
    )
    if api_key_input != st.session_state["openai_api_key"]:
        st.session_state["openai_api_key"] = api_key_input
    
    if st.session_state["openai_api_key"]:
        masked = st.session_state["openai_api_key"][:4] + "..." + st.session_state["openai_api_key"][-4:]
        st.success(f"✅ Key: {masked}")
    else:
        st.warning("⚠️ No API key - explanations will be disabled")
    
    st.markdown("---")
    st.markdown("### 🧹 PDF Clean-up")
    st.session_state["use_ocr_fallback"]=st.toggle("Use OCR fallback",value=st.session_state["use_ocr_fallback"])
    st.markdown("---")
    st.markdown("### 🐛 Debug")
    st.session_state["show_debug"]=st.toggle("Show debug",value=st.session_state["show_debug"])
    
    if st.session_state["show_debug"] and st.session_state["stage"]=="started":
        qs=st.session_state["questions"]; idx=st.session_state["current"]; q=qs[idx]
        if q.get("passage"):
            passage_len = len(q["passage"])
            st.info(f"Passage: {passage_len} chars")
            if passage_len < 300: st.warning("⚠️ Short")
        if q.get("answer_index") is None:
            st.error("❌ No key")
            if q.get("debug_text"):
                with st.expander("📄 Text"):
                    st.text_area("Lines",q["debug_text"],height=200,key="debug_area")
        else: st.success(f"✅ {LETTERS[q['answer_index']]}")
    
    st.markdown("---")
    st.markdown("### 🎯 Actions")
    prep_clicked=st.button("🔄 Load All Questions",use_container_width=True,disabled=st.session_state["stage"] in ["started","instructions"])
    reset_clicked=st.button("🔁 Reset",use_container_width=True)

if reset_clicked:
    st.session_state.update(stage="idle",questions=[],answers={},current=0,end_ts=None,duration_min=DEFAULT_DURATION_MIN,show_palette=False,explanations={})
    st.rerun()

if prep_clicked:
    st.session_state["duration_min"]=int(duration_min)
    with st.status("🔍 Loading all questions...",expanded=True) as status:
        pool=extract_pool_from_folder(folder,use_ocr=st.session_state["use_ocr_fallback"])
        subset=prepare_exam_set_sequential(pool)
        if not subset:
            st.error("❌ No questions"); status.update(label="❌ Failed",state="error")
        else:
            st.session_state["questions"]=subset; st.session_state["answers"]={}; st.session_state["current"]=0; st.session_state["stage"]="instructions"
            by_subj=defaultdict(int)
            for q in subset: by_subj[q["subject"]]+=1
            breakdown=", ".join([f"{s}: {c}" for s,c in sorted(by_subj.items())])
            with_passages=sum(1 for q in subset if q.get("passage"))
            with_answers=sum(1 for q in subset if q.get("answer_index") is not None)
            st.success(f"✅ {len(subset)} questions")
            st.info(f"📖 Passages: {with_passages}")
            st.info(f"✓ Keys: {with_answers}")
            st.info(f"📊 {breakdown}")
            status.update(label=f"✅ {len(subset)} loaded",state="complete")
    st.rerun()

def remaining_seconds():
    if not st.session_state.get("end_ts"): return None
    return max(0,int(st.session_state["end_ts"]-time.time()))

if st.session_state["stage"]=="started":
    rem=remaining_seconds()
    if rem is not None and rem<=0:
        st.warning("⏰ Time's up!")
        st.session_state["stage"]="submitted"
        st.rerun()

col1,col2,col3,col4=st.columns(4)
with col1: st.metric("📊 Status",st.session_state["stage"].title())
with col2: st.metric("📝 Questions",f"{len(st.session_state.get('questions',[]))}")
with col3:
    if st.session_state["stage"]=="started":
        rem=remaining_seconds()
        if rem is not None:
            mins,secs=divmod(rem,60); timer_icon="🔴" if rem<600 else "⏱️"
            st.metric(f"{timer_icon} Time",f"{mins:02d}:{secs:02d}")
        else: st.metric("⏱️ Time","--:--")
    else: st.metric("⏱️ Time","--:--")
with col4:
    attempted=sum(1 for v in st.session_state["answers"].values() if v is not None)
    st.metric("✓ Done",f"{attempted}")

st.markdown("---")

if st.session_state["stage"]=="idle":
    st.info("👋 Enter your OpenAI API key in sidebar, then click 'Load All Questions'")

elif st.session_state["stage"]=="instructions":
    st.markdown('<div class="instructions-panel">',unsafe_allow_html=True)
    st.markdown(CLAT_INSTRUCTIONS)
    total_q = len(st.session_state.get("questions", []))
    st.info(f"📝 This test contains **{total_q} questions** from all PDFs in exact sequence.")
    st.markdown('</div>',unsafe_allow_html=True)
    col1,col2,col3=st.columns([1,2,1])
    with col2:
        if st.button("▶️ Start Now",type="primary",use_container_width=True):
            st.session_state["end_ts"]=(datetime.now()+timedelta(minutes=st.session_state["duration_min"])).timestamp()
            st.session_state["stage"]="started"
            st.rerun()

elif st.session_state["stage"]=="started":
    qs=st.session_state["questions"]; idx=st.session_state["current"]; q=qs[idx]
    st.markdown(f'<div class="subject-badge">📚 {q["subject"]}</div>',unsafe_allow_html=True)
    if q.get("passage") and len(q["passage"])>80:
        title=f"📖 {q.get('passage_number','Passage')}"
        st.markdown(f'<div class="passage-box"><h4>{title}</h4><p>{q["passage"]}</p></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="question-box"><span class="question-number">Q{idx+1}/{len(qs)}</span><div class="question-text">{q["question"]}</div></div>',unsafe_allow_html=True)
    opts=[f"{LETTERS[i]}) {txt}" for i,txt in enumerate(q["options"])]
    saved=st.session_state["answers"].get(idx,None)
    default=0 if saved is None else (saved+1)
    choice=st.radio("Answer:",["⊗ Skip"]+opts,index=default,key=f"q_{idx}")
    chosen_idx=None if choice=="⊗ Skip" else LETTERS.index(choice[0])
    st.session_state["answers"][idx]=chosen_idx
    true_idx=q.get("answer_index",None)
    if chosen_idx is not None and true_idx is not None:
        if chosen_idx==true_idx:
            st.success(f"✅ Correct! +{MARKS_CORRECT}")
        else:
            st.error(f"❌ Wrong! Correct: **{LETTERS[true_idx]})** {q['options'][true_idx]}")
            st.info(f"*-{MARKS_NEGATIVE}*")
        if idx not in st.session_state["explanations"]:
            if st.session_state["openai_api_key"]:
                with st.spinner("Generating..."):
                    try: st.session_state["explanations"][idx]=llm_explain_after_answer(q, st.session_state["openai_api_key"])
                    except Exception as e: st.session_state["explanations"][idx]={"error":str(e)}
            else: st.session_state["explanations"][idx]={"error":"No API key provided"}
        expl=st.session_state["explanations"].get(idx)
        if isinstance(expl,dict) and "error" in expl:
            st.warning(f"AI: {expl['error']}")
        elif isinstance(expl,dict):
            st.markdown("### 🧠 Explanation")
            st.write(expl.get("explanation",""))
            st.markdown("#### Why correct")
            st.write(expl.get("why_correct",""))
            st.markdown("#### Why others wrong")
            why_others=expl.get("why_others",{})
            for L in LETTERS: st.write(f"{L}) {why_others.get(L,'')}")
    elif chosen_idx is not None and true_idx is None:
        st.warning("⚠️ No PDF key")
        with st.expander("🔧 Override"):
            mcols=st.columns(4)
            for i,L in enumerate(LETTERS):
                with mcols[i]:
                    if st.button(f"{L}",key=f"m_{idx}_{L}",use_container_width=True):
                        st.session_state["questions"][idx]["answer_index"]=i
                        st.rerun()
    st.markdown("---")
    nav=st.columns([2,2,2])
    with nav[0]:
        if st.button("⬅️ Prev",disabled=idx==0,use_container_width=True): st.session_state["current"]=idx-1; st.rerun()
    with nav[1]:
        if st.button("🔢 Palette",use_container_width=True): st.session_state["show_palette"]=not st.session_state.get("show_palette",False); st.rerun()
    with nav[2]:
        if st.button("Next ➡️",disabled=idx>=len(qs)-1,use_container_width=True): st.session_state["current"]=idx+1; st.rerun()
    st.markdown("---")
    col1,col2,col3=st.columns([1,2,1])
    with col2:
        if st.button("✓ Submit",type="primary",use_container_width=True): st.session_state["stage"]="submitted"; st.rerun()
    if st.session_state.get("show_palette",False):
        st.markdown("### 🔢 Palette")
        by_subject=defaultdict(list)
        for i,qx in enumerate(qs): by_subject[qx["subject"]].append(i)
        for subject in sorted(by_subject.keys()):
            indices=by_subject[subject]
            st.markdown(f'<div class="subject-header">{subject} ({len(indices)})</div>',unsafe_allow_html=True)
            per_row=10; rows=(len(indices)+per_row-1)//per_row
            for r in range(rows):
                cols=st.columns(per_row)
                for c in range(per_row):
                    pos=r*per_row+c
                    if pos>=len(indices): continue
                    k=indices[pos]; chosen=st.session_state["answers"].get(k,None); true=qs[k].get("answer_index",None); is_current=(k==idx)
                    if is_current: label=f"🟡{k+1}"; btn_type="primary"
                    elif chosen is not None and true is not None:
                        if chosen==true: label=f"✅{k+1}"; btn_type="secondary"
                        else: label=f"❌{k+1}"; btn_type="secondary"
                    elif chosen is not None: label=f"✓{k+1}"; btn_type="secondary"
                    else: label=f"⊗{k+1}"; btn_type="secondary"
                    if cols[c].button(label,key=f"j_{k}",use_container_width=True,type=btn_type): st.session_state["current"]=k; st.rerun()

elif st.session_state["stage"]=="submitted":
    qs=st.session_state["questions"]; ans=st.session_state["answers"]
    correct=wrong=unattempt=0; score=0.0
    subject_stats=defaultdict(lambda:{"correct":0,"wrong":0,"unattempt":0,"total":0})
    for i,q in enumerate(qs):
        chosen=ans.get(i,None); true_idx=q.get("answer_index",None); subj=q["subject"]
        subject_stats[subj]["total"]+=1
        if chosen is None: unattempt+=1; subject_stats[subj]["unattempt"]+=1
        elif true_idx is None: pass
        elif chosen==true_idx: correct+=1; score+=MARKS_CORRECT; subject_stats[subj]["correct"]+=1
        else: wrong+=1; score-=MARKS_NEGATIVE; subject_stats[subj]["wrong"]+=1
    st.balloons()
    st.markdown(f'<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem; border-radius: 16px; color: white; text-align: center;"><h1 style="margin: 0; font-size: 56px;">{score:.2f}</h1><p style="margin: 0.5rem 0 0 0; font-size: 24px;">Final Score</p></div>',unsafe_allow_html=True)
    st.markdown("### 📊 Performance")
    col1,col2,col3=st.columns(3)
    with col1: st.metric("✅ Correct",correct)
    with col2: st.metric("❌ Wrong",wrong)
    with col3: st.metric("⊗ Skipped",unattempt)
    st.markdown("### 📚 By Subject")
    for subj in sorted(subject_stats.keys()):
        stats=subject_stats[subj]
        with st.expander(f"{subj} - {stats['correct']}/{stats['total']}"):
            c1,c2,c3=st.columns(3)
            with c1: st.metric("Correct",stats["correct"])
            with c2: st.metric("Wrong",stats["wrong"])
            with c3: st.metric("Skipped",stats["unattempt"])
    c1,c2,c3=st.columns([1,2,1])
    with c2:
        if st.button("🔄 New Test",type="primary",use_container_width=True):
            st.session_state.update(stage="idle",questions=[],answers={},current=0,end_ts=None,show_palette=False,explanations={})
            st.rerun()
