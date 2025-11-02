import streamlit as st
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
2. **Questions:** All questions from JSON files in exact sequence
3. **Marking Scheme:** +1 for correct, -0.25 for wrong, 0 for unattempted
4. **Instant Feedback:** After you answer, see the correct option and an AI-generated explanation



**Good Luck! 🎓**
"""



def classify_subject(text):
    text_lower=text.lower(); scores={}
    for subject,keywords in SUBJECT_KEYWORDS.items():
        score=sum(1 for kw in keywords if kw in text_lower)
        if score>0: scores[subject]=score
    if scores: return max(scores,key=scores.get)
    return "Other Laws"



@st.cache_data(show_spinner=False,ttl=3600)
def extract_pool_from_folder(use_ocr=None):
    """Load questions from JSON files in questions_scripts folder"""
    json_folder = "questions_scripts"
    pool = []

    if not os.path.exists(json_folder):
        st.error(f"❌ Folder '{json_folder}' not found!")
        return pool

    json_files = sorted(glob.glob(os.path.join(json_folder, "*.json")))

    if not json_files:
        st.error(f"❌ No JSON files found in '{json_folder}' folder")
        return pool

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Handle both "passages" and "passages_with_questions" keys
            passages_list = []
            if isinstance(json_data, dict):
                if "passages" in json_data:
                    passages_list = json_data["passages"]
                elif "passages_with_questions" in json_data:
                    passages_list = json_data["passages_with_questions"]

            for passage in passages_list:
                passage_text = passage.get("passage_text", "")
                passage_number = passage.get("passage_number", "")

                if "questions" in passage:
                    for q in passage["questions"]:
                        options_list = []
                        options_dict = q.get("options", {})
                        for letter in LETTERS:
                            if letter in options_dict:
                                options_list.append(options_dict[letter])

                        if len(options_list) == 4:
                            answer_idx = None
                            correct_answer = q.get("correct_answer", "")
                            if correct_answer in LETTERS:
                                answer_idx = LETTERS.index(correct_answer)

                            # Handle both "question_text" and "question" keys
                            question_text = q.get("question_text") or q.get("question", "")
                            combined_text = (passage_text or "") + " " + question_text

                            question_obj = {
                                "passage": passage_text if len(passage_text) > 80 else None,
                                "passage_number": f"Passage {passage_number}" if passage_number else None,
                                "question": question_text,
                                "options": options_list,
                                "answer_index": answer_idx,
                                "subject": classify_subject(combined_text),
                                "source": os.path.basename(json_file),
                                "debug_text": None
                            }

                            if len(question_obj["question"]) > 10:
                                pool.append(question_obj)

        except Exception as e:
            st.warning(f"Error loading {os.path.basename(json_file)}: {str(e)}")
            continue

    seen = set()
    uniq = []
    for q in pool:
        key = (q["question"][:160].lower(), q["options"][0][:80].lower() if q["options"] else "")
        if key not in seen:
            seen.add(key)
            uniq.append(q)

    return uniq



def prepare_exam_set_sequential(pool):
    return pool



def test_openai_key(api_key):
    try:
        test_client = OpenAI(api_key=api_key.strip())
        response = test_client.chat.completions.create(model="gpt-4o-mini",messages=[{"role": "user", "content": "Hi"}],max_tokens=5)
        return True, "Valid key"
    except Exception as e:
        return False, str(e)



def llm_explain_after_answer(q, api_key):
    if not api_key or not api_key.strip(): raise RuntimeError("OpenAI API key not provided")
    client = OpenAI(api_key=api_key.strip())
    passage = (q.get("passage") or "").strip(); stem = q["question"].strip(); opts = q["options"]; correct_idx = q.get("answer_index")
    if correct_idx is None: raise RuntimeError("No answer key")
    correct_letter = LETTERS[correct_idx]
    response_schema = {"type": "json_schema","json_schema": {"name":"mcq_explanation","schema":{"type":"object","properties":{"explanation":{"type":"string"},"why_correct":{"type":"string"},"why_others":{"type":"object","properties":{"A":{"type":"string"},"B":{"type":"string"},"C":{"type":"string"},"D":{"type":"string"}},"required":["A","B","C","D"],"additionalProperties":False}},"required":["explanation","why_correct","why_others"],"additionalProperties":False},"strict":True}}
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
        response = client.chat.completions.create(model=MODEL,messages=[{"role": "system", "content": "You are a precise CLAT PG law tutor. Respect the provided correct option and explain concisely."},{"role": "user", "content": user_prompt}],response_format=response_schema,temperature=0.7)
        content = response.choices[0].message.content; data = json.loads(content); return data
    except Exception as e:
        try:
            response = client.chat.completions.create(model=MODEL,messages=[{"role": "system", "content": "You are a precise CLAT PG law tutor. Always respond with valid JSON only."},{"role": "user", "content": user_prompt + "\n\nRespond with valid JSON only in this exact format: {\"explanation\": \"...\", \"why_correct\": \"...\", \"why_others\": {\"A\": \"...\", \"B\": \"...\", \"C\": \"...\", \"D\": \"...\"}"}],temperature=0.7)
            content = response.choices[0].message.content; data = json.loads(content); return data
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
    st.session_state.setdefault("show_debug",False)
    st.session_state.setdefault("openai_api_key","")



ensure_state()



st.title("⚖️ CLAT PG Mock Test 2026")



with st.sidebar:
    st.markdown("### ⚙️ Test Settings")
    duration_min=st.number_input("⏱️ Duration (minutes)",10,480,st.session_state["duration_min"],5)
    st.markdown("---")
    st.markdown("### 🔐 OpenAI API Key")
    api_key_input = st.text_input("Enter your OpenAI API key",type="password",value=st.session_state["openai_api_key"],help="Get your key from https://platform.openai.com/api-keys",placeholder="sk-proj-...")
    if api_key_input != st.session_state["openai_api_key"]: st.session_state["openai_api_key"] = api_key_input.strip()
    if st.session_state["openai_api_key"]:
        masked = st.session_state["openai_api_key"][:7] + "..." + st.session_state["openai_api_key"][-4:]; st.success(f"✅ Key: {masked}")
        if st.button("🧪 Test API Key", use_container_width=True):
            with st.spinner("Testing..."):
                valid, msg = test_openai_key(st.session_state["openai_api_key"])
                if valid: st.success("✅ API key is valid!")
                else: st.error(f"❌ Invalid key: {msg}")
    else: st.warning("⚠️ No API key - explanations disabled"); st.info("💡 Get key from: https://platform.openai.com/api-keys")
    st.markdown("---")
    st.markdown("### 🐛 Debug")
    st.session_state["show_debug"]=st.toggle("Show debug",value=st.session_state["show_debug"])
    if st.session_state["show_debug"] and st.session_state["stage"]=="started":
        qs=st.session_state["questions"]; idx=st.session_state["current"]; q=qs[idx]
        if q.get("passage"): 
            passage_len = len(q["passage"]); st.info(f"Passage: {passage_len} chars")
            if passage_len < 300: st.warning("⚠️ Short")
        if q.get("answer_index") is None:
            st.error("❌ No key")
        else: st.success(f"✅ {LETTERS[q['answer_index']]}")
    st.markdown("---")
    st.markdown("### 🎯 Actions")
    prep_clicked=st.button("🔄 Load All Questions",use_container_width=True,disabled=st.session_state["stage"] in ["started","instructions"])
    reset_clicked=st.button("🔁 Reset",use_container_width=True)



if reset_clicked: st.session_state.update(stage="idle",questions=[],answers={},current=0,end_ts=None,show_palette=False,explanations={}); st.rerun()



if prep_clicked:
    st.session_state["duration_min"]=int(duration_min)
    with st.status("🔍 Loading all questions...",expanded=True) as status:
        pool=extract_pool_from_folder(); subset=prepare_exam_set_sequential(pool)
        if not subset: st.error("❌ No questions"); status.update(label="❌ Failed",state="error")
        else:
            st.session_state["questions"]=subset; st.session_state["answers"]={}; st.session_state["current"]=0; st.session_state["stage"]="instructions"
            by_subj=defaultdict(int)
            for q in subset: by_subj[q["subject"]]+=1
            breakdown=", ".join([f"{s}: {c}" for s,c in sorted(by_subj.items())]); with_passages=sum(1 for q in subset if q.get("passage")); with_answers=sum(1 for q in subset if q.get("answer_index") is not None)
            st.success(f"✅ {len(subset)} questions"); st.info(f"📖 Passages: {with_passages}"); st.info(f"✓ Keys: {with_answers}"); st.info(f"📊 {breakdown}")
            status.update(label=f"✅ {len(subset)} loaded",state="complete")
    st.rerun()



def remaining_seconds():
    if not st.session_state.get("end_ts"): return None
    return max(0,int(st.session_state["end_ts"]-time.time()))



if st.session_state["stage"]=="started":
    rem=remaining_seconds()
    if rem is not None and rem<=0: st.warning("⏰ Time's up!"); st.session_state["stage"]="submitted"; st.rerun()



col1,col2,col3,col4=st.columns(4)
with col1: st.metric("📊 Status",st.session_state["stage"].title())
with col2: st.metric("📝 Questions",f"{len(st.session_state.get('questions',[]))}")
with col3:
    if st.session_state["stage"]=="started":
        rem=remaining_seconds()
        if rem is not None: mins,secs=divmod(rem,60); timer_icon="🔴" if rem<600 else "⏱️"; st.metric(f"{timer_icon} Time",f"{mins:02d}:{secs:02d}")
        else: st.metric("⏱️ Time","--:--")
    else: st.metric("⏱️ Time","--:--")
with col4: attempted=sum(1 for v in st.session_state["answers"].values() if v is not None); st.metric("✓ Done",f"{attempted}")



st.markdown("---")



if st.session_state["stage"]=="idle": st.info("👋 Click 'Load All Questions' in sidebar to start")



elif st.session_state["stage"]=="instructions":
    st.markdown('<div class="instructions-panel">',unsafe_allow_html=True); st.markdown(CLAT_INSTRUCTIONS); total_q = len(st.session_state.get("questions", []))
    st.info(f"📝 This test contains **{total_q} questions** from all JSON files in exact sequence."); st.markdown('</div>',unsafe_allow_html=True)
    col1,col2,col3=st.columns([1,2,1])
    with col2:
        if st.button("▶️ Start Now",type="primary",use_container_width=True): st.session_state["end_ts"]=(datetime.now()+timedelta(minutes=st.session_state["duration_min"])).timestamp(); st.session_state["stage"]="started"; st.rerun()



elif st.session_state["stage"]=="started":
    qs=st.session_state["questions"]; idx=st.session_state["current"]; q=qs[idx]
    st.markdown(f'<div class="subject-badge">📚 {q["subject"]}</div>',unsafe_allow_html=True)
    if q.get("passage") and len(q["passage"])>80: title=f"📖 {q.get('passage_number','Passage')}"; st.markdown(f'<div class="passage-box"><h4>{title}</h4><p>{q["passage"]}</p></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="question-box"><span class="question-number">Q{idx+1}/{len(qs)}</span><div class="question-text">{q["question"]}</div></div>',unsafe_allow_html=True)
    opts=[f"{LETTERS[i]}) {txt}" for i,txt in enumerate(q["options"])]; saved=st.session_state["answers"].get(idx,None); default=0 if saved is None else (saved+1)
    choice=st.radio("Answer:",["⊗ Skip"]+opts,index=default,key=f"q_{idx}"); chosen_idx=None if choice=="⊗ Skip" else LETTERS.index(choice[0]); st.session_state["answers"][idx]=chosen_idx
    true_idx=q.get("answer_index",None)
    if chosen_idx is not None and true_idx is not None:
        if chosen_idx==true_idx: st.success(f"✅ Correct! +{MARKS_CORRECT}")
        else: st.error(f"❌ Wrong! Correct: **{LETTERS[true_idx]})** {q['options'][true_idx]}"); st.info(f"*-{MARKS_NEGATIVE}*")
        if idx not in st.session_state["explanations"]:
            if st.session_state["openai_api_key"]:
                with st.spinner("Generating..."):
                    try: st.session_state["explanations"][idx]=llm_explain_after_answer(q, st.session_state["openai_api_key"])
                    except Exception as e: st.session_state["explanations"][idx]={"error":str(e)}
            else: st.session_state["explanations"][idx]={"error":"No API key provided"}
        expl=st.session_state["explanations"].get(idx)
        if isinstance(expl,dict) and "error" in expl: st.warning(f"AI: {expl['error']}")
        elif isinstance(expl,dict): 
            st.markdown("### 🧠 Explanation"); st.write(expl.get("explanation","")); st.markdown("#### Why correct"); st.write(expl.get("why_correct","")); st.markdown("#### Why others wrong"); why_others=expl.get("why_others",{})
            for L in LETTERS: st.write(f"{L}) {why_others.get(L,'')}")
    elif chosen_idx is not None and true_idx is None:
        st.warning("⚠️ No answer key for this question")
        with st.expander("🔧 Override"): 
            mcols=st.columns(4)
            for i,L in enumerate(LETTERS):
                with mcols[i]:
                    if st.button(f"{L}",key=f"m_{idx}_{L}",use_container_width=True): st.session_state["questions"][idx]["answer_index"]=i; st.rerun()
    st.markdown("---"); nav=st.columns([2,2,2])
    with nav[0]:
        if st.button("⬅️ Prev",disabled=idx==0,use_container_width=True): st.session_state["current"]=idx-1; st.rerun()
    with nav[1]:
        if st.button("🔢 Palette",use_container_width=True): st.session_state["show_palette"]=not st.session_state.get("show_palette",False); st.rerun()
    with nav[2]:
        if st.button("Next ➡️",disabled=idx>=len(qs)-1,use_container_width=True): st.session_state["current"]=idx+1; st.rerun()
    st.markdown("---"); col1,col2,col3=st.columns([1,2,1])
    with col2:
        if st.button("✓ Submit",type="primary",use_container_width=True): st.session_state["stage"]="submitted"; st.rerun()
    if st.session_state.get("show_palette",False):
        st.markdown("### 🔢 Palette"); by_subject=defaultdict(list)
        for i,qx in enumerate(qs): by_subject[qx["subject"]].append(i)
        for subject in sorted(by_subject.keys()):
            indices=by_subject[subject]; st.markdown(f'<div class="subject-header">{subject} ({len(indices)})</div>',unsafe_allow_html=True); per_row=10; rows=(len(indices)+per_row-1)//per_row
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
    qs=st.session_state["questions"]; ans=st.session_state["answers"]; correct=wrong=unattempt=0; score=0.0
    subject_stats=defaultdict(lambda:{"correct":0,"wrong":0,"unattempt":0,"total":0})
    for i,q in enumerate(qs): 
        chosen=ans.get(i,None); true_idx=q.get("answer_index",None); subj=q["subject"]; subject_stats[subj]["total"]+=1
        if chosen is None: unattempt+=1; subject_stats[subj]["unattempt"]+=1
        elif true_idx is None: pass
        elif chosen==true_idx: correct+=1; score+=MARKS_CORRECT; subject_stats[subj]["correct"]+=1
        else: wrong+=1; score-=MARKS_NEGATIVE; subject_stats[subj]["wrong"]+=1
    st.balloons(); st.markdown(f'<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem; border-radius: 16px; color: white; text-align: center;"><h1 style="margin: 0; font-size: 56px;">{score:.2f}</h1><p style="margin: 0.5rem 0 0 0; font-size: 24px;">Final Score</p></div>',unsafe_allow_html=True)
    st.markdown("### 📊 Performance"); col1,col2,col3=st.columns(3)
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
        if st.button("🔄 New Test",type="primary",use_container_width=True): st.session_state.update(stage="idle",questions=[],answers={},current=0,end_ts=None,show_palette=False,explanations={}); st.rerun()
