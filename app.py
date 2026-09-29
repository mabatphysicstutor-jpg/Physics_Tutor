import os
import json
import uuid
import mimetypes
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import gradio as gr
import pandas as pd
from google import genai
from google.genai import types
from google.cloud import bigquery
from google.oauth2 import service_account

# --- CONFIGURATION ---
CLASSIFY_MODEL = os.getenv("CLASSIFY_MODEL", "gemini-3.1-flash-lite")
SOLVE_MODEL = os.getenv("SOLVE_MODEL", "gemini-3.5-flash")

BQ_PROJECT = os.getenv("BQ_PROJECT", "physics-tutoring-504312")
BQ_DATASET = "bagrut_data"
BQ_TABLE = "questions"
BQ_CONV_TABLE = "conversations"

ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")

# --- LOAD PASSWORDS FROM HUGGING FACE SECRET ---
passwords_json = os.getenv("APP_PASSWORDS", "").strip()

if passwords_json:
    try:
        PASSWORDS = json.loads(passwords_json)
        print(f"[Config] Successfully loaded {len(PASSWORDS)} user passwords from APP_PASSWORDS secret.")
    except Exception as e:
        print(f"[Config Error] Failed to parse APP_PASSWORDS secret: {e}")
        PASSWORDS = {}
else:
    print("[Config Warning] APP_PASSWORDS secret missing or empty. Falling back to default testing accounts.")
    PASSWORDS = {
        "eitanstudent": {"role": "student", "school": "eitan"},
        "eitanteacher": {"role": "teacher", "school": "eitan"},
        "regularstudent": {"role": "student", "school": None},
    }

SCHOOL_HEBREW = {
    "eitan": "איתן",
    None: "ללא שיוך לבית ספר",
}

SCOPE_MY_SCHOOL = "בית הספר שלי"
SCOPE_ALL_SCHOOLS = "כל בתי הספר"

TOPIC_OPTIONS = [
    "Kinematics",
    "Newton's Laws",
    "Work and Energy",
    "Momentum and Impulse",
    "Circular Motion",
    "Gravitation and Kepler's Laws",
    "Simple Harmonic Motion",
    "Torque and Equilibrium",
    "Electrostatics",
    "DC Circuits",
    "Magnetism",
    "Geometric Optics",
    "Waves and Sound",
]

TOPIC_HEBREW = {
    "Kinematics": "קינמטיקה",
    "Newton's Laws": "חוקי ניוטון",
    "Work and Energy": "עבודה ואנרגיה",
    "Momentum and Impulse": "תנע ואיפולס",
    "Circular Motion": "תנועה מעגלית",
    "Gravitation and Kepler's Laws": "כבידה וחוקי קפלר",
    "Simple Harmonic Motion": "תנועה הרמונית פשוטה",
    "Torque and Equilibrium": "מומנט ושיווי משקל",
    "Electrostatics": "אלקטרוסטטיקה",
    "DC Circuits": "מעגלי זרם ישר",
    "Magnetism": "מגנטיות",
    "Geometric Optics": "אופטיקה גאומטרית",
    "Waves and Sound": "גלים וקול",
    "Uncategorized": "לא מסווג",
}

CLASSIFY_PROMPT = """Look at this image. Your ONLY job is to decide: is this a high-school physics problem, diagram, or graph? Do not solve it.
CRITICAL DISQUALIFIERS (Instant "PHYSICS: no"):
- Any image featuring animals (unicorns, horses, cats, dogs, etc.), fantasy creatures, natural landscapes, portraits, or clip-art illustration WITHOUT explicit physics annotations overlaying it.
- Abstract art, general photography, textbook covers, logos, UI screenshots, or handwritten notes that do not contain a specific physics problem statement or diagram.
- Pure math or pure geometry problems without physical units, physical forces, or dynamic motion context.
ISRAELI HIGH-SCHOOL PHYSICS SCOPE:
The problem MUST belong to one of these Bagrut (5-unit) topics:
1. Mechanics: Kinematics (motion graphs, free fall, vectors), Newton's Laws (force/free-body diagrams, tension, friction, inclines), Work & Energy, Momentum & Impulse, Circular Motion (centripetal force, banked curves, vertical loops), Universal Gravitation & Kepler's Laws, Simple Harmonic Motion (pendulums, springs), Torque & Static Equilibrium.
2. Electromagnetism: Electrostatics (Coulomb's Law, field lines, potential), DC Circuits (resistors, internal resistance, EMF, meters), Magnetism (Lorentz force, right-hand rule, induction, magnetic flux).
3. Optics & Waves (if applicable): Geometric optics (refraction, Snell's law, lenses, mirrors) or wave properties (interference, diffraction, sound).
REQUIRED VISUAL SIGNALS (Must have AT LEAST ONE clear signal to say "yes"):
- Explicit physics symbols used as variables: v, a, F, m, T, ω, r, g, α, h, θ, E, P, q, B, I, R, ε, λ.
- SI unit symbols attached to numbers (in English or Hebrew/Arabic contexts): m/s, m/s², N, kg, Hz, J, W, rad/s, cm, °, V, A, Ω, C, T.
- Schematic textbook/exam diagrams: Free-body force diagrams (arrows representing F_g, N, f, T), circuit schematics (battery, resistor symbols), inclined planes, pulleys, curved tracks with labeled points (A, B, C), ray-tracing diagrams for lenses/mirrors.
- Text framing in Hebrew/Arabic/English that presents a formal physics question (e.g., wording like "גוף שמסתו", "כוח", "מהירות", "תנועה מעגלית", "מערכת צירי זמן", "חשב את").
TOPIC CLASSIFICATION (only if PHYSICS is yes):
Choose exactly ONE topic from this fixed list that best matches the problem:
Kinematics, Newton's Laws, Work and Energy, Momentum and Impulse, Circular Motion, Gravitation and Kepler's Laws, Simple Harmonic Motion, Torque and Equilibrium, Electrostatics, DC Circuits, Magnetism, Geometric Optics, Waves and Sound.
FORMAT REQUIREMENT:
Respond in EXACTLY this format, nothing else:
PHYSICS: yes or no
TOPIC: <one topic from the list above, or N/A if PHYSICS is no>
DESCRIPTION: <one sentence IN HEBREW describing what's in the image, so a teacher can tell what the question was about without seeing the picture>
"""

# Text-only counterpart to CLASSIFY_PROMPT, used when a student types a
# question with no image attached. Same fixed topic list and output format,
# but judges the typed Hebrew/English/Arabic text itself instead of looking
# for visual signals.
TEXT_CLASSIFY_PROMPT = """The following is a message a student typed, with no image attached. Your ONLY job is to decide: is this a high-school (Israeli Bagrut, 5-unit) physics question? Do not solve it.
CRITICAL DISQUALIFIERS (Instant "PHYSICS: no"):
- Small talk, greetings, or anything that isn't a physics problem statement or a specific physics concept question.
- Pure math problems with no physical quantities, forces, or motion context.
- Questions about other school subjects.
ISRAELI HIGH-SCHOOL PHYSICS SCOPE:
The question MUST belong to one of these Bagrut (5-unit) topics:
1. Mechanics: Kinematics, Newton's Laws, Work & Energy, Momentum & Impulse, Circular Motion, Universal Gravitation & Kepler's Laws, Simple Harmonic Motion, Torque & Static Equilibrium.
2. Electromagnetism: Electrostatics, DC Circuits, Magnetism.
3. Optics & Waves: Geometric optics or wave/sound properties.
TOPIC CLASSIFICATION (only if PHYSICS is yes):
Choose exactly ONE topic from this fixed list that best matches the question:
Kinematics, Newton's Laws, Work and Energy, Momentum and Impulse, Circular Motion, Gravitation and Kepler's Laws, Simple Harmonic Motion, Torque and Equilibrium, Electrostatics, DC Circuits, Magnetism, Geometric Optics, Waves and Sound.
FORMAT REQUIREMENT:
Respond in EXACTLY this format, nothing else:
PHYSICS: yes or no
TOPIC: <one topic from the list above, or N/A if PHYSICS is no>
DESCRIPTION: <one sentence IN HEBREW summarizing the question, so a teacher can tell what it was about at a glance>

Student's message:
"""

SOCRATIC_SYSTEM_PROMPT = """You are an encouraging, highly expert high school physics private tutor for Israeli students.
Your primary goal is NOT to solve the problem directly, but to guide the student step-by-step using the Socratic method.
CRITICAL FORMATTING & BEHAVIOR RULES:
1. MAX LENGTH: Keep your response strictly under 1 short paragraph (3-4 sentences maximum).
2. ONE STEP AT A TIME: Address only the immediate first step or core concept needed to begin solving the problem, or the immediate next step if the student already responded.
3. SOCRATIC ENDING: Always conclude your response with a single, direct leading question asking the student what they think the next step is, which formula applies, or where specifically they feel stuck.
4. LANGUAGE & TONE: Always respond in clear, natural Hebrew. Maintain a warm, encouraging, and supportive personal tutor tone.
5. NO FULL SOLUTIONS: Never list out the entire math derivation or final answer in a single turn, even across multiple turns of conversation!
6. CONTINUITY: This is an ongoing conversation. Use the student's previous replies to decide what to ask next — don't repeat a question you already asked.
"""

# --- GEMINI CLIENT SETUP ---
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# --- BIGQUERY SETUP ---
try:
    _sa_info = json.loads(os.environ["GCP_SA_KEY"])
    _bq_credentials = service_account.Credentials.from_service_account_info(_sa_info)
    bq_client = bigquery.Client(project=BQ_PROJECT, credentials=_bq_credentials)
except Exception as e:
    print(f"[BigQuery] Client init failed, question logging disabled: {e}")
    bq_client = None


# --- HELPER FUNCTIONS ---
def file_to_part(path: str) -> types.Part:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"
    with open(path, "rb") as f:
        data = f.read()
    return types.Part.from_bytes(data=data, mime_type=mime)


def log_question_to_bigquery(topic: str, description: str, school) -> None:
    if bq_client is None:
        return
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    row = {
        "topic": topic,
        "description": description,
        "school": school,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        errors = bq_client.insert_rows_json(table_id, [row])
        if errors:
            print(f"[BigQuery] Insert returned errors: {errors}")
    except Exception as e:
        print(f"[BigQuery] Insert failed: {e}")


def log_conversation_turn(session_id: str, topic: str, role: str, message: str, school) -> None:
    if bq_client is None or not session_id:
        return
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_CONV_TABLE}"
    row = {
        "session_id": session_id,
        "topic": topic,
        "role": role,
        "message": message,
        "school": school,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        errors = bq_client.insert_rows_json(table_id, [row])
        if errors:
            print(f"[BigQuery] Conversation insert returned errors: {errors}")
    except Exception as e:
        print(f"[BigQuery] Conversation insert failed: {e}")


def _parse_classification(raw_text: str) -> dict:
    """Shared PHYSICS/TOPIC/DESCRIPTION line parser for both the image and
    text classifiers below — same output format, different prompt/input."""
    parsed = {"physics": False, "topic": "N/A", "description": ""}
    for line in raw_text.splitlines():
        line = line.strip()
        if line.upper().startswith("PHYSICS:"):
            parsed["physics"] = "yes" in line.lower()
        elif line.upper().startswith("TOPIC:"):
            parsed["topic"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("DESCRIPTION:"):
            parsed["description"] = line.split(":", 1)[1].strip()
    if parsed["topic"] not in TOPIC_OPTIONS:
        parsed["topic"] = "Uncategorized"
    return parsed


def gemini_classify(image_part) -> dict:
    if not client:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")
    result = client.models.generate_content(
        model=CLASSIFY_MODEL,
        contents=[image_part, CLASSIFY_PROMPT],
    )
    return _parse_classification(result.text)


def gemini_classify_text(text: str) -> dict:
    """Same as gemini_classify, but for a student's typed question with no
    image attached — lets a session start from text alone."""
    if not client:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")
    result = client.models.generate_content(
        model=CLASSIFY_MODEL,
        contents=[TEXT_CLASSIFY_PROMPT + text],
    )
    return _parse_classification(result.text)


def start_tutoring_session(topic, opening_content, logged_opening, user_school):
    """Creates a new Gemini chat session, sends the opening turn, logs both
    sides of it to BigQuery, and returns (reply, session_state). Shared by
    the image-upload path and the text-only path in handle_message so a new
    session is created identically either way."""
    chat_session = client.chats.create(
        model=SOLVE_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SOCRATIC_SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=500,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    session_id = str(uuid.uuid4())
    log_conversation_turn(session_id, topic, "user", logged_opening, user_school)
    result = chat_session.send_message(opening_content)
    reply = result.text
    log_conversation_turn(session_id, topic, "assistant", reply, user_school)
    return reply, {"chat": chat_session, "session_id": session_id, "topic": topic}


def get_topic_stats(school_filter=None):
    """Returns a dict of {english_topic: count}, or None on query failure.
    Deliberately has NO ORDER BY — row order from BigQuery is never relied
    upon. Category ordering for display is handled entirely in
    auto_refresh_dashboard so the chart's x-axis order is fixed and
    identical across every scope/refresh, instead of shifting with
    count-based ranking (which was causing bars and labels to drift out of
    sync when switching between school scopes)."""
    if bq_client is None:
        return None

    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    where_clause = "WHERE school = @school" if school_filter else ""
    query = f"""
        SELECT topic, COUNT(*) AS count
        FROM `{table_id}`
        {where_clause}
        GROUP BY topic
    """
    job_config = None
    if school_filter:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("school", "STRING", school_filter)]
        )
    try:
        results = bq_client.query(query, job_config=job_config).result()
        return {r.topic: int(r.count) for r in results}
    except Exception as e:
        print(f"[BigQuery] Stats query failed: {e}")
        return None


def get_top_questions(school_filter=None, limit=5):
    """Returns up to `limit` (description, count) pairs for questions whose
    description text was logged more than once, ordered most-repeated first.
    Grouping is on exact description string — Gemini's classifier phrasing
    is fairly consistent turn-to-turn, so near-duplicate problems tend to
    collapse into the same string in practice."""
    if bq_client is None:
        return []
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    where_clause = "WHERE school = @school" if school_filter else ""
    query = f"""
        SELECT description, COUNT(*) AS count
        FROM `{table_id}`
        {where_clause}
        GROUP BY description
        HAVING COUNT(*) > 1
        ORDER BY count DESC
        LIMIT {limit}
    """
    job_config = None
    if school_filter:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("school", "STRING", school_filter)]
        )
    try:
        results = bq_client.query(query, job_config=job_config).result()
        return [(r.description, int(r.count)) for r in results]
    except Exception as e:
        print(f"[BigQuery] Top questions query failed: {e}")
        return []


def build_summary_markdown(counts, top_questions):
    """Builds the Hebrew summary panel text from already-fetched stats.
    Takes `counts` (from get_topic_stats) and `top_questions` (from
    get_top_questions) rather than querying again, so the side panel always
    reflects the exact same scope/data as the table and chart next to it."""
    if counts is None:
        return "**לא ניתן לטעון נתונים כרגע.**"

    total = sum(counts.values())
    if total == 0:
        return "### 📊 סיכום מהיר\n\n_עדיין לא נרשמו שאלות._"

    top_topic_en = max(counts, key=counts.get)
    top_topic_he = TOPIC_HEBREW.get(top_topic_en, top_topic_en)
    top_topic_count = counts[top_topic_en]

    lines = [
        "### 📊 סיכום מהיר",
        f"**סה\"כ שאלות שנרשמו:** {total}",
        f"**הנושא הנפוץ ביותר:** {top_topic_he} ({top_topic_count})",
        "",
        "**שאלות שחוזרות על עצמן:**",
    ]

    if top_questions:
        for desc, count in top_questions:
            lines.append(f"- {desc} — **{count}** פעמים")
    else:
        lines.append("_אין עדיין שאלות שחוזרות על עצמן._")

    return "\n\n".join(lines)


def get_sessions_list(school_filter=None):
    if bq_client is None:
        return []
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_CONV_TABLE}"
    where_clause = "WHERE school = @school" if school_filter else ""
    query = f"""
        SELECT session_id, ANY_VALUE(topic) AS topic, MIN(created_at) AS started_at
        FROM `{table_id}`
        {where_clause}
        GROUP BY session_id
        ORDER BY started_at DESC
        LIMIT 200
    """
    job_config = None
    if school_filter:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("school", "STRING", school_filter)]
        )
    try:
        results = bq_client.query(query, job_config=job_config).result()
        choices = []
        for r in results:
            topic_he = TOPIC_HEBREW.get(r.topic, r.topic)
            ts = r.started_at.astimezone(ISRAEL_TZ).strftime("%d/%m %H:%M")
            label = f"{ts} - {topic_he}"
            choices.append((label, r.session_id))
        return choices
    except Exception as e:
        print(f"[BigQuery] Sessions query failed: {e}")
        return []


def get_conversation_dialog(session_id):
    if bq_client is None or not session_id:
        return []
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_CONV_TABLE}"
    query = f"""
        SELECT role, message, created_at
        FROM `{table_id}`
        WHERE session_id = @session_id
        ORDER BY created_at ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("session_id", "STRING", session_id)]
    )
    try:
        results = bq_client.query(query, job_config=job_config).result()
        dialog = [{"role": r.role, "content": r.message} for r in results]
        return dialog
    except Exception as e:
        print(f"[BigQuery] Dialog query failed: {e}")
        return []


# --- AUTO-REFRESH DASHBOARD ---
def auto_refresh_dashboard(scope, user_school):
    """
    Explicitly determines the school filter based on scope:
    - If scope is 'כל בתי הספר', explicitly set school_filter to None.
    - If scope is 'בית הספר שלי', set school_filter to the teacher's user_school.
    """
    if scope == SCOPE_ALL_SCHOOLS:
        school_filter = None
    else:
        school_filter = user_school

    counts = get_topic_stats(school_filter)
    top_questions = get_top_questions(school_filter)
    summary_markdown = build_summary_markdown(counts, top_questions)

    if counts is None:
        df = pd.DataFrame({"נושא": [], "מספר שאלות": []})
        upper = 5
        hebrew_order = []
    else:
        # Fixed, deterministic topic order — identical on every render,
        # regardless of scope or which topics currently have zero questions.
        # Topics with no questions yet are zero-filled rather than dropped.
        # Any topic BigQuery returns that isn't in TOPIC_OPTIONS (e.g.
        # "Uncategorized") is appended at the end, in first-seen order.
        ordered_topics = list(TOPIC_OPTIONS)
        for t in counts:
            if t not in ordered_topics:
                ordered_topics.append(t)

        rows = [[TOPIC_HEBREW.get(t, t), counts.get(t, 0)] for t in ordered_topics]
        df = pd.DataFrame(rows, columns=["נושא", "מספר שאלות"])
        df["מספר שאלות"] = pd.to_numeric(df["מספר שאלות"], errors="coerce").fillna(0).astype(int)
        max_count = df["מספר שאלות"].max() if not df.empty else 0
        upper = max(int(max_count * 1.2), 5)
        hebrew_order = df["נושא"].tolist()

    plot_update = gr.BarPlot(
        value=df,
        x="נושא",
        y="מספר שאלות",
        title="התפלגות שאלות לפי נושא",
        y_lim=(0, upper),
        # Pin the x-axis to our fixed order explicitly — without this,
        # Vega-Lite is free to re-sort a nominal axis on its own, which is
        # what let the bars and labels drift out of sync between scopes.
        sort=hebrew_order if hebrew_order else None,
    )

    choices = get_sessions_list(school_filter)
    dropdown_update = gr.update(choices=choices, value=None)
    dialog_update = []

    return df, plot_update, dropdown_update, dialog_update, summary_markdown


# --- LOGIN AND ATOMIC LOAD ---
def check_password_and_load(entered_password, scope):
    clean_password = (entered_password or "").strip()
    entry = PASSWORDS.get(clean_password)

    if entry is None:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "**סיסמה שגויה, נסה/י שוב.**",
            None,
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        )

    school = entry["school"]

    if entry["role"] == "student":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            school,
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        )

    # Teacher login: fetch and return refreshed dashboard components directly
    df, plot_update, dropdown_update, dialog_update, summary_markdown = auto_refresh_dashboard(scope, school)

    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        "",
        school,
        df, plot_update, dropdown_update, dialog_update, summary_markdown
    )


def go_to_login():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), None


# --- CHAT PIPELINE ---
def handle_message(message, history, session_state, user_school):
    history = history or []
    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    clear_value = {"text": "", "files": []}

    if files:
        image_path = files[0]
        history.append({"role": "user", "content": {"path": image_path}})
        if text:
            history.append({"role": "user", "content": text})

        image_part = file_to_part(image_path)

        try:
            classification = gemini_classify(image_part)
        except Exception as e:
            history.append({"role": "assistant", "content": f"שגיאה בסיווג התמונה: {str(e)}"})
            return history, clear_value, session_state

        if not classification["physics"]:
            history.append({
                "role": "assistant",
                "content": "**זו לא נראית שאלת פיזיקה לבגרות.**"
            })
            return history, clear_value, session_state

        topic = classification["topic"]
        log_question_to_bigquery(topic, classification["description"], user_school)

        if not client:
            history.append({"role": "assistant", "content": "שגיאה: GEMINI_API_KEY אינו מוגדר."})
            return history, clear_value, session_state

        opening_message = text if text else "שלום, אני צריך עזרה להתחיל לפתור את השאלה הזו."

        # What gets logged/shown in the teacher's transcript viewer: if the
        # student typed something alongside the image, that text is already
        # the clearest signal of what the question is — show it as-is, no
        # need for Gemini's own description too. Only fall back to the
        # classifier's Hebrew description when the image came in with no
        # text at all (there's nothing else to show in its place).
        logged_opening = text if text else f"📷 {classification['description']}"

        try:
            reply, session_state = start_tutoring_session(
                topic, [image_part, opening_message], logged_opening, user_school
            )
        except Exception as e:
            history.append({"role": "assistant", "content": f"שגיאה בפנייה ל-Gemini: {str(e)}"})
            return history, clear_value, session_state

        history.append({"role": "assistant", "content": reply})
        return history, clear_value, session_state

    if not text:
        return history, clear_value, session_state

    history.append({"role": "user", "content": text})

    if session_state is None:
        # No image attached — start a new session straight from the typed
        # question, using the text classifier instead of the image one.
        try:
            classification = gemini_classify_text(text)
        except Exception as e:
            history.append({"role": "assistant", "content": f"שגיאה בסיווג השאלה: {str(e)}"})
            return history, clear_value, session_state

        if not classification["physics"]:
            history.append({
                "role": "assistant",
                "content": "**זו לא נראית שאלת פיזיקה לבגרות.**"
            })
            return history, clear_value, session_state

        topic = classification["topic"]
        log_question_to_bigquery(topic, classification["description"], user_school)

        if not client:
            history.append({"role": "assistant", "content": "שגיאה: GEMINI_API_KEY אינו מוגדר."})
            return history, clear_value, session_state

        try:
            reply, session_state = start_tutoring_session(topic, text, text, user_school)
        except Exception as e:
            history.append({"role": "assistant", "content": f"שגיאה בפנייה ל-Gemini: {str(e)}"})
            return history, clear_value, session_state

        history.append({"role": "assistant", "content": reply})
        return history, clear_value, session_state

    chat_session = session_state["chat"]
    log_conversation_turn(session_state["session_id"], session_state["topic"], "user", text, user_school)
    try:
        result = chat_session.send_message(text)
        reply = result.text
    except Exception as e:
        reply = f"שגיאה בפנייה ל-Gemini: {str(e)}"

    log_conversation_turn(session_state["session_id"], session_state["topic"], "assistant", reply, user_school)
    history.append({"role": "assistant", "content": reply})
    return history, clear_value, session_state


# --- UI CONSTRUCTION ---
CUSTOM_CSS = """
#main-wrap {
    max-width: 700px !important;
    margin: 40px auto !important;
}
#teacher-wrap {
    max-width: 1000px !important;
}
#summary-panel {
    background: rgba(0, 0, 0, 0.03);
    border-radius: 10px;
    padding: 16px;
}
"""

with gr.Blocks(title="Bagrut Physics Tutor") as demo:

    with gr.Column(elem_id="main-wrap") as main_wrap:

        user_school_state = gr.State(None)

        # --- Page 0: Password Gate ---
        with gr.Column(visible=True) as password_page:
            gr.Markdown("## מורה פרטי לפיזיקה - בגרות", rtl=True)
            gr.Markdown("### נא להזין סיסמה כדי להיכנס", rtl=True)
            password_box = gr.Textbox(
                label="סיסמה",
                type="password",
                rtl=True,
            )
            password_error = gr.Markdown("", rtl=True)
            password_submit_btn = gr.Button("כניסה", variant="primary")

        # --- Page 1: Student Chat Interface ---
        with gr.Column(visible=False) as student_page:
            student_back_btn = gr.Button("⬅ חזרה", size="lg")
            gr.Markdown("## מורה פרטי לפיזיקה - בגרות", rtl=True)

            session_state = gr.State(None)

            chatbot = gr.Chatbot(
                label="Tutor",
                rtl=True,
                sanitize_html=False,
                height=500,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ],
            )

            msg_box = gr.MultimodalTextbox(
                label="",
                placeholder="כתבו שאלה או צרפו תמונה של בעיה...",
                file_types=["image"],
                sources=["upload"],
                rtl=True,
            )

            msg_box.submit(
                fn=handle_message,
                inputs=[msg_box, chatbot, session_state, user_school_state],
                outputs=[chatbot, msg_box, session_state],
            )

        # --- Page 2: Teacher Dashboard ---
        # Widened via #teacher-wrap so the summary side panel has room
        # alongside the table/chart instead of squeezing the 700px chat width.
        with gr.Column(visible=False, elem_id="teacher-wrap") as teacher_page:
            teacher_back_btn = gr.Button("⬅ חזרה", size="lg")
            gr.Markdown("## לוח בקרה למורה - סטטיסיקת שאלות", rtl=True)

            scope_radio = gr.Radio(
                label="טווח הנתונים",
                choices=[SCOPE_MY_SCHOOL, SCOPE_ALL_SCHOOLS],
                value=SCOPE_MY_SCHOOL,
                rtl=True,
            )

            refresh_btn = gr.Button("🔄 רענן נתונים", variant="primary")

            # Summary panel placed as the FIRST column so it renders on the
            # right in the page's left-to-right flex order — the natural
            # starting point for RTL (Hebrew) reading. Swap the order of
            # these two `with gr.Column(...)` blocks to move it to the left.
            with gr.Row():
                with gr.Column(scale=1, elem_id="summary-panel"):
                    summary_panel = gr.Markdown(
                        build_summary_markdown(None, []),
                        rtl=True,
                    )

                with gr.Column(scale=2):
                    stats_table = gr.Dataframe(
                        headers=["נושא", "מספר שאלות"],
                        datatype=["str", "number"],
                        row_count=(0, "dynamic"),
                        column_count=(2, "fixed"),
                        interactive=False,
                    )

                    stats_plot = gr.BarPlot(
                        x="נושא",
                        y="מספר שאלות",
                        title="התפלגות שאלות לפי נושא",
                        y_lim=(0, 10),
                    )

            gr.Markdown("### שיחות תלמידים", rtl=True)

            sessions_dropdown = gr.Dropdown(
                label="בחר/י שיחה לצפייה",
                choices=[],
            )

            dialog_view = gr.Chatbot(
                label="תמלול השיחה",
                rtl=True,
                height=450,
                sanitize_html=False,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ],
            )

            # Auto-refresh whenever the radio toggle changes.
            # NOTE: queue=False was removed here — with it set, rapid toggles fired
            # unserialized concurrent requests, and a slower query could resolve
            # after a faster one, overwriting the dashboard with stale/mismatched
            # data (table and chart could end up showing different scopes).
            # Using the default queue keeps requests serialized in order.
            scope_radio.change(
                fn=auto_refresh_dashboard,
                inputs=[scope_radio, user_school_state],
                outputs=[stats_table, stats_plot, sessions_dropdown, dialog_view, summary_panel],
            )

            # Manual refresh button
            refresh_btn.click(
                fn=auto_refresh_dashboard,
                inputs=[scope_radio, user_school_state],
                outputs=[stats_table, stats_plot, sessions_dropdown, dialog_view, summary_panel],
            )

            sessions_dropdown.change(
                fn=get_conversation_dialog,
                inputs=[sessions_dropdown],
                outputs=[dialog_view],
            )

        # --- Navigation Wiring ---
        password_submit_btn.click(
            fn=check_password_and_load,
            inputs=[password_box, scope_radio],
            outputs=[
                password_page, student_page, teacher_page, 
                password_error, user_school_state,
                stats_table, stats_plot, sessions_dropdown, dialog_view, summary_panel
            ],
        )

        password_box.submit(
            fn=check_password_and_load,
            inputs=[password_box, scope_radio],
            outputs=[
                password_page, student_page, teacher_page, 
                password_error, user_school_state,
                stats_table, stats_plot, sessions_dropdown, dialog_view, summary_panel
            ],
        )

        student_back_btn.click(
            fn=go_to_login, inputs=None, outputs=[password_page, student_page, teacher_page, user_school_state]
        )
        teacher_back_btn.click(
            fn=go_to_login, inputs=None, outputs=[password_page, student_page, teacher_page, user_school_state]
        )

demo.launch(css=CUSTOM_CSS)
