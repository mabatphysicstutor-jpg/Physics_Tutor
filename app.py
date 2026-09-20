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
DESCRIPTION: <one sentence describing what's in the image>
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


def gemini_classify(image_part) -> dict:
    if not client:
        raise ValueError("GEMINI_API_KEY environment variable is missing.")
    result = client.models.generate_content(
        model=CLASSIFY_MODEL,
        contents=[image_part, CLASSIFY_PROMPT],
    )
    raw_text = result.text
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

    return df, plot_update, dropdown_update, dialog_update


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
            gr.update(), gr.update(), gr.update(), gr.update()
        )

    school = entry["school"]

    if entry["role"] == "student":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            school,
            gr.update(), gr.update(), gr.update(), gr.update()
        )

    # Teacher login: fetch and return refreshed dashboard components directly
    df, plot_update, dropdown_update, dialog_update = auto_refresh_dashboard(scope, school)

    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        "",
        school,
        df, plot_update, dropdown_update, dialog_update
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
                "content": f"**זו לא נראית שאלת פיזיקה לבגרות.**\n\nתיאור: {classification['description']}"
            })
            return history, clear_value, session_state

        topic = classification["topic"]
        session_id = str(uuid.uuid4())
        log_question_to_bigquery(topic, classification["description"], user_school)

        if not client:
            history.append({"role": "assistant", "content": "שגיאה: GEMINI_API_KEY אינו מוגדר."})
            return history, clear_value, session_state

        chat_session = client.chats.create(
            model=SOLVE_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SOCRATIC_SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=500,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        opening_message = text if text else "שלום, אני צריך עזרה להתחיל לפתור את השאלה הזו."
        log_conversation_turn(session_id, topic, "user", opening_message, user_school)
        try:
            result = chat_session.send_message([image_part, opening_message])
            reply = result.text
        except Exception as e:
            history.append({"role": "assistant", "content": f"שגיאה בפנייה ל-Gemini: {str(e)}"})
            return history, clear_value, session_state

        log_conversation_turn(session_id, topic, "assistant", reply, user_school)
        history.append({"role": "assistant", "content": reply})
        session_state = {"chat": chat_session, "session_id": session_id, "topic": topic}
        return history, clear_value, session_state

    if not text:
        return history, clear_value, session_state

    history.append({"role": "user", "content": text})

    if session_state is None:
        history.append({
            "role": "assistant",
            "content": "כדי להתחיל, צריך להעלות תמונה של שאלת פיזיקה 📎"
        })
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
"""

with gr.Blocks(title="Bagrut Physics Tutor") as demo:

    with gr.Column(elem_id="main-wrap"):

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
        with gr.Column(visible=False) as teacher_page:
            teacher_back_btn = gr.Button("⬅ חזרה", size="lg")
            gr.Markdown("## לוח בקרה למורה - סטטיסיקת שאלות", rtl=True)

            scope_radio = gr.Radio(
                label="טווח הנתונים",
                choices=[SCOPE_MY_SCHOOL, SCOPE_ALL_SCHOOLS],
                value=SCOPE_MY_SCHOOL,
                rtl=True,
            )

            refresh_btn = gr.Button("🔄 רענן נתונים", variant="primary")

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
                outputs=[stats_table, stats_plot, sessions_dropdown, dialog_view],
            )

            # Manual refresh button
            refresh_btn.click(
                fn=auto_refresh_dashboard,
                inputs=[scope_radio, user_school_state],
                outputs=[stats_table, stats_plot, sessions_dropdown, dialog_view],
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
                stats_table, stats_plot, sessions_dropdown, dialog_view
            ],
        )

        password_box.submit(
            fn=check_password_and_load,
            inputs=[password_box, scope_radio],
            outputs=[
                password_page, student_page, teacher_page, 
                password_error, user_school_state,
                stats_table, stats_plot, sessions_dropdown, dialog_view
            ],
        )

        student_back_btn.click(
            fn=go_to_login, inputs=None, outputs=[password_page, student_page, teacher_page, user_school_state]
        )
        teacher_back_btn.click(
            fn=go_to_login, inputs=None, outputs=[password_page, student_page, teacher_page, user_school_state]
        )

demo.launch(css=CUSTOM_CSS)
