import os
import json
import uuid
import mimetypes
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import gradio as gr
from google import genai
from google.genai import types
from google.cloud import bigquery
from google.oauth2 import service_account

# --- CONFIGURATION ---
CLASSIFY_MODEL = "gemini-3.1-flash-lite"
SOLVE_MODEL = "gemini-3.5-flash"

BQ_PROJECT = "physics-tutoring-504312"
BQ_DATASET = "bagrut_data"
BQ_TABLE = "questions"
BQ_CONV_TABLE = "conversations"

APP_PASSWORD = os.environ.get("APP_PASSWORD", "princecanute")
ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")

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

# English topic label (as stored in BigQuery) -> Hebrew label (shown to teachers)
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

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

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


def log_question_to_bigquery(topic: str, description: str) -> None:
    """Insert one row into bagrut_data.questions for the teacher dashboard stats.
    Never raises - a logging failure must not break the student's session."""
    if bq_client is None:
        return
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    row = {
        "topic": topic,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        errors = bq_client.insert_rows_json(table_id, [row])
        if errors:
            print(f"[BigQuery] Insert returned errors: {errors}")
    except Exception as e:
        print(f"[BigQuery] Insert failed: {e}")


def log_conversation_turn(session_id: str, topic: str, role: str, message: str) -> None:
    """Insert one turn (student question or tutor reply) into bagrut_data.conversations.
    Never raises - a logging failure must not break the student's session."""
    if bq_client is None or not session_id:
        return
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_CONV_TABLE}"
    row = {
        "session_id": session_id,
        "topic": topic,
        "role": role,
        "message": message,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        errors = bq_client.insert_rows_json(table_id, [row])
        if errors:
            print(f"[BigQuery] Conversation insert returned errors: {errors}")
    except Exception as e:
        print(f"[BigQuery] Conversation insert failed: {e}")


def gemini_classify(image_part) -> dict:
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
    # Guard against the model returning a topic outside our fixed list
    if parsed["topic"] not in TOPIC_OPTIONS:
        parsed["topic"] = "Uncategorized"
    return parsed


def get_topic_stats():
    """Query BigQuery for per-topic question counts, translated to Hebrew.
    Returns a list of [hebrew_topic, count] rows, sorted descending by count."""
    if bq_client is None:
        return [["שגיאה: אין חיבור ל-BigQuery", 0]]

    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    query = f"""
        SELECT topic, COUNT(*) AS count
        FROM `{table_id}`
        GROUP BY topic
        ORDER BY count DESC
    """
    try:
        results = bq_client.query(query).result()
        rows = [[TOPIC_HEBREW.get(r.topic, r.topic), r.count] for r in results]
        if not rows:
            return [["אין נתונים עדיין", 0]]
        return rows
    except Exception as e:
        print(f"[BigQuery] Stats query failed: {e}")
        return [[f"שגיאה בשליפת נתונים: {str(e)}", 0]]


def get_sessions_list():
    """Query BigQuery for distinct chat sessions, most recent first.
    Returns choices for a gr.Dropdown: list of (label, session_id) tuples."""
    if bq_client is None:
        return []
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_CONV_TABLE}"
    query = f"""
        SELECT session_id, ANY_VALUE(topic) AS topic, MIN(created_at) AS started_at
        FROM `{table_id}`
        GROUP BY session_id
        ORDER BY started_at DESC
        LIMIT 200
    """
    try:
        results = bq_client.query(query).result()
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
    """Fetch every turn for one session_id, ordered by time.
    Returns a list of {"role", "content"} dicts, ready for gr.Chatbot."""
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


def refresh_sessions():
    """Repopulate the sessions dropdown and clear the old dialog view."""
    choices = get_sessions_list()
    return gr.update(choices=choices, value=None), []


# --- CHAT PIPELINE (student side) ---
def handle_message(message, history, session_state):
    """Single entry point for the unified chat box.
    `message` is a dict from MultimodalTextbox: {"text": str, "files": [paths]}.
    `session_state` holds {"chat": genai chat session, "session_id": str, "topic": str}."""
    history = history or []
    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    clear_value = {"text": "", "files": []}

    # --- Case 1: an image was attached -> start a fresh problem ---
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
        log_question_to_bigquery(topic, classification["description"])

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
        log_conversation_turn(session_id, topic, "user", opening_message)
        try:
            result = chat_session.send_message([image_part, opening_message])
            reply = result.text
        except Exception as e:
            history.append({"role": "assistant", "content": f"שגיאה בפנייה ל-Gemini: {str(e)}"})
            return history, clear_value, session_state

        log_conversation_turn(session_id, topic, "assistant", reply)
        history.append({"role": "assistant", "content": reply})
        session_state = {"chat": chat_session, "session_id": session_id, "topic": topic}
        return history, clear_value, session_state

    # --- Case 2: plain text follow-up, no image ---
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
    log_conversation_turn(session_state["session_id"], session_state["topic"], "user", text)
    try:
        result = chat_session.send_message(text)
        reply = result.text
    except Exception as e:
        reply = f"שגיאה בפנייה ל-Gemini: {str(e)}"

    log_conversation_turn(session_state["session_id"], session_state["topic"], "assistant", reply)
    history.append({"role": "assistant", "content": reply})
    return history, clear_value, session_state


# --- UI ---
CUSTOM_CSS = """
#main-wrap {
    max-width: 700px !important;
    margin: 40px auto !important;
}
#role-page h1 {
    font-size: 2.4rem !important;
}
#role-page h2 {
    font-size: 1.6rem !important;
    margin-bottom: 16px !important;
}
#role-page .role-btn {
    font-size: 1.3rem !important;
    padding: 28px 20px !important;
    height: auto !important;
}
"""

with gr.Blocks(title="Bagrut Physics Tutor") as demo:

    with gr.Column(elem_id="main-wrap"):

        # --- Page 0: password gate ---
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

        # --- Page 1: role selection ---
        with gr.Column(visible=False, elem_id="role-page") as role_page:
            gr.Markdown("# מורה פרטי לפיזיקה - בגרות", rtl=True)
            gr.Markdown("## מי אתה?", rtl=True)
            with gr.Row():
                student_btn = gr.Button("👨‍🎓 אני תלמיד/ה", variant="primary", size="lg", elem_classes="role-btn")
                teacher_btn = gr.Button("👩‍🏫 אני מורה", variant="secondary", size="lg", elem_classes="role-btn")

        # --- Page 2: student chat interface ---
        with gr.Column(visible=False) as student_page:
            student_back_btn = gr.Button("⬅ חזרה", size="sm")
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
                inputs=[msg_box, chatbot, session_state],
                outputs=[chatbot, msg_box, session_state],
            )

        # --- Page 3: teacher dashboard ---
        with gr.Column(visible=False) as teacher_page:
            teacher_back_btn = gr.Button("⬅ חזרה", size="lg")
            gr.Markdown("## לוח בקרה למורה - סטטיסיקת שאלות", rtl=True)

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

            def refresh_stats():
                rows = get_topic_stats()
                import pandas as pd
                df = pd.DataFrame(rows, columns=["נושא", "מספר שאלות"])
                max_count = df["מספר שאלות"].max() if not df.empty else 0
                upper = max(int(max_count * 1.2), 5)
                plot_update = gr.BarPlot(
                    x="נושא",
                    y="מספר שאלות",
                    title="התפלגות שאלות לפי נושא",
                    y_lim=(0, upper),
                    value=df,
                )
                return df, plot_update

            refresh_btn.click(
                fn=refresh_stats, inputs=None, outputs=[stats_table, stats_plot]
            ).then(
                fn=refresh_sessions, inputs=None, outputs=[sessions_dropdown, dialog_view]
            )

            sessions_dropdown.change(
                fn=get_conversation_dialog,
                inputs=[sessions_dropdown],
                outputs=[dialog_view],
            )

        # --- Navigation wiring ---
        def check_password(entered_password):
            if entered_password == APP_PASSWORD:
                return gr.update(visible=False), gr.update(visible=True), ""
            return gr.update(visible=True), gr.update(visible=False), "**סיסמה שגויה, נסה/י שוב.**"

        def go_to_student():
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

        def go_to_teacher():
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

        def go_to_role():
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

        password_submit_btn.click(
            fn=check_password,
            inputs=[password_box],
            outputs=[password_page, role_page, password_error],
        )
        password_box.submit(
            fn=check_password,
            inputs=[password_box],
            outputs=[password_page, role_page, password_error],
        )

        student_btn.click(fn=go_to_student, inputs=None, outputs=[role_page, student_page, teacher_page])
        teacher_btn.click(
            fn=go_to_teacher, inputs=None, outputs=[role_page, student_page, teacher_page]
        ).then(
            fn=refresh_stats, inputs=None, outputs=[stats_table, stats_plot]
        ).then(
            fn=refresh_sessions, inputs=None, outputs=[sessions_dropdown, dialog_view]
        )
        student_back_btn.click(fn=go_to_role, inputs=None, outputs=[role_page, student_page, teacher_page])
        teacher_back_btn.click(fn=go_to_role, inputs=None, outputs=[role_page, student_page, teacher_page])

demo.launch(css=CUSTOM_CSS)
