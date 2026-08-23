import os
import mimetypes
import gradio as gr
from google import genai
from google.genai import types

# --- CONFIGURATION ---
CLASSIFY_MODEL = "gemini-3.1-flash-lite"
SOLVE_MODEL = "gemini-3.5-flash"

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
FORMAT REQUIREMENT:
Respond in EXACTLY this format, nothing else:
PHYSICS: yes or no
DESCRIPTION: <one sentence describing what's in the image>
"""

SOCRATIC_SYSTEM_PROMPT = """
You are an encouraging, highly expert high school physics private tutor for Israeli students.
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


# --- HELPER FUNCTIONS ---
def file_to_part(path: str) -> types.Part:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"
    with open(path, "rb") as f:
        data = f.read()
    return types.Part.from_bytes(data=data, mime_type=mime)


def gemini_classify(image_part) -> dict:
    result = client.models.generate_content(
        model=CLASSIFY_MODEL,
        contents=[image_part, CLASSIFY_PROMPT],
    )
    raw_text = result.text
    parsed = {"physics": False, "description": ""}
    for line in raw_text.splitlines():
        line = line.strip()
        if line.upper().startswith("PHYSICS:"):
            parsed["physics"] = "yes" in line.lower()
        elif line.upper().startswith("DESCRIPTION:"):
            parsed["description"] = line.split(":", 1)[1].strip()
    return parsed


# --- CHAT PIPELINE ---
def handle_message(message, history, chat_session):
    """Single entry point for the unified chat box.
    `message` is a dict from MultimodalTextbox: {"text": str, "files": [paths]}."""
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
            return history, clear_value, None

        if not classification["physics"]:
            history.append({
                "role": "assistant",
                "content": f"**זו לא נראית שאלת פיזיקה לבגרות.**\n\nתיאור: {classification['description']}"
            })
            return history, clear_value, None

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

        try:
            result = chat_session.send_message([image_part, opening_message])
            reply = result.text
        except Exception as e:
            history.append({"role": "assistant", "content": f"שגיאה בפנייה ל-Gemini: {str(e)}"})
            return history, clear_value, None

        history.append({"role": "assistant", "content": reply})
        return history, clear_value, chat_session

    # --- Case 2: plain text follow-up, no image ---
    if not text:
        return history, clear_value, chat_session

    history.append({"role": "user", "content": text})

    if chat_session is None:
        history.append({
            "role": "assistant",
            "content": "כדי להתחיל, צריך להעלות תמונה של שאלת פיזיקה 📎"
        })
        return history, clear_value, chat_session

    try:
        result = chat_session.send_message(text)
        reply = result.text
    except Exception as e:
        reply = f"שגיאה בפנייה ל-Gemini: {str(e)}"

    history.append({"role": "assistant", "content": reply})
    return history, clear_value, chat_session


# --- UI ---
with gr.Blocks(title="Bagrut Physics Tutor") as demo:
    gr.Markdown("## מורה פרטי לפיזיקה - בגרות")

    chat_session_state = gr.State(None)

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
        inputs=[msg_box, chatbot, chat_session_state],
        outputs=[chatbot, msg_box, chat_session_state],
    )

demo.launch()
