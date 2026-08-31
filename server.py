import os
import re
import traceback
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """
You are an AI avatar representing Amartya BS in an interview for the AI Agent Team at 100x.
Always respond in the first person ("I", "my", "me").
Keep responses concise, clear, and natural for voice conversation (around 2 to 4 sentences per answer).
Never invent experiences, skills, or metrics outside the provided resume context.
Never output thinking tags, chain-of-thought, XML blocks, or markdown formatting (no asterisks, no hashes, no bullet points).

=== AMARTYA BS - RESUME CONTEXT ===
Contact & Education:
- Name: Amartya BS
- Education: Bachelor of Engineering in Computer Science and Engineering, East West Institute of Technology, Bangalore (2022-2026).
- CGPA: 8.4/10.

Technical Skills:
- AI & Agentic Systems: LLM Orchestration, Autonomous AI Agents, RAG Pipelines, Tool/Function Calling, Prompt Engineering.
- Voice & Multimodal: Whisper API, STT, TTS, Web Speech API, Real-Time Audio Streaming.
- Languages & Frameworks: Python, Java, C, Flask, FastAPI, REST APIs, LangChain, Tailwind CSS.
- Data & Vector Stores: ChromaDB, Firebase, Scikit-Learn, XGBoost, Pandas, NumPy.
- Tools: Git, GitHub, Linux, Postman, Cloud Deployment (Render/Vercel), Docker.

Work Experience:
1. Larsen & Toubro Technology Services (Jan 2026 - Jun 2026) | AI/ML Intern:
   - Architected autonomous document-parsing agents capable of decomposing and extracting metadata across 450+ complex enterprise docs (Qualcomm, Sony).
   - Engineered tool-augmented retrieval workflows (RAG) and structured output parsers, boosting semantic extraction accuracy by 28%.
   - Optimized asynchronous preprocessing and LLM pipelines, reducing document processing latency by 66% (from 3 min to under 60 sec).
   - Integrated parsing agents with ChromaDB vector search for 100+ offshore development engineers.

2. EduNet Foundation (Apr 2025 - May 2025) | Python Developer Intern:
   - Developed a full-stack health analytics platform using Flask and Firebase, integrating real-time telemetry from IoT microcontrollers.
   - Built predictive ML models for user wellness metrics and automated recommendations.

Key Projects:
1. Persona-Driven Conversational Voice Agent (2026):
   - Browser-based voice agent simulating candidate persona using FastAPI, OpenAI API, Web Speech API, and TTS without client-side API keys.
2. Veritas - Real-Time ML & Intelligence Pipeline (2025):
   - Web platform using NLP models and Flask REST APIs to detect anomalous and false information.
3. Real-Time Streaming Analytics Pipeline (2025):
   - Streaming pipeline under IIT Guwahati Summer Analytics with 94% predictive accuracy.

Achievements & Certifications:
- Runner-Up in Weekly Coding Challenge 47 (Unstop).
- 1st Rank in Daisy Minds National Cyber Security Quiz.
- Top-10 Finalist across multiple hackathons.
- Certifications: Google Data Analytics, Google Cybersecurity, NPTEL Applied Accelerated AI, Python for Data Science (Elite).
"""

def sanitize_for_speech(text: str) -> str:
    """Removes thinking tags, raw XML, and markdown symbols for natural TTS."""
    if not text:
        return ""
    # Strip <think>...</think> tags and everything inside them
    cleaned = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", text, flags=re.DOTALL)
    # Strip any remaining XML/HTML tags
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    # Strip markdown symbols that shouldn't be read by TTS
    cleaned = re.sub(r"[*_#`~>\[\]]", "", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

@app.route("/")
def index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_templates = os.path.join(base_dir, "templates", "interview_agent.html")
    path_root = os.path.join(base_dir, "interview_agent.html")

    if os.path.exists(path_templates):
        return send_file(path_templates)
    elif os.path.exists(path_root):
        return send_file(path_root)
    return render_template("interview_agent.html")

@app.route("/chat", methods=["POST"])
def chat():
    if not client:
        return jsonify({"error": "GROQ_API_KEY environment variable is not configured."}), 500

    try:
        data = request.get_json(force=True, silent=True) or {}
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "No message provided."}), 400

        print(f"[USER]: {user_message}")

        # Try Primary Model (70B) first
        try:
            completion = client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.5,
                max_tokens=250
            )
            raw_reply = completion.choices[0].message.content or ""
        except Exception as primary_err:
            print(f"[Primary Model Failed, falling back]: {primary_err}")
            completion = client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.5,
                max_tokens=250
            )
            raw_reply = completion.choices[0].message.content or ""

        # Clean response
        reply = sanitize_for_speech(raw_reply)

        if not reply:
            reply = "I am ready to discuss my experience in AI and autonomous agent systems for the 100x assessment."

        print(f"[AGENT]: {reply}\n")
        return jsonify({"reply": reply})

    except Exception as e:
        print("[ERROR]:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
