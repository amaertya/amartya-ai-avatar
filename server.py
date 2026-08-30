import os
import re
import traceback
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

# Replace with your active Groq API Key if not set as an environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
client = Groq(api_key=GROQ_API_KEY)

def select_working_model(groq_client):
    """Dynamically tests models on your account and picks the first active, non-reasoning chat model."""
    try:
        model_list = [m.id for m in groq_client.models.list().data]
        print("\n================ AVAILABLE MODELS ON YOUR ACCOUNT ================")
        for m in model_list:
            print(f" -> {m}")
        print("==================================================================\n")

        # Exclude audio, vision, guard, rate-limited, and reasoning models
        blocked_keywords = [
            "whisper", "orpheus", "canopylabs", "guard", "vision", 
            "embed", "120b", "deepseek", "r1", "reason", "thinking", "compound"
        ]
        
        candidates = [m for m in model_list if not any(k in m.lower() for k in blocked_keywords)]

        # Prioritize standard fast conversational models
        candidates.sort(
            key=lambda x: (
                "llama-3.3" in x.lower() or "llama3" in x.lower(),
                "gemma" in x.lower() or "mixtral" in x.lower(),
                "8b" in x.lower() or "70b" in x.lower()
            ),
            reverse=True
        )

        for model_id in candidates:
            try:
                print(f"Testing model: {model_id} ...", end=" ")
                groq_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=5
                )
                print("[READY]")
                print(f"\n>>> ACTIVE CHAT MODEL SELECTED: {model_id}\n")
                return model_id
            except Exception as err:
                print(f"[SKIPPED: {err}]")
                continue

    except Exception as e:
        print(f"[Model Discovery Failed]: {e}")

    return "llama-3.3-70b-versatile"

ACTIVE_MODEL = select_working_model(client)

SYSTEM_PROMPT = """
You are an AI avatar representing Amartya BS in an interview for the AI Agent Team at 100x.
Always respond in the first person ("I", "my", "me").
Keep responses concise, clear, and natural for voice conversation (around 2 to 4 sentences per answer).
Never invent experiences, skills, or metrics outside the provided resume context.

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

=== SAMPLE STAGE 1 BEHAVIORAL ANSWERS ===
- Life Story: "I am a Computer Science graduate from East West Institute of Technology with an 8.4 CGPA. Over the past couple of years, my focus shifted heavily into Generative AI and autonomous agent workflows, having built production-grade document intelligence pipelines at L&T Technology Services and competitive ML platforms."
- #1 Superpower: "My primary strength is execution speed with engineering pragmatism—taking complex AI research or modern LLM architectures and turning them into stable, low-latency, production-ready tools."
- Top 3 Areas to Grow: "First, mastering multi-agent consensus and orchestrations using frameworks like LangGraph. Second, low-level inference optimization for open-weights models. Third, scaling distributed real-time voice architectures."
- Coworker Misconception: "Colleagues sometimes perceive me as purely heads-down on code and architecture, but I actually rely heavily on cross-functional communication and fast iterative feedback loops."
- Pushing Boundaries: "I push my boundaries by taking on end-to-end deployment challenges under tight constraints, whether it's building document parsing systems for 450+ enterprise specs or shipping live hackathon prototypes."
"""

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
    try:
        data = request.get_json(force=True)
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "No message provided."}), 400

        print(f"[USER]: {user_message}")

        completion = client.chat.completions.create(
            model=ACTIVE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.6,
            max_tokens=350
        )

        raw_reply = completion.choices[0].message.content or ""
        
        # Strip all think tags cleanly (both closed and unclosed)
        clean_reply = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", raw_reply).strip()
        reply = clean_reply if clean_reply else raw_reply.strip()

        if not reply:
            reply = "I'm doing well, thank you! Ready to discuss my experience and projects for the 100x AI Agent Team."

        print(f"[AGENT]: {reply}\n")
        return jsonify({"reply": reply})

    except Exception as e:
        print("[ERROR]:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)