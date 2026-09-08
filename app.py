from __future__ import annotations

import datetime
import gc
import json
import os
import time
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Zentrix Research Assistant",
    page_icon="assests/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global background particles HTML ──────────────────────────────────────────

PARTICLES_BG_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body {
      width: 100%;
      height: 100%;
      background: transparent;
      overflow: hidden;
    }
    #particles-js {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
    }
  </style>
</head>
<script>
    // Auto-resize iframe to fill available viewport height
    (function() {
      function resize() {
        const vh = window.innerHeight;
        document.documentElement.style.height = vh + 'px';
        document.body.style.height = vh + 'px';
        try {
          // Tell Streamlit to resize this component
          window.parent.postMessage({
            type: 'streamlit:setFrameHeight',
            height: Math.max(vh - 140, 500)
          }, '*');
        } catch(e) {}
      }
      resize();
      window.addEventListener('resize', resize);
    })();
  </script>
<body>
  <div id="particles-js"></div>
  <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
  <script>
    try {
      // 1. Try to inject particles directly into parent window body to break out of iframe transforms/clipping
      const parentDoc = window.parent.document;
      if (!parentDoc.getElementById('global-particles-bg')) {
        const bgDiv = parentDoc.createElement('div');
        bgDiv.id = 'global-particles-bg';
        bgDiv.style.position = 'fixed';
        bgDiv.style.top = '0';
        bgDiv.style.left = '0';
        bgDiv.style.width = '100vw';
        bgDiv.style.height = '100vh';
        bgDiv.style.zIndex = '-9999';
        bgDiv.style.pointerEvents = 'none';
        bgDiv.style.backgroundColor = 'transparent';
        parentDoc.body.appendChild(bgDiv);
        
        if (!window.parent.particlesJS) {
          const script = parentDoc.createElement('script');
          script.src = 'https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js';
          script.onload = function() {
            window.parent.particlesJS("global-particles-bg", getParticlesConfig());
          };
          parentDoc.head.appendChild(script);
        } else {
          window.parent.particlesJS("global-particles-bg", getParticlesConfig());
        }
      }
    } catch (e) {
      console.warn("Same-origin access blocked; falling back to sandboxed iframe background.", e);
      // 2. Fallback: Initialize particles inside this iframe
      particlesJS("particles-js", getParticlesConfig());
    }

    function getParticlesConfig() {
      return {
        "particles": {
          "number": { "value": 150, "density": { "enable": true, "value_area": 1000 } },
          "color": { "value": "#a78bfa" },
          "shape": { "type": "circle" },
          "opacity": {
            "value": 0.5,
            "random": true,
            "anim": { "enable": true, "speed": 0.5, "opacity_min": 0.1, "sync": false }
          },
          "size": {
            "value": 2,
            "random": true,
            "anim": { "enable": false }
          },
          "line_linked": {
            "enable": true,
            "distance": 130,
            "color": "#7c3aed",
            "opacity": 0.25,
            "width": 1
          },
          "move": {
            "enable": true,
            "speed": 0.45,
            "direction": "none",
            "random": true,
            "straight": false,
            "out_mode": "out",
            "bounce": false
          }
        },
        "interactivity": {
          "detect_on": "canvas",
          "events": {
            "onhover": { "enable": false },
            "onclick": { "enable": false },
            "resize": true
          }
        },
        "retina_detect": true
      };
    }
  </script>
</body>
</html>"""


# ── Global CSS injection ───────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Import a distinctive font ── */
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }

        /* ── Dark deep-space background ── */
        .stApp {
            background: linear-gradient(135deg, #0a0a14 0%, #0f0a1e 50%, #0a0f1a 100%);
            color: #e2e8f0;
        }

        /* ── Main content area & padding from edges ── */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 6rem !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
            max-width: 1200px !important;
            margin: 0 auto !important;
        }

        /* Bottom chat input container alignment */
        [data-testid="stBottom"] {
            background: transparent !important;
        }
        [data-testid="stBottom"] > div {
            padding-left: 3rem !important;
            padding-right: 3rem !important;
            max-width: 1200px !important;
            margin: 0 auto !important;
        }

        /* ── Make the welcome iframe fill full height with no gaps ── */
        .block-container > div:first-child > div:first-child > div:first-child iframe {
            display: block !important;
            margin: 0 !important;
            border: none !important;
        }
        [data-testid="stVerticalBlock"] > [data-testid="element-container"]:has(iframe) {
            padding: 0 !important;
            margin: 0 !important;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d0d1f 0%, #10101e 100%);
            border-right: 1px solid rgba(124, 58, 237, 0.25);
        }

        [data-testid="stSidebar"] .stMarkdown h2 {
            font-family: 'Syne', sans-serif;
            font-weight: 800;
            color: #a78bfa;
            letter-spacing: -0.02em;
        }

        /* ── Remove ALL avatars completely ── */
        [data-testid^="stChatMessageAvatar"],
        [data-testid="stChatMessageAvatar"],
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"],
        .stChatMessageAvatar,
        [data-testid="stChatMessage"] > div:first-child:not([data-testid="stChatMessageContent"]) {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* ── Base message container ── */
        [data-testid="stChatMessage"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 0.5rem !important;
            display: flex !important;
            width: 100% !important;
            margin-bottom: 1rem !important;
            box-sizing: border-box !important;
        }

        /* Standardize the bubble content container */
        [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
            padding: 0.85rem 1.25rem !important;
            backdrop-filter: blur(4px) !important;
        }

        /* User message bubble alignment (Right Side - ChatGPT style) */
        [data-testid="stChatMessage"]:has(.user-message-marker) {
            justify-content: flex-end !important;
        }

        [data-testid="stChatMessage"]:has(.user-message-marker) [data-testid="stChatMessageContent"] {
            margin-left: auto !important;
            margin-right: 0 !important;
            background: linear-gradient(135deg, rgba(124, 58, 237, 0.28) 0%, rgba(91, 33, 182, 0.35) 100%) !important;
            border: 1px solid rgba(167, 139, 250, 0.4) !important;
            border-radius: 20px 20px 4px 20px !important; /* ChatGPT style tail on bottom right */
            color: #f3f4f6 !important;
            max-width: 75% !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        }

        /* Assistant message bubble alignment (Left Side - ChatGPT style) */
        [data-testid="stChatMessage"]:has(.assistant-message-marker) {
            justify-content: flex-start !important;
        }

        [data-testid="stChatMessage"]:has(.assistant-message-marker) [data-testid="stChatMessageContent"] {
            margin-right: auto !important;
            margin-left: 0 !important;
            background: rgba(255, 255, 255, 0.035) !important;
            border: 1px solid rgba(124, 58, 237, 0.15) !important;
            border-radius: 20px 20px 20px 4px !important; /* ChatGPT style tail on bottom left */
            color: #e2e8f0 !important;
            max-width: 85% !important;
        }

        /* Support normal streamlit chat messages styling as a fallback */
        [data-testid="stChatMessage"]:not(:has(.user-message-marker)):not(:has(.assistant-message-marker)) {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(124, 58, 237, 0.12);
            border-radius: 20px;
            margin-bottom: 0.5rem;
            padding: 0.75rem 1rem !important;
            backdrop-filter: blur(4px);
        }

        /* ── Chat input box ── */
        [data-testid="stChatInput"] {
            background: rgba(15, 10, 30, 0.85) !important;
            border: 1px solid rgba(124, 58, 237, 0.45) !important;
            border-radius: 20px !important;
            transition: all 0.2s ease !important;
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: #7c3aed !important;
            box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.18) !important;
        }

        [data-testid="stChatInputTextArea"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #e2e8f0 !important;
            font-family: 'DM Sans', sans-serif !important;
        }

        /* ── Metric cards in sidebar ── */
        [data-testid="metric-container"] {
            background: rgba(124, 58, 237, 0.08);
            border: 1px solid rgba(124, 58, 237, 0.2);
            border-radius: 10px;
            padding: 0.5rem 0.75rem;
        }

        [data-testid="stMetricValue"] {
            color: #a78bfa !important;
            font-family: 'Syne', sans-serif !important;
            font-weight: 700 !important;
        }

        /* ── Expander (memory notes) ── */
        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(124, 58, 237, 0.18) !important;
            border-radius: 10px !important;
        }

        [data-testid="stExpander"]:hover {
            border-color: rgba(167, 139, 250, 0.4) !important;
        }

        /* ── Dividers ── */
        hr {
            border-color: rgba(124, 58, 237, 0.2) !important;
        }

        /* ── Buttons ── */
        .stButton > button {
            background: rgba(124, 58, 237, 0.15) !important;
            border: 1px solid rgba(124, 58, 237, 0.4) !important;
            color: #c4b5fd !important;
            border-radius: 8px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.8rem !important;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            background: rgba(124, 58, 237, 0.3) !important;
            border-color: #7c3aed !important;
            color: #ede9fe !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.25);
        }

        /* ── Title ── */
        h1 {
            font-family: 'Syne', sans-serif !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 50%, #60a5fa 100%);
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            letter-spacing: -0.03em;
        }

        /* ── Caption / small text ── */
        .stCaption, [data-testid="stCaptionContainer"] {
            color: #6b7280 !important;
            font-size: 0.78rem !important;
        }

        /* ── Success / Error / Info boxes ── */
        [data-testid="stAlert"] {
            border-radius: 10px !important;
            border-left: 3px solid #7c3aed !important;
            background: rgba(124, 58, 237, 0.08) !important;
        }

        /* ── Session info monospace ── */
        code {
            background: rgba(124, 58, 237, 0.12) !important;
            color: #a78bfa !important;
            border-radius: 4px !important;
            padding: 0.1em 0.35em !important;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: rgba(124, 58, 237, 0.35);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover { background: rgba(124, 58, 237, 0.6); }

        /* ── Full-screen background particles iframe ── */
        div[data-testid="stHtml"] iframe {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            border: none !important;
            z-index: -9999 !important;
            pointer-events: none !important;
        }
        div[data-testid="stHtml"] {
            height: 0px !important;
            margin: 0 !important;
            padding: 0 !important;
            position: absolute !important;
        }

        /* ── Welcome Screen Hero styles ── */
        .hero {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 5rem 1rem 3rem 1rem;
            pointer-events: none;
            width: 100%;
        }

        .badge {
            pointer-events: auto;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(124, 58, 237, 0.18);
            border: 1px solid rgba(167, 139, 250, 0.45);
            border-radius: 999px;
            padding: 6px 20px;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #c4b5fd;
            margin-bottom: 1.6rem;
        }

        .title {
            font-family: 'Syne', sans-serif;
            font-weight: 800;
            font-size: clamp(2.8rem, 7vw, 5rem);
            line-height: 1.05;
            background: linear-gradient(135deg, #ede9fe 0%, #a78bfa 40%, #818cf8 70%, #60a5fa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1.2rem;
            letter-spacing: -0.04em;
        }

        .subtitle {
            font-size: 0.6rem;
            color: #94a3b8;
            max-width: 500px;
            line-height: 1.7;
            margin-bottom: 2.4rem;
            font-weight: 300;
        }

        .pills {
            pointer-events: auto;
            display: flex;
            gap: 0.65rem;
            flex-wrap: wrap;
            justify-content: center;
            margin-bottom: 2.2rem;
        }

        .pill {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(124, 58, 237, 0.25);
            border-radius: 999px;
            padding: 8px 18px;
            font-size: 0.8rem;
            color: #cbd5e1;
            transition: all 0.22s ease;
            cursor: default;
        }

        .pill:hover {
            background: rgba(124, 58, 237, 0.18);
            border-color: rgba(167, 139, 250, 0.55);
            color: #ede9fe;
            transform: translateY(-2px);
        }

        .hint {
            font-size: 0.75rem;
            color: #3f4a5a;
            letter-spacing: 0.06em;
            animation: pulse 2.8s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.4; }
            50%       { opacity: 1;   }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading AI models and database…")
def get_graph():
    from agent.graph import build_graph
    return build_graph()


@st.cache_resource(show_spinner=False)
def get_memory():
    from memory.chroma_manager import ChromaMemoryManager
    manager = ChromaMemoryManager()
    manager.initialise()
    return manager



def _init_session() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "session_start" not in st.session_state:
        st.session_state.session_start = datetime.datetime.now()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "status_text" not in st.session_state:
        st.session_state.status_text = ""
    if "last_intent" not in st.session_state:
        st.session_state.last_intent = ""
    if "last_memory_hit" not in st.session_state:
        st.session_state.last_memory_hit = False
    if "refresh_requested" not in st.session_state:
        st.session_state.refresh_requested = None


_init_session()


def _intent_badge(intent: str, memory_hit: bool) -> str:
    if memory_hit:
        return "🧠 Answered from memory"
    elif intent == "research_and_answer":
        return "🌐 Researched from web"
    elif intent == "answer_from_context":
        return "💬 Answered from conversation"
    else:
        return ""


def _confidence_color(confidence: float) -> str:
    if confidence >= 0.80:
        return "#22c55e"
    elif confidence >= 0.65:
        return "#f59e0b"
    else:
        return "#ef4444"


def _stream_text(text: str):
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.012)


def run_graph_with_status(user_query: str, status_placeholder) -> dict:
    from agent.graph import get_initial_state

    graph = get_graph()
    session_id = st.session_state.session_id
    config = {"configurable": {"thread_id": session_id}}

    if len(st.session_state.messages) <= 1:
        initial_input = get_initial_state(session_id, user_query)
    else:
        initial_input = {"messages": [{"type": "human", "content": user_query}]}

    NODE_STATUS = {
        "classify_intent":      "🔍 Searching memory…",
        "retrieve_from_memory": "📚 Reading stored notes…",
        "search_web":           "🌐 Searching the web…",
        "save_to_memory":       "✍️  Synthesising and saving…",
        "generate_answer":      "💬 Generating answer…",
    }

    final_state = {}

    for chunk in graph.stream(
        initial_input,
        config=config,
        stream_mode="updates",
    ):
        for node_name, state_update in chunk.items():
            status_msg = NODE_STATUS.get(node_name, f"⚙️ Running {node_name}…")
            status_placeholder.markdown(
                f'<div style="color:#a78bfa;font-size:0.82rem;'
                f'background:rgba(124,58,237,0.08);padding:6px 12px;'
                f'border-radius:8px;border-left:2px solid #7c3aed;'
                f'display:inline-block">{status_msg}</div>',
                unsafe_allow_html=True,
            )
            if isinstance(state_update, dict):
                final_state.update(state_update)

    status_placeholder.empty()
    return final_state

def render_particles_background() -> None:
    pass  # handled inside render_welcome

def render_welcome() -> None:
    """Particles canvas + hero text together in one iframe at real height."""
    import streamlit.components.v1 as components
    components.html("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body { width: 100%; height: 100%; background: #0a0a14; overflow: hidden; font-family: 'DM Sans', sans-serif; }
    #particles-js { position: absolute; inset: 0; width: 100%; height: 100%; z-index: 0; }
    .hero {
      position: absolute; inset: 0; z-index: 10;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      text-align: center; padding: 2rem; pointer-events: none;
    }
    .badge {
      pointer-events: auto;
      display: inline-flex; align-items: center; gap: 6px;
      background: rgba(124,58,237,0.18); border: 1px solid rgba(167,139,250,0.45);
      border-radius: 999px; padding: 6px 20px;
      font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
      color: #c4b5fd; margin-bottom: 1.6rem;
    }
    .title {
      font-family: 'Georgia', sans-serif; font-weight: 800;
      font-size: clamp(2.8rem, 7vw, 5rem); line-height: 1.05;
      background: linear-gradient(135deg, #ede9fe 0%, #a78bfa 40%, #818cf8 70%, #60a5fa 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
      margin-bottom: 1.2rem; letter-spacing: -0.04em;
    }
    .subtitle { font-size: 1.05rem; color: #94a3b8; max-width: 500px; line-height: 1.7; margin-bottom: 2.4rem; font-weight: 300; }
    .pills { pointer-events: auto; display: flex; gap: 0.65rem; flex-wrap: wrap; justify-content: center; margin-bottom: 2.2rem; }
    .pill {
      background: rgba(255,255,255,0.05); border: 1px solid rgba(124,58,237,0.25);
      border-radius: 999px; padding: 8px 18px; font-size: 0.8rem; color: #cbd5e1;
      transition: all 0.22s ease; cursor: default;
    }
    .pill:hover { background: rgba(124,58,237,0.18); border-color: rgba(167,139,250,0.55); color: #ede9fe; transform: translateY(-2px); }
    .hint { font-size: 0.75rem; color: #3f4a5a; letter-spacing: 0.06em; animation: pulse 2.8s ease-in-out infinite; }
    @keyframes pulse { 0%,100%{opacity:0.4} 50%{opacity:1} }
  </style>
</head>
<body>
  <div id="particles-js"></div>
  <div class="hero">
    <div class="badge">✦ AI Research Assistant</div>
    <div class="title">Zentrix</div>
    <div class="subtitle">Ask me anything. I search the web, synthesise findings,<br>and remember what I've learned.<br> Getting smarter with every question.</div>
    <div class="pills">
      <span class="pill">🌐 Live web research</span>
      <span class="pill">🧠 Persistent memory</span>
      <span class="pill">⚡ Instant recall</span>
      <span class="pill">🔄 Auto-refresh topics</span>
    </div>
    <div class="hint">↓ Type your first question below to begin</div>
  </div>
  <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
  <script>
    particlesJS("particles-js", {
      "particles": {
        "number": { "value": 180, "density": { "enable": true, "value_area": 900 } },
        "color": { "value": "#a78bfa" },
        "shape": { "type": "circle" },
        "opacity": { "value": 0.6, "random": true, "anim": { "enable": true, "speed": 0.6, "opacity_min": 0.1, "sync": false } },
        "size": { "value": 2.2, "random": true },
        "line_linked": { "enable": true, "distance": 120, "color": "#7c3aed", "opacity": 0.32, "width": 1 },
        "move": { "enable": true, "speed": 0.45, "direction": "none", "random": true, "out_mode": "out" }
      },
      "interactivity": {
        "detect_on": "canvas",
        "events": { "onhover": { "enable": true, "mode": "grab" }, "onclick": { "enable": true, "mode": "bubble" }, "resize": true },
        "modes": {
          "grab": { "distance": 180, "line_linked": { "opacity": 1 } },
          "bubble": { "distance": 260, "size": 6, "duration": 1.5, "opacity": 0.9 }
        }
      },
      "retina_detect": true
    });
  </script>
</body>
</html>""", height=800, scrolling=False)

def render_sidebar() -> None:
    import base64
    logo_html = "🧠"
    try:
        logo_path = "assests/logo.png"
        if not os.path.exists(logo_path):
            logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assests", "logo.png")
        if os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode("utf-8")
            logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="width:22px;height:22px;object-fit:contain;mix-blend-mode:screen;"/>'
    except Exception:
        pass

    with st.sidebar:
        # Logo / title
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem">
              <div style="width:36px;height:36px;border-radius:10px;
                          background:linear-gradient(135deg,#7c3aed,#3b82f6);
                          display:flex;align-items:center;justify-content:center;
                          flex-shrink:0">
                {logo_html}
              </div>
              <div>
                <div style="font-family:'Georgia',sans-serif;font-weight:800;
                             font-size:1.1rem;color:#ede9fe;letter-spacing:-0.02em">
                  Zentrix
                </div>
                <div style="font-size:0.68rem;color:#6b7280;letter-spacing:0.05em;
                             text-transform:uppercase">Memory Explorer</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Everything this agent has learned and stored.")

        memory = get_memory()
        note_count = memory.note_count()
        chunk_count = memory.chunk_count()

        col1, col2 = st.columns(2)
        col1.metric("Research Notes", note_count)
        col2.metric("Source Chunks", chunk_count)

        st.divider()

        if note_count == 0:
            st.markdown(
                """
                <div style="background:rgba(124,58,237,0.07);border:1px solid
                rgba(124,58,237,0.18);border-radius:10px;padding:14px 16px;
                color:#94a3b8;font-size:0.83rem;line-height:1.6">
                  No notes stored yet.<br><br>
                  Ask a question and the agent will research it and
                  save the findings here automatically.
                </div>
                """,
                unsafe_allow_html=True,
            )
            _render_session_footer()
            return

        try:
            raw = memory._notes_collection.get(include=["metadatas", "documents"])
        except Exception:
            st.warning("Could not load memory notes.")
            return

        metadatas = raw.get("metadatas", [])
        documents = raw.get("documents", [])

        note_data = sorted(
            zip(metadatas, documents),
            key=lambda x: float(x[0].get("timestamp", 0)),
            reverse=True,
        )

        for meta, doc in note_data:
            topic      = meta.get("topic", "unknown")
            timestamp  = float(meta.get("timestamp", 0))
            confidence = float(meta.get("confidence", 0))
            key_facts_raw = meta.get("key_facts", "[]")

            try:
                key_facts = json.loads(key_facts_raw)
            except Exception:
                key_facts = []

            age_days = (time.time() - timestamp) / 86400
            if age_days < 1:
                age_str = "Today"
            elif age_days < 2:
                age_str = "Yesterday"
            else:
                age_str = f"{int(age_days)}d ago"

            stored_date = datetime.datetime.fromtimestamp(timestamp).strftime("%b %d")
            conf_color  = _confidence_color(confidence)
            conf_pct    = int(confidence * 100)

            with st.expander(f"**{topic.title()}**", expanded=False):
                st.markdown(
                    f'<span style="background:{conf_color};color:white;'
                    f'padding:2px 8px;border-radius:12px;font-size:0.72rem;'
                    f'font-weight:600">{conf_pct}%&nbsp;confidence</span>'
                    f'&nbsp;&nbsp;'
                    f'<span style="color:#6b7280;font-size:0.72rem">'
                    f'{age_str}&nbsp;·&nbsp;{stored_date}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")
                st.caption(doc[:200] + "…" if len(doc) > 200 else doc)

                if key_facts:
                    st.markdown(
                        '<div style="font-size:0.78rem;color:#94a3b8;'
                        'margin-top:6px;font-weight:600">Key facts</div>',
                        unsafe_allow_html=True,
                    )
                    for fact in key_facts[:3]:
                        st.markdown(
                            f'<div style="font-size:0.78rem;color:#cbd5e1;'
                            f'padding:2px 0 2px 10px;border-left:2px solid '
                            f'rgba(124,58,237,0.4)">{fact}</div>',
                            unsafe_allow_html=True,
                        )

                if st.button(
                    "🔄 Refresh topic",
                    key=f"refresh_{topic}",
                    help="Deletes this note and re-researches on your next query",
                ):
                    deleted = memory.delete_note(topic)
                    if deleted:
                        st.success(f"Deleted '{topic}'. Ask again to refresh.")
                        st.rerun()
                    else:
                        st.error("Could not delete note. Try again.")

        st.divider()
        _render_session_footer()


def _render_session_footer() -> None:
    st.markdown(
        f"""
        <div style="font-size:0.72rem;color:#374151;line-height:1.8">
          <span style="color:#6b7280">Session</span>&nbsp;
          <code style="font-size:0.68rem">{st.session_state.session_id[:10]}…</code><br>
          <span style="color:#6b7280">Started</span>&nbsp;
          <span style="color:#4b5563">
            {st.session_state.session_start.strftime('%H:%M:%S')}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_header() -> None:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🧠 Zentrix Research Assistant")
        st.caption(
            "Ask me anything · I search the web, synthesise findings into memory, "
            "and recall them instantly in future sessions."
        )
    with col2:
        st.markdown(
            f"""
            <div style="text-align:right;font-size:0.75rem;color:#6b7280;
                        padding-top:0.5rem;line-height:1.9">
              <span style="color:#4b5563">Session</span><br>
              <code>{st.session_state.session_id[:10]}…</code><br>
              <span style="color:#4b5563">
                {st.session_state.session_start.strftime('%H:%M:%S')}
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chat() -> None:
    """Render message history and handle new input."""

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(f'<div class="user-message-marker"></div>\n\n{msg["content"]}', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message-marker"></div>\n\n{msg["content"]}', unsafe_allow_html=True)
            if msg.get("badge"):
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#7c3aed;'
                    f'margin-top:4px">{msg["badge"]}</div>',
                    unsafe_allow_html=True,
                )

    status_placeholder = st.empty()

    user_input = st.chat_input("Ask me to research anything…")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message-marker"></div>\n\n{user_input}', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        answer_placeholder.markdown(
            '<div class="assistant-message-marker"></div><span style="color:#6b7280;font-style:italic">Thinking…</span>',
            unsafe_allow_html=True,
        )

        try:
            final_state = run_graph_with_status(user_input, status_placeholder)
        except Exception as exc:
            error_msg = (
                f"⚠️ An error occurred: {exc}\n\n"
                "Please check your API keys in `.env` and try again."
            )
            answer_placeholder.markdown(f'<div class="assistant-message-marker"></div>\n\n{error_msg}', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            return

        answer     = final_state.get("final_answer", "I was unable to generate a response.")
        intent     = final_state.get("intent", "")
        memory_hit = final_state.get("memory_hit", False)

        streamed = ""
        for chunk in _stream_text(answer):
            streamed += chunk
            answer_placeholder.markdown(f'<div class="assistant-message-marker"></div>\n\n{streamed}▌', unsafe_allow_html=True)
        answer_placeholder.markdown(f'<div class="assistant-message-marker"></div>\n\n{answer}', unsafe_allow_html=True)

        badge = _intent_badge(intent, memory_hit)
        if badge:
            st.markdown(
                f'<div style="font-size:0.75rem;color:#7c3aed;margin-top:4px">{badge}</div>',
                unsafe_allow_html=True,
            )

        st.session_state.last_intent    = intent
        st.session_state.last_memory_hit = memory_hit
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "badge": badge,
        })


def main() -> None:
    render_particles_background()
    _inject_css()
    render_sidebar()

    has_messages = bool(st.session_state.messages)

    if not has_messages:
        # Show constellation welcome screen
        render_welcome()
    else:
        # Normal chat layout
        render_header()
        st.divider()

    render_chat()


if __name__ == "__main__":
    main()