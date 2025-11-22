# frontend.py – DARK PRO FITNESS COACH UI
import streamlit as st
from backend import chatbot, print_graph_in_terminal
from langchain_core.messages import HumanMessage
import uuid

st.set_page_config(page_title="FitCoach AI", page_icon="black_circle", layout="centered")

# ── DARK THEME CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] {background:#0d1117; color:#e6edf3; font-family:'Inter',sans-serif;}
    .main-header {text-align:center; padding:3rem 1rem 2rem; background:linear-gradient(135deg,#161b22,#0d1117); border-bottom:1px solid #30363d;}
    h1 {font-size:4.2rem !important; font-weight:800; background:linear-gradient(90deg,#00d4ff,#8b5cf6,#ff6bcb);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-1px;}
    .subtitle {color:#8b949e; font-size:1.4rem; margin-top:0.8rem;}
    .badge {background:#21262d; color:#58a6ff; padding:0.4rem 1rem; border-radius:50px; border:1px solid #30363d; font-size:0.9rem;}
    .chat-container {max-width:800px; margin:0 auto; padding:0 1rem;}
    [data-testid="stChatMessage"] {border-radius:16px !important; padding:16px 20px !important; margin:16px 0 !important;
        border:1px solid #30363d; box-shadow:0 4px 12px rgba(0,0,0,0.3);}
    div[data-testid="stChatMessage"][aria-label="user"] {background:#238636; color:white; border-left:4px solid #2ea043;}
    div[data-testid="stChatMessage"][aria-label="assistant"] {background:#1a1f2e; color:#f0f6fc; border-left:4px solid #8b5cf6;}
    .stChatInput > div > div {background:#161b22 !important; border:1px solid #30363d !important; border-radius:16px !important;}
    .stChatInput input {color:#e6edf3 !important; font-size:1.1rem !important;}
    .stChatInput > div > div:focus-within {border-color:#8b5cf6 !important; box-shadow:0 0 0 3px rgba(139,92,246,0.2);}
    .footer {text-align:center; padding:2rem; color:#8b949e; font-size:0.9rem; margin-top:3rem; border-top:1px solid #30363d;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>Fitness & Nutrition Coach</h1>
    <div class="subtitle">Your Personal Fitness & Nutrition Coach</div>
    <div class="badge">Gemini 2.5 Pro • Full Memory • Smart Routing</div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome = """
**Hey Champion!** black_circle

I'm your dedicated **Fitness & Nutrition Coach** — here to help you:
- Build muscle • Lose fat • Get stronger
- Custom workout programs
- Meal plans & macros
- Supplements & recovery

Just tell me your goal, stats, or any question and I’ll build you a **100% personalized plan**.

**What are we crushing today?**
    """
    st.session_state.messages.append({"role": "assistant", "content": welcome})

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# User input
if prompt := st.chat_input("Ask about training, diet, supplements, or goals..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking like a pro coach..."):
            result = chatbot.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            )
            response = result["messages"][-1].content
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    print_graph_in_terminal()
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    FitCoach AI • Built with LangGraph + Gemini 2.5 Pro • 2025
</div>
""", unsafe_allow_html=True)