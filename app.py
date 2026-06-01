"""
AI Study Chatbot — Streamlit UI powered by Google Gemini (google-genai SDK).

Run: streamlit run app.py
"""

from __future__ import annotations

import os
from typing import Iterator, List

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID = "gemini-2.5-flash"
ENV_KEY_NAME = "GEMINI_API_KEY"

SYSTEM_INSTRUCTION = """You are an AI Study Mentor. Your style is:
- Beginner-friendly: define terms simply, avoid unnecessary jargon.
- Educational: explain the "why" behind facts and procedures.
- Motivational: encourage effort, growth mindset, and steady habits.
- Practical: give concrete steps, mini-plans, and examples learners can try today.
- Structured: use short headings, numbered steps, and bullet lists when helpful.

If a question is unclear, ask one brief clarifying question before diving in.
Never invent citations or claim you browsed the web unless tools provided that data.
"""


# =============================================================================
# GEMINI CLIENT & MESSAGES
# =============================================================================


def get_api_key() -> str:
    """Read Gemini API key from environment (populated via python-dotenv / OS)."""
    load_dotenv()
    key = os.getenv(ENV_KEY_NAME, "").strip()
    if not key:
        raise RuntimeError(
            f"Missing {ENV_KEY_NAME}. Add it to your .env file in the project root."
        )
    return key


def get_genai_client() -> genai.Client:
    """Create a single GenAI client for this session."""
    return genai.Client(api_key=get_api_key())


def ui_messages_to_gemini_contents(
    messages: List[dict],
) -> List[types.Content]:
    """
    Convert Streamlit chat history into Gemini `Content` objects.

    The UI uses roles 'user' and 'assistant'. Gemini expects 'user' and 'model'.
    """
    contents: List[types.Content] = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])],
            )
        )
    return contents


def stream_model_reply(
    client: genai.Client,
    contents: List[types.Content],
) -> Iterator[str]:
    """Stream plain-text chunks from Gemini for a ChatGPT-style typing effect."""
    config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
    stream = client.models.generate_content_stream(
        model=MODEL_ID,
        contents=contents,
        config=config,
    )
    for chunk in stream:
        text = getattr(chunk, "text", None) or ""
        if text:
            yield text


# =============================================================================
# STREAMLIT SESSION STATE
# =============================================================================


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "client" not in st.session_state:
        try:
            st.session_state.client = get_genai_client()
            st.session_state.client_error = None
        except RuntimeError as exc:
            st.session_state.client = None
            st.session_state.client_error = str(exc)


# =============================================================================
# UI
# =============================================================================


def render_header() -> None:
    st.markdown(
        """
        <style>
        /* Light, readable chat surface */
        div[data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f6f8fc 0%, #eef2fb 100%);
        }
        section[data-testid="stSidebar"] { background-color: #f0f3fa; }
        .block-container { padding-top: 1.5rem; max-width: 800px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([4, 1])
    with c1:
        st.title("📚 AI Study Mentor")
    with c2:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    st.caption(
        "Ask anything about learning, coding, math, exams, or study habits — "
        "clear explanations, practical steps, and encouragement."
    )


def render_chat() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def main() -> None:
    st.set_page_config(
        page_title="AI Study Mentor",
        page_icon="📚",
        layout="centered",
    )
    init_session_state()
    render_header()

    if st.session_state.client is None:
        st.error(st.session_state.get("client_error", "Could not initialize API client."))
        st.stop()

    render_chat()

    if prompt := st.chat_input("Message your study mentor…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        contents = ui_messages_to_gemini_contents(st.session_state.messages)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_reply: List[str] = []
            try:
                for piece in stream_model_reply(st.session_state.client, contents):
                    full_reply.append(piece)
                    placeholder.markdown("".join(full_reply) + "▌")
                text = "".join(full_reply).strip() or (
                    "I didn't get a reply. Try rephrasing or asking again."
                )
                placeholder.markdown(text)
            except Exception as exc:
                placeholder.error(f"Something went wrong: {exc}")
                st.session_state.messages.pop()  # remove the unanswered user message
                st.stop()

        st.session_state.messages.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    main()
