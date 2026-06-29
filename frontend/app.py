"""
Streamlit Frontend — RAG Document Assistant.

Features:
  - Document upload (PDF, TXT, MD) via sidebar
  - Chat sessions with history persistence
  - Streaming responses with live token display
  - API key authentication support
  - Graceful error handling for offline backend
"""

import json

import requests
import sseclient
import streamlit as st

# ==========================================
# Configuration
# ==========================================
BASE_URL = "http://localhost:8000"
UPLOAD_URL = f"{BASE_URL}/api/upload"
CHAT_URL = f"{BASE_URL}/api/chat"
STREAM_URL = f"{BASE_URL}/api/chat/stream"
HISTORY_URL = f"{BASE_URL}/api/chat/history"

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="centered",
)


# ==========================================
# Session State Initialisation
# ==========================================
def _init_state():
    defaults = {
        "messages": [],
        "session_id": None,
        "api_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ==========================================
# API Helpers
# ==========================================
def _headers() -> dict:
    h = {}
    if st.session_state.api_key:
        h["X-API-Key"] = st.session_state.api_key
    return h


def upload_file_to_backend(file):
    """Upload a file to the backend ingestion pipeline."""
    try:
        import mimetypes

        mime_type, _ = mimetypes.guess_type(file.name)
        if not mime_type:
            mime_type = "application/octet-stream"

        files = {"file": (file.name, file.getvalue(), mime_type)}
        response = requests.post(UPLOAD_URL, files=files, headers=_headers(), timeout=30)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, str(e)


def send_chat_message(message: str, session_id: str | None = None):
    """Send a chat message and get a complete JSON response."""
    try:
        payload = {"question": message}
        if session_id:
            payload["session_id"] = session_id
        response = requests.post(CHAT_URL, json=payload, headers=_headers(), timeout=60)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, str(e)


def stream_chat_message(message: str, session_id: str | None = None):
    """Send a chat message and return an SSE stream iterator."""
    payload = {"question": message}
    if session_id:
        payload["session_id"] = session_id
    response = requests.post(
        STREAM_URL,
        json=payload,
        headers={**_headers(), "Accept": "text/event-stream"},
        stream=True,
        timeout=120,
    )
    response.raise_for_status()
    return response


# ── History helpers ──────────────────────────────────────────────────────────


def fetch_sessions():
    try:
        r = requests.get(HISTORY_URL, headers=_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def create_session(title: str = "New Chat"):
    try:
        r = requests.post(HISTORY_URL, json={"title": title}, headers=_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def fetch_session_messages(session_id: str):
    try:
        r = requests.get(f"{HISTORY_URL}/{session_id}", headers=_headers(), timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("messages", [])
    except Exception:
        return []


def delete_session(session_id: str):
    try:
        r = requests.delete(f"{HISTORY_URL}/{session_id}", headers=_headers(), timeout=10)
        r.raise_for_status()
        return True
    except Exception:
        return False


# ==========================================
# UI Components
# ==========================================
def render_sidebar():
    """Renders sidebar with auth, file upload, and chat history."""
    with st.sidebar:
        st.header("🤖 RAG Assistant")

        # ── API Key ──────────────────────────────────────────────────────
        with st.expander("🔑 Authentication", expanded=False):
            api_key = st.text_input(
                "API Key",
                type="password",
                value=st.session_state.api_key,
                placeholder="Leave empty if auth is disabled",
            )
            if api_key != st.session_state.api_key:
                st.session_state.api_key = api_key

        st.divider()

        # ── File Upload ──────────────────────────────────────────────────
        st.subheader("📁 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["pdf", "txt", "md"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            if st.button("Upload to Backend", use_container_width=True):
                with st.spinner("Uploading and processing…"):
                    success, result = upload_file_to_backend(uploaded_file)
                    if success:
                        st.toast("File uploaded successfully!", icon="✅")
                    else:
                        st.error(f"Upload failed.\n\n{result}")

        st.divider()

        # ── Chat Sessions ────────────────────────────────────────────────
        st.subheader("💬 Chat Sessions")

        if st.button("➕ New Chat", use_container_width=True):
            session = create_session()
            if session:
                st.session_state.session_id = session["id"]
                st.session_state.messages = []
                st.rerun()

        sessions = fetch_sessions()
        for s in sessions:
            cols = st.columns([4, 1])
            label = s["title"][:30] + ("…" if len(s["title"]) > 30 else "")
            is_active = st.session_state.session_id == s["id"]
            if cols[0].button(
                f"{'▶ ' if is_active else ''}{label}",
                key=f"sess_{s['id']}",
                use_container_width=True,
            ):
                st.session_state.session_id = s["id"]
                # Load messages from history
                msgs = fetch_session_messages(s["id"])
                st.session_state.messages = [
                    {
                        "role": m["role"],
                        "content": m["content"],
                        "sources": m.get("sources", []),
                    }
                    for m in msgs
                ]
                st.rerun()
            if cols[1].button("🗑️", key=f"del_{s['id']}"):
                delete_session(s["id"])
                if st.session_state.session_id == s["id"]:
                    st.session_state.session_id = None
                    st.session_state.messages = []
                st.rerun()


def render_source_references(sources):
    """Renders source references inside an expander."""
    if not sources:
        return
    with st.expander("📚 View Source References"):
        for idx, source in enumerate(sources):
            st.markdown(f"**Source {idx + 1}**")
            fname = source.get("filename", "Unknown")
            page = source.get("page", "—")
            st.markdown(f"**File:** `{fname}`  |  **Page:** `{page}`")
            st.info(source.get("text", "No text snippet available."))
            if idx < len(sources) - 1:
                st.divider()


def render_chat_message(message):
    """Renders a single chat message with optional source refs."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_source_references(message["sources"])


# ==========================================
# Streaming Helpers
# ==========================================
def consume_sse_stream(response):
    """
    Consume an SSE stream from the backend.
    Yields text tokens. Returns sources list after stream ends.
    """
    sources = []
    client = sseclient.SSEClient(response)
    for event in client.events():
        if event.event == "sources":
            try:
                sources = json.loads(event.data)
            except json.JSONDecodeError:
                pass
        elif event.event == "done":
            break
        else:
            # Default "message" event — yield token
            yield event.data
    # Attach sources to the generator (caller reads via .sources attribute)
    consume_sse_stream._last_sources = sources


# ==========================================
# Main Application
# ==========================================
def main():
    st.title("📄 RAG Document Assistant")

    render_sidebar()

    # Render chat history
    for message in st.session_state.messages:
        render_chat_message(message)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents…"):
        # Auto-create session if none exists
        if not st.session_state.session_id:
            session = create_session(prompt[:80])
            if session:
                st.session_state.session_id = session["id"]

        # Append and render user message
        user_msg = {"role": "user", "content": prompt, "sources": []}
        st.session_state.messages.append(user_msg)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Handle assistant response via streaming
        with st.chat_message("assistant"):
            try:
                response = stream_chat_message(prompt, st.session_state.session_id)

                # Stream tokens into the UI
                full_answer_parts = []
                placeholder = st.empty()

                client = sseclient.SSEClient(response)
                sources = []
                for event in client.events():
                    if event.event == "sources":
                        try:
                            sources = json.loads(event.data)
                        except json.JSONDecodeError:
                            pass
                    elif event.event == "done":
                        break
                    else:
                        full_answer_parts.append(event.data)
                        placeholder.markdown("".join(full_answer_parts) + "▌")

                full_answer = "".join(full_answer_parts)
                placeholder.markdown(full_answer)

                render_source_references(sources)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_answer, "sources": sources}
                )

                # Persist to backend history
                if st.session_state.session_id:
                    try:
                        send_chat_message(prompt, st.session_state.session_id)
                    except Exception:
                        pass  # History is best-effort; streaming already worked

            except requests.exceptions.RequestException:
                # Fallback: try non-streaming endpoint
                success, response_data = send_chat_message(
                    prompt, st.session_state.session_id
                )
                if success:
                    answer = response_data.get("answer", "No answer provided.")
                    sources = response_data.get("sources", [])
                    st.markdown(answer)
                    render_source_references(sources)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                else:
                    error_msg = (
                        "**Error connecting to backend.** "
                        f"Please ensure it is running at `{BASE_URL}`."
                    )
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg, "sources": []}
                    )


if __name__ == "__main__":
    main()
