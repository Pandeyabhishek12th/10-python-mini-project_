"""Streamlit chat UI for a model running on a local Ollama server."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


OLLAMA_URL = "http://localhost:11434"

st.set_page_config(page_title="Local AI Assistant", page_icon="AI", layout="centered")
st.title("Local AI Assistant")
st.caption("A private chatbot powered by a model running on your computer.")


@st.cache_data(ttl=15)
def get_models() -> list[str]:
    """Return model names available from the local Ollama server."""
    request = Request(f"{OLLAMA_URL}/api/tags")
    with urlopen(request, timeout=5) as response:
        data = json.loads(response.read().decode("utf-8"))
    return [item["name"] for item in data.get("models", []) if item.get("name")]


try:
    models = get_models()
except HTTPError as exc:
    st.error(f"Ollama returned HTTP {exc.code}: {exc.reason}")
    st.stop()
except (URLError, TimeoutError, json.JSONDecodeError) as exc:
    st.error(f"Cannot connect to Ollama at {OLLAMA_URL}: {exc}")
    st.info("Start Ollama, then use the Refresh models button to try again.")
    st.stop()

if not models:
    st.warning("No Ollama models are installed yet.")
    st.code("ollama pull llama3.2:3b")
    st.stop()

with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Choose a model", models)
    if st.button("Refresh models"):
        get_models.clear()
        st.rerun()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask me anything...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer: list[str] = []
        payload = json.dumps(
            {
                "model": selected_model,
                "messages": st.session_state.messages,
                "stream": True,
            }
        ).encode("utf-8")
        request = Request(
            f"{OLLAMA_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=600) as response:
                completed = False
                for line in response:
                    if not line.strip():
                        continue
                    chunk = json.loads(line.decode("utf-8"))
                    if not isinstance(chunk, dict):
                        raise ValueError("Ollama returned an invalid response.")
                    if chunk.get("error"):
                        raise RuntimeError(str(chunk["error"]))

                    message = chunk.get("message", {})
                    if not isinstance(message, dict):
                        raise ValueError("Ollama returned an invalid response.")
                    content = message.get("content", "")
                    if not isinstance(content, str):
                        raise ValueError("Ollama returned an invalid response.")
                    if content:
                        answer.append(content)
                        placeholder.markdown("".join(answer) + "...")
                    if chunk.get("done") is True:
                        completed = True
                        break

            if not completed:
                raise RuntimeError("Ollama ended the response before it was complete.")
            final_answer = "".join(answer)
            placeholder.markdown(final_answer or "_The model returned an empty response._")
            st.session_state.messages.append(
                {"role": "assistant", "content": final_answer}
            )
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            st.session_state.messages.pop()
            placeholder.error(f"Ollama returned HTTP {exc.code}: {detail or exc.reason}")
        except (URLError, TimeoutError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
            st.session_state.messages.pop()
            placeholder.error(f"Ollama API error: {exc}")
