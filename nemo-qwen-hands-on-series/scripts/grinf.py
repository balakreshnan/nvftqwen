"""
NVIDIA API Chat UI — Material Design 3 Light, Gradio 6.
Reads credentials from scripts/.env (NVIDIA_API_KEY, NVIDIA_ENDPOINT, NVIDIA_MODEL).
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv(Path(__file__).parent / ".env")

API_KEY      = os.getenv("NVIDIA_API_KEY", "")
ENDPOINT     = os.getenv("NVIDIA_ENDPOINT", "https://integrate.api.nvidia.com/v1")
DEFAULT_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-ultra-253b-v1")

_seen: set = set()
MODELS = [
    m for m in [
        DEFAULT_MODEL,
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "nvidia/llama-3.3-nemotron-super-49b-v1",
        "meta/llama-3.1-405b-instruct",
        "meta/llama-3.3-70b-instruct",
        "mistralai/mixtral-8x22b-instruct-v0.1",
        "google/gemma-2-27b-it",
    ]
    if not (_seen.__contains__(m) or _seen.add(m))  # type: ignore[func-returns-value]
]


def build_client(api_key: str, endpoint: str) -> OpenAI:
    return OpenAI(base_url=endpoint, api_key=api_key)


def chat(
    message: str,
    history: list,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    endpoint: str,
) -> str:
    if not message.strip():
        return ""
    if not api_key.strip():
        return "⚠️  Please enter your NVIDIA API key in the Settings panel."

    messages: list[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    try:
        client = build_client(api_key.strip(), endpoint.strip())
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=int(max_tokens),
        )
        return resp.choices[0].message.content
    except Exception as exc:
        return f"❌  Error: {exc}"


# ── Colours only — no layout overrides ────────────────────────────────────────
# Gradio 6 manages its own flex/grid layout; we only paint it here.
CSS = """
/* MD3 Light colour tokens */
:root {
  --md-primary:    #6750A4;
  --md-on-primary: #FFFFFF;
  --md-primary-c:  #EADDFF;
  --md-surface:    #FFFBFE;
  --md-surf-c:     #F3EDF7;
  --md-surf-ch:    #ECE6F0;
  --md-surf-cx:    #E6E0E9;
  --md-on-surf:    #1C1B1F;
  --md-on-surf-v:  #49454F;
  --md-outline:    #79747E;
  --md-outline-v:  #CAC4D0;
}

/* Page background */
body, .gradio-container {
  background: var(--md-surface) !important;
  font-family: 'Google Sans', Roboto, system-ui, sans-serif !important;
}

/* Top bar */
.nv-topbar {
  background: var(--md-surf-c);
  border-bottom: 1px solid var(--md-outline-v);
  padding: 10px 20px;
  display: flex; align-items: center; gap: 12px;
}
.nv-topbar h1 {
  margin: 0; font-size: 20px; font-weight: 500;
  color: var(--md-on-surf);
}
.nv-topbar .badge {
  background: var(--md-primary); color: var(--md-on-primary);
  border-radius: 6px; padding: 2px 8px;
  font-size: 11px; font-weight: 600;
}

/* Rail (left settings panel) */
.nv-rail {
  background: var(--md-surf-c) !important;
  border-right: 1px solid var(--md-outline-v) !important;
  padding: 12px !important;
}
.rail-hdr {
  color: var(--md-primary);
  font-size: 11px; font-weight: 600;
  letter-spacing: .8px; text-transform: uppercase;
  padding: 8px 2px 4px;
  border-bottom: 1px solid var(--md-outline-v);
  margin-bottom: 6px;
}

/* Inputs and selects inside rail */
.nv-rail textarea,
.nv-rail input[type=text],
.nv-rail input[type=password] {
  background: var(--md-surface) !important;
  border: 1px solid var(--md-outline-v) !important;
  border-radius: 8px !important;
  color: var(--md-on-surf) !important;
  font-size: 13px !important;
}
.nv-rail label > span {
  color: var(--md-on-surf-v) !important;
  font-size: 11px !important;
}
input[type=range] { accent-color: var(--md-primary); }

/* Chat column background */
.nv-chat {
  background: var(--md-surface) !important;
}

/* Chatbot scrollable area — Gradio 6 honours the height param natively */
#nv-chatbot {
  border: none !important;
  border-radius: 12px !important;
  background: var(--md-surface) !important;
}

/* Message bubbles — Gradio 6 class names */
#nv-chatbot .message-wrap .user > .message {
  background: var(--md-primary-c) !important;
  color: var(--md-on-surf) !important;
  border-radius: 18px 18px 4px 18px !important;
  border: none !important;
}
#nv-chatbot .message-wrap .bot > .message {
  background: var(--md-surf-ch) !important;
  color: var(--md-on-surf) !important;
  border-radius: 18px 18px 18px 4px !important;
  border: 1px solid var(--md-outline-v) !important;
}

/* Input bar */
.nv-input-bar {
  background: var(--md-surf-c) !important;
  border-top: 1px solid var(--md-outline-v) !important;
  padding: 8px 12px !important;
  border-radius: 0 !important;
}
.nv-input-bar textarea {
  background: var(--md-surface) !important;
  border: 1.5px solid var(--md-outline) !important;
  border-radius: 12px !important;
  color: var(--md-on-surf) !important;
  font-size: 14px !important;
  resize: none !important;
}
.nv-input-bar textarea:focus {
  border-color: var(--md-primary) !important;
  box-shadow: 0 0 0 2px rgba(103,80,164,.15) !important;
  outline: none !important;
}

/* Buttons */
#nv-send {
  background: var(--md-primary) !important;
  color: var(--md-on-primary) !important;
  border-radius: 100px !important;
  border: none !important;
  font-weight: 500 !important;
}
#nv-send:hover { filter: brightness(0.9); }

#nv-clear {
  background: var(--md-primary-c) !important;
  color: var(--md-primary) !important;
  border-radius: 100px !important;
  border: none !important;
  font-weight: 500 !important;
  width: 100% !important;
}

/* Thin scrollbars */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb {
  background: var(--md-outline-v); border-radius: 3px;
}
"""

# Chatbot height: leaves ~180 px for topbar + input row so nothing scrolls at page level
CHAT_HEIGHT = "calc(100vh - 190px)"

with gr.Blocks(
    css=CSS,
    title="NVIDIA Chat",
    fill_height=True,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Roboto"),
    ),
) as demo:

    # ── Top App Bar (pure HTML, always at the very top) ────────────────────────
    gr.HTML("""
    <div class="nv-topbar">
      <svg width="30" height="30" viewBox="0 0 30 30">
        <rect width="30" height="30" rx="6" fill="#76B900"/>
        <text x="4" y="21" font-size="14" font-weight="700"
              fill="white" font-family="Arial">NV</text>
      </svg>
      <h1>NVIDIA API Chat</h1>
      <span class="badge">NIM</span>
    </div>
    """)

    # ── Two-column body: rail on left, chat on right ───────────────────────────
    with gr.Row():

        # ── Settings Rail ──────────────────────────────────────────────────────
        with gr.Column(scale=0, min_width=260, elem_classes=["nv-rail"]):

            gr.HTML('<div class="rail-hdr">Connection</div>')
            api_key_input = gr.Textbox(
                label="API Key", value=API_KEY,
                type="password", placeholder="nvapi-…", lines=1,
            )
            endpoint_input = gr.Textbox(
                label="Endpoint URL", value=ENDPOINT, lines=1,
            )

            gr.HTML('<div class="rail-hdr">Model</div>')
            model_dropdown = gr.Dropdown(
                choices=MODELS, value=MODELS[0],
                label="Model", allow_custom_value=True,
            )

            gr.HTML('<div class="rail-hdr">Parameters</div>')
            temperature_slider = gr.Slider(
                minimum=0.0, maximum=2.0, value=0.7, step=0.05,
                label="Temperature",
            )
            max_tokens_slider = gr.Slider(
                minimum=64, maximum=4096, value=1024, step=64,
                label="Max Tokens",
            )

            gr.HTML('<div class="rail-hdr">System Prompt</div>')
            system_prompt_input = gr.Textbox(
                label="", lines=5,
                value="You are a helpful NVIDIA AI assistant.",
                placeholder="Optional system instructions…",
            )

            clear_btn = gr.Button(
                "🗑  Clear Conversation",
                variant="secondary",
                elem_id="nv-clear",
            )

        # ── Chat Column ────────────────────────────────────────────────────────
        # Declaration order is render order:
        #   1. chatbot  →  takes up most of the height, scrolls internally
        #   2. input row → pinned at the natural bottom of the column
        with gr.Column(scale=1, elem_classes=["nv-chat"]):

            chatbot = gr.Chatbot(
                label="",
                height=CHAT_HEIGHT,          # fills viewport minus topbar+input
                min_height=300,
                render_markdown=True,
                autoscroll=True,
                layout="bubble",
                elem_id="nv-chatbot",
            )

            with gr.Row(elem_classes=["nv-input-bar"]):
                msg_input = gr.Textbox(
                    label="",
                    placeholder="Message NVIDIA AI…   (Enter = send · Shift+Enter = newline)",
                    lines=1,
                    max_lines=5,
                    scale=9,
                    show_label=False,
                    container=False,
                    autofocus=True,
                )
                send_btn = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    min_width=80,
                    elem_id="nv-send",
                )

    # ── Event wiring ───────────────────────────────────────────────────────────

    def respond(message, history, model, system_prompt, temperature, max_tokens, api_key, endpoint):
        if not message.strip():
            return history, ""
        reply = chat(message, history, model, system_prompt,
                     temperature, max_tokens, api_key, endpoint)
        return history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": reply},
        ], ""

    inputs = [
        msg_input, chatbot, model_dropdown, system_prompt_input,
        temperature_slider, max_tokens_slider, api_key_input, endpoint_input,
    ]

    msg_input.submit(respond, inputs, [chatbot, msg_input])
    send_btn.click(respond, inputs, [chatbot, msg_input])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
    )
