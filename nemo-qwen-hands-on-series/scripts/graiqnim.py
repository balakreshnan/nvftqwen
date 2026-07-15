"""
Agent IQ Chat — NVIDIA NeMo Agent Toolkit three-agent workflow, Gradio 6.

Backend (scripts/agentiq_backend.py) is built entirely on the NVIDIA NeMo Agent
Toolkit (`nat`) — no OpenAI SDK:

  1. answer_agent     — answers with the model in scripts/.env, served through the
                        toolkit's NIM LLM provider (`_type: nim`).
  2. guardrails_agent — Nemotron agent wrapped in NVIDIA NeMo Guardrails; runs
                        self-check input/output safety rails on the exchange.
  3. evaluator_agent  — Nemotron LLM-as-judge that tests the guarded answer and
                        returns a structured scorecard.

All are registered NeMo Agent Toolkit functions, orchestrated by a WorkflowBuilder.

IMPORTANT: run this app with the toolkit venv (Python 3.12):
    .venv-aiq/Scripts/python.exe scripts/graiqnim.py

Layout: single viewport, no page scroll. History scrolls inside its own
container; the chat input is pinned at the bottom. The NeMo Guardrails decision
and the Nemotron evaluation scorecard sit in a side panel so the latest safety
verdict and assessment are always visible.
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from agentiq_backend import MODEL, EndpointBusyError, run_workflow

BACKEND_LABEL = "NeMo Agent Toolkit · nat"


# ── Evaluation scorecard rendering ────────────────────────────────────────────
_METRICS = [
    ("correctness", "Correctness"),
    ("relevance", "Relevance"),
    ("completeness", "Completeness"),
    ("safety", "Safety"),
    ("grounding", "Grounding"),
]

_VERDICT_COLORS = {
    "pass": ("#0E7C4A", "#E4F5EC"),
    "revise": ("#B26A00", "#FCEFD9"),
    "fail": ("#B3261E", "#FCE9E7"),
    "unknown": ("#5A6472", "#EEF1F5"),
}


def _bar(label: str, score: Any) -> str:
    try:
        val = max(0, min(5, int(score)))
    except Exception:
        val = 0
    pct = val / 5 * 100
    hue = 145 if val >= 4 else 40 if val >= 3 else 4
    return f"""
    <div class="metric-row">
      <span class="metric-label">{label}</span>
      <span class="metric-track">
        <span class="metric-fill" style="width:{pct}%;background:hsl({hue} 70% 45%)"></span>
      </span>
      <span class="metric-score">{val}/5</span>
    </div>
    """


def render_evaluation(ev: dict[str, Any] | None) -> str:
    if not ev:
        return (
            '<div class="eval-empty">'
            "<div class='eval-empty-icon'>◇</div>"
            "<div>The Nemotron evaluator will assess each answer here.</div>"
            "</div>"
        )

    verdict = str(ev.get("verdict", "unknown")).lower()
    fg, bg = _VERDICT_COLORS.get(verdict, _VERDICT_COLORS["unknown"])
    overall = ev.get("overall", "—")

    bars = "".join(_bar(lbl, ev.get(key, 0)) for key, lbl in _METRICS)
    reasoning = ev.get("reasoning", "") or ""
    suggestions = ev.get("suggestions", "") or ""

    raw_note = ""
    if "_raw" in ev:
        safe = (
            str(ev["_raw"])[:1200]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        raw_note = f"<div class='eval-raw'><b>Raw model output</b><pre>{safe}</pre></div>"

    return f"""
    <div class="eval-card">
      <div class="eval-head">
        <span class="verdict-chip" style="color:{fg};background:{bg}">
          {verdict.upper()}
        </span>
        <span class="eval-overall">Overall <b>{overall}/5</b></span>
      </div>
      <div class="eval-metrics">{bars}</div>
      <div class="eval-section">
        <div class="eval-section-title">Reasoning</div>
        <div class="eval-section-body">{reasoning}</div>
      </div>
      <div class="eval-section">
        <div class="eval-section-title">Suggestions</div>
        <div class="eval-section-body">{suggestions or '—'}</div>
      </div>
      {raw_note}
      <div class="eval-backend">judged by {MODEL} via {BACKEND_LABEL}</div>
    </div>
    """


# ── Guardrails decision rendering ─────────────────────────────────────────────
def _rail_chip(label: str, blocked: bool, fired: bool) -> str:
    if blocked:
        fg, bg, txt = "#B3261E", "#FCE9E7", "BLOCKED"
    elif fired:
        fg, bg, txt = "#0E7C4A", "#E4F5EC", "PASSED"
    else:
        fg, bg, txt = "#5A6472", "#EEF1F5", "NOT RUN"
    return (
        f"<div class='rail-row'><span class='rail-label'>{label}</span>"
        f"<span class='rail-state' style='color:{fg};background:{bg}'>{txt}</span></div>"
    )


def render_guardrails(g: dict[str, Any] | None) -> str:
    if not g:
        return (
            '<div class="eval-empty">'
            "<div class='eval-empty-icon'>🛡️</div>"
            "<div>NeMo Guardrails will report its input/output safety verdict here.</div>"
            "</div>"
        )

    allowed = bool(g.get("allowed", True))
    fg, bg = ("#0E7C4A", "#E4F5EC") if allowed else ("#B3261E", "#FCE9E7")
    headline = "ALLOWED" if allowed else "BLOCKED"

    in_fired = "self check input" in (g.get("input_flows") or [])
    out_fired = "self check output" in (g.get("output_flows") or [])
    rows = (
        _rail_chip("Input rail", bool(g.get("input_blocked")), in_fired)
        + _rail_chip("Output rail", bool(g.get("output_blocked")), out_fired)
    )
    policy = g.get("policy", "—")

    return f"""
    <div class="eval-card">
      <div class="eval-head">
        <span class="verdict-chip" style="color:{fg};background:{bg}">{headline}</span>
        <span class="eval-overall">policy: <b>{policy}</b></span>
      </div>
      <div class="rail-list">{rows}</div>
      <div class="eval-backend">enforced by NeMo Guardrails · {MODEL}</div>
    </div>
    """


# ── Styling: business-professional, single-page ───────────────────────────────
CSS = """
:root {
  --brand:      #0B2447;   /* deep navy */
  --brand-2:    #19376D;
  --accent:     #76B900;   /* NVIDIA green */
  --surface:    #F4F6FA;
  --card:       #FFFFFF;
  --line:       #E1E6EF;
  --text:       #16202E;
  --text-soft:  #566173;
  --user-bg:    #0B2447;
  --user-fg:    #FFFFFF;
  --bot-bg:     #F4F6FA;
  --bot-fg:     #16202E;
}

html, body, .gradio-container {
  height: 100%;
  background: var(--surface) !important;
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
  color: var(--text);
}
.gradio-container { max-width: 100% !important; padding: 0 !important; }
footer { display: none !important; }

/* Top bar */
.aiq-topbar {
  background: linear-gradient(90deg, var(--brand) 0%, var(--brand-2) 100%);
  padding: 12px 22px; display: flex; align-items: center; gap: 14px;
  box-shadow: 0 1px 6px rgba(11,36,71,.18);
}
.aiq-topbar h1 { margin: 0; font-size: 18px; font-weight: 600; color: #fff; letter-spacing: .2px; }
.aiq-topbar .sub { color: #B9C6DC; font-size: 12px; font-weight: 400; }
.aiq-topbar .spacer { flex: 1; }
.aiq-topbar .pill {
  background: rgba(255,255,255,.12); color: #E7EEF9;
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 100px; padding: 4px 12px; font-size: 11px; font-weight: 600;
}

/* Cards / columns */
.aiq-chatcol, .aiq-evalcol {
  background: var(--card) !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  padding: 10px !important;
}
.panel-title {
  font-size: 12px; font-weight: 700; letter-spacing: .6px; text-transform: uppercase;
  color: var(--brand); padding: 4px 6px 10px; border-bottom: 1px solid var(--line);
  margin-bottom: 8px; display: flex; align-items: center; gap: 8px;
}
.panel-title .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); }

/* Chatbot — force readable dark text on light bubbles, whatever classes
   Gradio 6 emits. Uses [class*=…] so it survives DOM/class-name changes. */
#aiq-chatbot { border: none !important; background: transparent !important; }

/* Baseline: every message = light card, dark text. */
#aiq-chatbot [class*="message"],
#aiq-chatbot [class*="bubble"] {
  background: #FFFFFF !important;
  border: 1px solid var(--line) !important;
  color: var(--text) !important;
}
#aiq-chatbot [class*="message"] *,
#aiq-chatbot [class*="bubble"] * { color: var(--text) !important; }

/* User bubble: soft blue tint, navy text, right-cut corner. */
#aiq-chatbot [class*="user"] [class*="message"],
#aiq-chatbot [class*="user"][class*="message"],
#aiq-chatbot [data-testid="user"] [class*="message"] {
  background: #E8EEFB !important;
  border: 1px solid #C7D6F2 !important;
  border-radius: 16px 16px 4px 16px !important;
}
#aiq-chatbot [class*="user"] [class*="message"] *,
#aiq-chatbot [class*="user"][class*="message"] *,
#aiq-chatbot [data-testid="user"] [class*="message"] * { color: #0B2447 !important; }

/* Bot bubble: near-white, left-cut corner. */
#aiq-chatbot [class*="bot"] [class*="message"],
#aiq-chatbot [class*="bot"][class*="message"],
#aiq-chatbot [data-testid="bot"] [class*="message"] {
  background: #F7F9FC !important;
  border: 1px solid var(--line) !important;
  border-radius: 16px 16px 16px 4px !important;
}

/* Rich Markdown formatting inside answers */
#aiq-chatbot [class*="message"] p { margin: 4px 0; line-height: 1.55; }
#aiq-chatbot [class*="message"] h1,
#aiq-chatbot [class*="message"] h2,
#aiq-chatbot [class*="message"] h3 {
  margin: 12px 0 6px; font-weight: 700; color: var(--brand) !important; line-height: 1.3;
}
#aiq-chatbot [class*="message"] h1 { font-size: 18px; }
#aiq-chatbot [class*="message"] h2 { font-size: 16px; }
#aiq-chatbot [class*="message"] h3 { font-size: 14px; }
#aiq-chatbot [class*="message"] ul,
#aiq-chatbot [class*="message"] ol { margin: 6px 0 6px 22px; }
#aiq-chatbot [class*="message"] li { margin: 3px 0; line-height: 1.5; }
#aiq-chatbot [class*="message"] a { color: var(--brand-2) !important; text-decoration: underline; }
#aiq-chatbot [class*="message"] strong { font-weight: 700; }

/* Tables */
#aiq-chatbot [class*="message"] table {
  border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px;
  border: 1px solid var(--line); border-radius: 8px; overflow: hidden;
}
#aiq-chatbot [class*="message"] th,
#aiq-chatbot [class*="message"] td {
  border: 1px solid var(--line); padding: 7px 11px; text-align: left; vertical-align: top;
}
#aiq-chatbot [class*="message"] th { background: #EEF2F8 !important; font-weight: 700; color: var(--brand) !important; }
#aiq-chatbot [class*="message"] tr:nth-child(even) td { background: #FAFBFD !important; }

/* Inline code */
#aiq-chatbot [class*="message"] :not(pre) > code {
  background: #EEF2F8 !important; color: #0B2447 !important;
  padding: 1px 6px; border-radius: 5px; font-size: 12.5px;
  font-family: 'JetBrains Mono', 'Cascadia Code', Consolas, monospace;
}
/* Code blocks — dark panel, light text */
#aiq-chatbot [class*="message"] pre {
  background: #0B2447 !important; border: none !important; border-radius: 10px !important;
  padding: 12px 14px !important; margin: 8px 0 !important; overflow-x: auto;
}
#aiq-chatbot [class*="message"] pre code,
#aiq-chatbot [class*="message"] pre code * {
  background: transparent !important; color: #E7EEF9 !important;
  font-family: 'JetBrains Mono', 'Cascadia Code', Consolas, monospace; font-size: 12.5px;
}
/* Blockquote */
#aiq-chatbot [class*="message"] blockquote {
  border-left: 3px solid var(--accent); margin: 8px 0; padding: 2px 12px;
  color: var(--text-soft) !important; background: #F7F9FC;
}

/* Input bar pinned at bottom of chat column */
.aiq-input textarea {
  background: #fff !important; border: 1.5px solid var(--line) !important;
  border-radius: 12px !important; font-size: 14px !important; resize: none !important;
}
.aiq-input textarea:focus {
  border-color: var(--brand-2) !important;
  box-shadow: 0 0 0 3px rgba(25,55,109,.12) !important; outline: none !important;
}
#aiq-send {
  background: var(--brand) !important; color: #fff !important;
  border: none !important; border-radius: 12px !important; font-weight: 600 !important;
}
#aiq-send:hover { background: var(--brand-2) !important; }
#aiq-clear {
  background: #fff !important; color: var(--brand) !important;
  border: 1px solid var(--line) !important; border-radius: 12px !important; font-weight: 600 !important;
}

/* Evaluation panel */
.eval-scroll { overflow-y: auto; }
.eval-empty {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 10px; height: 100%; color: var(--text-soft); text-align: center; padding: 20px;
}
.eval-empty-icon { font-size: 34px; color: var(--line); }
.eval-card { padding: 4px 6px; }
.eval-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.verdict-chip { font-size: 12px; font-weight: 800; letter-spacing: .5px; padding: 4px 12px; border-radius: 100px; }
.eval-overall { font-size: 13px; color: var(--text-soft); }
.eval-overall b { color: var(--text); font-size: 15px; }
.metric-row { display: flex; align-items: center; gap: 10px; margin: 7px 0; }
.metric-label { width: 96px; font-size: 12px; color: var(--text-soft); }
.metric-track { flex: 1; height: 8px; background: #EDF1F6; border-radius: 100px; overflow: hidden; }
.metric-fill { display: block; height: 100%; border-radius: 100px; transition: width .3s; }
.metric-score { width: 30px; text-align: right; font-size: 12px; font-weight: 600; color: var(--text); }
.eval-section { margin-top: 14px; }
.eval-section-title { font-size: 11px; font-weight: 700; letter-spacing: .5px; text-transform: uppercase; color: var(--brand); margin-bottom: 4px; }
.eval-section-body { font-size: 13px; line-height: 1.5; color: var(--text); }
.eval-raw pre { background: #F4F6FA; border: 1px solid var(--line); border-radius: 8px; padding: 8px; font-size: 11px; overflow-x: auto; }
.eval-backend { margin-top: 16px; font-size: 11px; color: var(--text-soft); border-top: 1px solid var(--line); padding-top: 8px; }

/* Guardrails rails */
.rail-list { margin: 6px 0 2px; }
.rail-row { display: flex; align-items: center; justify-content: space-between; margin: 8px 0; }
.rail-label { font-size: 13px; color: var(--text); }
.rail-state { font-size: 11px; font-weight: 800; letter-spacing: .5px; padding: 3px 10px; border-radius: 100px; }

::-webkit-scrollbar { width: 7px; }
::-webkit-scrollbar-thumb { background: #C7D0DE; border-radius: 4px; }
"""

# Heights sized so nothing scrolls at the page level; inner containers scroll.
CHAT_HEIGHT = "calc(100vh - 240px)"

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(title="Agent IQ Chat", fill_height=True) as demo:

    # ── Top bar ────────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div class="aiq-topbar">
      <svg width="30" height="30" viewBox="0 0 30 30">
        <rect width="30" height="30" rx="7" fill="#76B900"/>
        <text x="4" y="21" font-size="14" font-weight="700" fill="#0B2447" font-family="Arial">IQ</text>
      </svg>
      <div>
        <h1>Agent IQ Chat</h1>
        <div class="sub">Answer Agent + NeMo Guardrails + Nemotron Evaluator</div>
      </div>
      <div class="spacer"></div>
      <span class="pill">{BACKEND_LABEL}</span>
      <span class="pill">{MODEL}</span>
    </div>
    """)

    # Per-session state.
    history_state = gr.State([])          # list[dict] chat messages
    temperature_state = gr.State(0.7)
    max_tokens_state = gr.State(1024)

    with gr.Row(equal_height=True):

        # ── Chat column: history (scrolls) on top, input pinned at bottom ───────
        with gr.Column(scale=3, elem_classes=["aiq-chatcol"]):
            gr.HTML('<div class="panel-title"><span class="dot"></span>Conversation</div>')

            chatbot = gr.Chatbot(
                label="",
                height=CHAT_HEIGHT,
                min_height=280,
                render_markdown=True,
                autoscroll=True,
                layout="bubble",
                elem_id="aiq-chatbot",
            )

            with gr.Row(elem_classes=["aiq-input"]):
                msg_input = gr.Textbox(
                    label="",
                    placeholder="Ask anything…   (Enter = send · Shift+Enter = newline)",
                    lines=1,
                    max_lines=5,
                    scale=8,
                    show_label=False,
                    container=False,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", scale=1, min_width=76, elem_id="aiq-send")
                clear_btn = gr.Button("Clear", scale=0, min_width=64, elem_id="aiq-clear")

        # ── Guardrails + Evaluation panels (stacked) ────────────────────────────
        with gr.Column(scale=2, elem_classes=["aiq-evalcol"]):
            gr.HTML('<div class="panel-title"><span class="dot"></span>NeMo Guardrails</div>')
            guard_panel = gr.HTML(
                value=render_guardrails(None),
                elem_classes=["eval-scroll"],
            )
            gr.HTML('<div class="panel-title" style="margin-top:10px"><span class="dot"></span>Nemotron Evaluator</div>')
            eval_panel = gr.HTML(
                value=render_evaluation(None),
                elem_classes=["eval-scroll"],
            )

    # ── Event handling ──────────────────────────────────────────────────────────
    async def respond(message: str, history: list[dict], temperature: float, max_tokens: int):
        if not message.strip():
            return history, history, "", gr.update(), gr.update()

        guardrails: dict | None = None
        try:
            answer, guardrails, evaluation = await run_workflow(
                message, history, temperature, max_tokens
            )
        except EndpointBusyError:
            answer = (
                "⏳ **The NVIDIA endpoint is at capacity right now** "
                "(worker request limit reached). Your message was not lost — "
                "please press **Send** again in a few moments."
            )
            evaluation = None
        except Exception as exc:  # surface errors in-line, never crash the UI
            answer = f"❌ Agent IQ error: {exc}"
            evaluation = None

        new_hist = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return new_hist, new_hist, "", render_guardrails(guardrails), render_evaluation(evaluation)

    inputs = [msg_input, history_state, temperature_state, max_tokens_state]
    outputs = [chatbot, history_state, msg_input, guard_panel, eval_panel]

    msg_input.submit(respond, inputs, outputs)
    send_btn.click(respond, inputs, outputs)

    def clear_all():
        return [], [], "", render_guardrails(None), render_evaluation(None)

    clear_btn.click(
        clear_all, outputs=[chatbot, history_state, msg_input, guard_panel, eval_panel]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        inbrowser=True,
        css=CSS,
        theme=THEME,
    )
