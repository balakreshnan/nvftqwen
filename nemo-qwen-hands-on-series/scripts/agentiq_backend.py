"""
Agent IQ backend — two agents built on the NVIDIA NeMo Agent Toolkit (`nat`).

Agents (registered as NeMo Agent Toolkit functions):
  1. answer_agent     — answers the user with the model from scripts/.env, served
                        through the toolkit's NIM LLM provider (`_type: nim`).
  2. evaluator_agent  — Nemotron LLM-as-judge that tests the answer and returns a
                        structured scorecard.

Both run through a `WorkflowBuilder`: the toolkit constructs the NIM LLM, wraps it
for LangChain, and invokes each agent via the `Function` interface. No OpenAI SDK.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.runtime.loader import PluginTypes, discover_and_register_plugins

# Load toolkit plugins (registers the NIM LLM provider and its LangChain client).
discover_and_register_plugins(PluginTypes.COMPONENT)

# ── Configuration (everything from .env) ──────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

API_KEY = os.getenv("NVIDIA_API_KEY", "")
ENDPOINT = os.getenv("NVIDIA_ENDPOINT", "https://integrate.api.nvidia.com/v1")
MODEL = os.getenv("NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-ultra-253b-v1")

LLM_NAME = "nim_llm"

ANSWER_SYSTEM = (
    "You are a knowledgeable, precise NVIDIA AI assistant. "
    "Answer the user's question directly and completely. Be accurate, state "
    "assumptions, and prefer concrete detail. Use Markdown when it helps."
)

EVAL_SYSTEM = (
    "You are a rigorous evaluation agent (Nemotron LLM-as-judge). "
    "You receive a USER QUESTION and a candidate ANSWER from another agent. "
    "Critically assess the answer and respond with ONLY a JSON object (no prose, "
    "no code fences) with exactly these keys:\n"
    '  "correctness": integer 1-5,\n'
    '  "relevance": integer 1-5,\n'
    '  "completeness": integer 1-5,\n'
    '  "safety": integer 1-5,\n'
    '  "grounding": integer 1-5,\n'
    '  "overall": integer 1-5,\n'
    '  "verdict": one of "pass" | "revise" | "fail",\n'
    '  "reasoning": short string (2-4 sentences),\n'
    '  "suggestions": short string with concrete improvement advice.\n'
    "Scores: 1 = poor, 5 = excellent. Be strict and specific."
)


# ── Agent 1: answer_agent ─────────────────────────────────────────────────────
class AnswerAgentConfig(FunctionBaseConfig, name="answer_agent"):
    """Answers the user with the environment-defined NVIDIA model."""

    llm_name: str = LLM_NAME


@register_function(config_type=AnswerAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def answer_agent(config: AnswerAgentConfig, builder: Builder):
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _run(question: str) -> str:
        resp = await llm.ainvoke(
            [SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=question)]
        )
        return getattr(resp, "content", str(resp)).strip()

    yield FunctionInfo.from_fn(
        _run, description="Answers the user's question using the NVIDIA model from .env."
    )


# ── Agent 2: evaluator_agent (Nemotron LLM-as-judge) ──────────────────────────
class EvaluatorAgentConfig(FunctionBaseConfig, name="evaluator_agent"):
    """Nemotron evaluator that tests an answer and returns a structured verdict."""

    llm_name: str = LLM_NAME


@register_function(config_type=EvaluatorAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def evaluator_agent(config: EvaluatorAgentConfig, builder: Builder):
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _run(payload: str) -> str:
        try:
            data = json.loads(payload)
            question, answer = data["question"], data["answer"]
        except Exception:
            question, answer = "", payload
        user = (
            f"USER QUESTION:\n{question}\n\n"
            f"CANDIDATE ANSWER:\n{answer}\n\n"
            "Evaluate the candidate answer now."
        )
        resp = await llm.ainvoke(
            [SystemMessage(content=EVAL_SYSTEM), HumanMessage(content=user)]
        )
        return getattr(resp, "content", str(resp)).strip()

    yield FunctionInfo.from_fn(
        _run, description="Evaluates a candidate answer and returns a JSON scorecard."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _compose_prompt(question: str, history: list[dict]) -> str:
    """Fold prior turns into a single prompt so the answer agent stays stateless."""
    if not history:
        return question
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    lines.append(f"User: {question}")
    return "\n".join(lines)


def parse_evaluation(raw: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {
        "verdict": "unknown",
        "reasoning": "Evaluator did not return parseable JSON.",
        "suggestions": "",
        "_raw": raw,
    }


def _nim_config(temperature: float, max_tokens: int) -> NIMModelConfig:
    return NIMModelConfig(
        base_url=ENDPOINT,
        api_key=API_KEY,
        model_name=MODEL,
        temperature=float(temperature),
        top_p=1.0,
        max_tokens=int(max_tokens),
        num_retries=8,  # toolkit already retries 429/500/502/503/504
    )


# Errors that mean "endpoint temporarily busy" — safe to retry.
_TRANSIENT = ("503", "resourceexhausted", "request limit", "too many requests", "429", "service unavailable")


class EndpointBusyError(Exception):
    """Raised when the NVIDIA endpoint is saturated after all retries."""


def _is_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(sig in text for sig in _TRANSIENT)


async def _with_backoff(coro_factory, attempts: int = 4, base_delay: float = 2.0):
    """Retry an async call with exponential backoff on transient endpoint errors."""
    import asyncio

    last: Exception | None = None
    for i in range(attempts):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001
            if not _is_transient(exc):
                raise
            last = exc
            if i < attempts - 1:
                await asyncio.sleep(base_delay * (2**i))  # 2s, 4s, 8s
    raise EndpointBusyError(
        "The NVIDIA endpoint is at capacity (worker request limit reached). "
        "Please retry in a few moments."
    ) from last


# ── Orchestration ─────────────────────────────────────────────────────────────
async def run_workflow(
    question: str,
    history: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> tuple[str, dict[str, Any]]:
    """Build the toolkit workflow, run answer_agent then evaluator_agent."""
    async with WorkflowBuilder() as builder:
        await builder.add_llm(LLM_NAME, _nim_config(temperature, max_tokens))
        answer_fn = await builder.add_function("answer_agent", AnswerAgentConfig())
        eval_fn = await builder.add_function("evaluator_agent", EvaluatorAgentConfig())

        prompt = _compose_prompt(question, history)
        answer = await _with_backoff(lambda: answer_fn.ainvoke(prompt, to_type=str))

        eval_payload = json.dumps({"question": question, "answer": answer})
        eval_raw = await _with_backoff(lambda: eval_fn.ainvoke(eval_payload, to_type=str))

    return answer, parse_evaluation(eval_raw)


if __name__ == "__main__":
    import asyncio
    import sys

    # Windows consoles default to cp1252; force UTF-8 so model output prints cleanly.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ans, ev = asyncio.run(run_workflow("What is CUDA in one sentence?", []))
    print("ANSWER:\n", ans, "\n")
    print("EVAL:\n", json.dumps(ev, indent=2, ensure_ascii=False))
