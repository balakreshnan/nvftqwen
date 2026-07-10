#!/usr/bin/env python3
"""Run a local chat completion with a small Qwen instruct model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROMPT = "Explain LoRA in two short sentences."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download (on first use) and run a Qwen instruct model locally with "
            "Hugging Face Transformers."
        )
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
        help=f"User prompt (default: {DEFAULT_PROMPT!r})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model ID or local model directory (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a concise and helpful assistant.",
        help="System message supplied to the chat template",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Inference device; auto prefers CUDA, then Apple MPS, then CPU",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Model data type; auto uses float32 on CPU and reduced precision on GPUs",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of response tokens (default: 128)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; zero uses deterministic greedy decoding",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus-sampling probability when temperature is above zero",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling (default: 42)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face download/cache directory",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached/local model files only; do not access Hugging Face",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow model repository Python code. Enable only for a trusted model.",
    )
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be greater than 0 and no more than 1")
    return args


def select_device(torch: Any, requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but PyTorch cannot access Apple MPS.")
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(torch: Any, device: str, requested: str) -> Any:
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if requested != "auto":
        if device == "cpu" and requested == "float16":
            raise RuntimeError(
                "float16 is not a safe CPU default. Use --dtype float32 or bfloat16."
            )
        return dtypes[requested]
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def main() -> int:
    args = parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(
            "Missing local-inference dependencies. Install them with:\n"
            "  python -m pip install -r requirements-local.txt",
            file=sys.stderr,
        )
        print(f"Import detail: {exc}", file=sys.stderr)
        return 2

    try:
        device = select_device(torch, args.device)
        dtype = select_dtype(torch, device, args.dtype)
    except RuntimeError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    common_kwargs: dict[str, Any] = {
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else None,
        "local_files_only": args.offline,
        "trust_remote_code": args.trust_remote_code,
    }
    # Do not send an explicit null cache directory to older Transformers releases.
    common_kwargs = {key: value for key, value in common_kwargs.items() if value is not None}

    print(f"Model:  {args.model}")
    print(f"Device: {device}")
    print(f"Dtype:  {dtype}")
    if not args.offline:
        print("The first run may download model and tokenizer files from Hugging Face.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, **common_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **common_kwargs,
        )
        model.to(device)
        model.eval()
    except OSError as exc:
        print(f"Could not load model {args.model!r}: {exc}", file=sys.stderr)
        if args.offline:
            print("Offline mode requires the complete model to be cached locally.", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Model loading failed: {exc}", file=sys.stderr)
        print("Try --device cpu, a smaller model, or free additional RAM/VRAM.", file=sys.stderr)
        return 1

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([rendered_prompt], return_tensors="pt").to(device)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    try:
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)
    except RuntimeError as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        if "out of memory" in str(exc).lower():
            print(
                "Reduce --max-new-tokens, close other applications, or use a smaller model.",
                file=sys.stderr,
            )
        return 1

    new_tokens = generated[0, inputs.input_ids.shape[1] :]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print("\nResponse:\n")
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
