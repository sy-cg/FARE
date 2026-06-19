# -*- coding: utf-8 -*-
"""
scripts/run_baselines.py

Unified baseline runner for ID-only sequential recommendation backbones.

Supported models:
- --model sasrec   -> SASRec-ID
- --model gru4rec  -> GRU4Rec-ID
- --model bert4rec -> BERT4Rec-style ID

This file is intentionally a thin compatibility wrapper around
scripts/run_id_backbone.py. The goal is to keep old commands working while
adding --model support.

Old SASRec command remains valid:
    python scripts/run_baselines.py \
      --dataset Video_Games \
      --config configs/sasrec_3090.yaml \
      --batch_size 256 \
      --eval_batch_size 512

New commands:
    python scripts/run_baselines.py \
      --model gru4rec \
      --dataset Video_Games \
      --config configs/gru4rec_3090.yaml

    python scripts/run_baselines.py \
      --model bert4rec \
      --dataset Video_Games \
      --config configs/bert4rec_3090.yaml

Why wrapper instead of duplicated training code?
- run_id_backbone.py already implements the shared data loading, training,
  full-sort evaluation, topk export, checkpointing, and metrics summary logic.
- Keeping run_baselines.py as a compatibility entrypoint avoids silent drift
  between SASRec / GRU4Rec / BERT4Rec training protocols.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


SUPPORTED_MODELS = {"sasrec", "gru4rec", "bert4rec"}
DEFAULT_CONFIG_BY_MODEL = {
    "sasrec": "configs/sasrec_3090.yaml",
    "gru4rec": "configs/gru4rec_3090.yaml",
    "bert4rec": "configs/bert4rec_3090.yaml",
}


def _extract_arg(argv: List[str], name: str) -> Optional[str]:
    """Extract value of an argparse-style option without consuming argv."""
    prefix = name + "="
    for i, token in enumerate(argv):
        if token == name and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _has_arg(argv: List[str], name: str) -> bool:
    prefix = name + "="
    return any(token == name or token.startswith(prefix) for token in argv)


def _normalize_legacy_args(argv: List[str]) -> List[str]:
    """Normalize old run_baselines.py CLI into run_id_backbone.py CLI.

    Old scripts did not have --model and implicitly meant SASRec-ID.
    This wrapper injects --model sasrec when missing.
    It also injects a default config if --config is omitted.
    """
    out = list(argv)

    model = _extract_arg(out, "--model")
    if model is None:
        model = "sasrec"
        out = ["--model", model] + out
    else:
        model = model.lower().strip()
        if model not in SUPPORTED_MODELS:
            raise SystemExit(f"Unsupported --model {model!r}. Choose from: {sorted(SUPPORTED_MODELS)}")

    if not _has_arg(out, "--config"):
        out.extend(["--config", DEFAULT_CONFIG_BY_MODEL[model]])

    return out


def _print_help() -> None:
    text = f"""
Unified baseline runner.

Supported models:
  --model sasrec    SASRec-ID baseline, default for backward compatibility
  --model gru4rec   GRU4Rec-ID baseline
  --model bert4rec  BERT4Rec-style ID baseline

Examples:
  python scripts/run_baselines.py --dataset Video_Games --config configs/sasrec_3090.yaml

  python scripts/run_baselines.py \\
    --model gru4rec \\
    --dataset Video_Games \\
    --config configs/gru4rec_3090.yaml

  python scripts/run_baselines.py \\
    --model bert4rec \\
    --dataset Video_Games \\
    --config configs/bert4rec_3090.yaml

All other training/evaluation arguments are forwarded to scripts/run_id_backbone.py,
including:
  --run_id, --epochs, --batch_size, --eval_batch_size, --lr, --weight_decay,
  --max_seq_len, --hidden_size, --num_layers, --num_heads, --dropout,
  --patience, --eval_every, --num_workers, --cpu
"""
    print(text.strip())


def main() -> None:
    raw_argv = sys.argv[1:]

    if any(x in raw_argv for x in ["-h", "--help"]):
        _print_help()
        print("\nForwarded runner help:\n")
        try:
            import run_id_backbone  # type: ignore
            old_argv = sys.argv[:]
            try:
                sys.argv = [sys.argv[0], "--help"]
                run_id_backbone.main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv
        except Exception:
            pass
        return

    forwarded = _normalize_legacy_args(raw_argv)

    try:
        import run_id_backbone  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "scripts/run_baselines.py now delegates to scripts/run_id_backbone.py. "
            "Please make sure scripts/run_id_backbone.py exists and is syntactically valid."
        ) from exc

    # Replace argv and call the real runner.
    sys.argv = [sys.argv[0]] + forwarded
    run_id_backbone.main()


if __name__ == "__main__":
    main()
