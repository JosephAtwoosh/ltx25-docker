#!/usr/bin/env python3
"""Library-level preflight checks for the LTX-2.5 container.

Exit codes: 0 = fine, 2 = blocking problem found.
Kept separate from doctor.sh so the shell can capture a real exit status.
"""
from __future__ import annotations

import sys

problems = 0


def ok(msg: str) -> None:
    print(f"  ok   {msg}")


def bad(msg: str) -> None:
    global problems
    problems += 1
    print(f"  XX   {msg}")


def warn(msg: str) -> None:
    print(f"  !!   {msg}")


try:
    import torch

    ok(f"torch {torch.__version__} (CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        bad("torch.cuda.is_available() is False — no usable GPU is visible to PyTorch")
    else:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        ok(f"torch sees {name} (sm_{major}{minor}, {total:.1f} GiB)")
        if major < 8:
            bad(
                f"compute capability sm_{major}{minor} predates Ampere; "
                "FP8-cast and the CUDA 13 stack need sm_80+"
            )
except Exception as exc:  # noqa: BLE001 - report anything as a failed check
    bad(f"torch import/CUDA check failed: {exc}")

try:
    import natten

    ok(f"natten {getattr(natten, '__version__', '?')} (fast diffusion-VAE decode)")
except Exception as exc:  # noqa: BLE001
    warn(f"natten unavailable ({type(exc).__name__}); decode falls back to Triton/eager (slower)")

sys.exit(2 if problems else 0)
