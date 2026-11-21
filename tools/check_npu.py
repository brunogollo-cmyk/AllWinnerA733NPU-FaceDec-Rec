#!/usr/bin/env python3
"""Self-test for the A733 NPU path.

Loads the committed .nb graphs through the same ctypes binding facegate uses
and runs one inference with a neutral input. Prints driver version, chip id,
tensor shapes and timings.

Usage:
    ./.venv/bin/python tools/check_npu.py [--lib /usr/local/lib]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from facegate.npu import NpuUnavailable, load_session_reason  # noqa: E402


def run_one(nb_path: Path, lib_dir: Path) -> bool:
    label = nb_path.name
    session, err = load_session_reason(nb_path, lib_dir)
    if session is None:
        print(f"{label:>16}  FAIL  {err}")
        return False

    if session.input_is_image_u8:
        tensor = np.zeros((session.input_side, session.input_side, 3), np.uint8)
    elif session.input_data_format == 1:  # FP16
        dims = session.input_dims
        shape = (1, dims[2], dims[1], dims[0]) if len(dims) >= 3 else (1, 128)
        tensor = np.zeros(shape, np.float16)
    else:
        tensor = np.zeros((1, 1), np.uint8)

    start = time.perf_counter()
    outs = session.run_all(tensor)
    elapsed = (time.perf_counter() - start) * 1000.0

    info = (
        f"{label:>16}  in={session.in_size}B dims={session.input_dims} "
        f"fmt={session.input_data_format} outs={session.output_count} "
        f"run={elapsed:.1f} ms  OK"
    )
    print(info)
    if outs is not None and outs:
        print(f"{'':>16}  out shapes: {[o.shape for o in outs][:4]}"
              f"{' ...' if len(outs) > 4 else ''}")
    session.close()
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", default="/usr/local/lib",
                    help="directory with libNBGlinker.so (default: /usr/local/lib)")
    args = ap.parse_args()
    lib_dir = Path(args.lib)

    print(f"driver directory: {lib_dir}")
    print(f"device: {'present' if Path('/dev/vipcore').exists() else 'MISSING'}")

    ok = True
    for name in ("yunet.nb", "sface_int8.nb"):
        nb = ROOT / "models" / "npu" / name
        if not nb.exists():
            print(f"{name:>16}  FAIL  not found: {nb}")
            ok = False
            continue
        try:
            ok = run_one(nb, lib_dir) and ok
        except NpuUnavailable as exc:
            print(f"{name:>16}  FAIL  {exc}")
            ok = False

    print("NPU OK" if ok else "NPU NOT USABLE (facegate will fall back to CPU)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
