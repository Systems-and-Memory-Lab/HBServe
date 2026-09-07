#!/usr/bin/env python3
"""Generate a deterministic HBServe request trace."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hbserve.contracts import HBServeError  # noqa: E402
from hbserve.io import (  # noqa: E402
    load_synthetic_request_config,
    write_json_atomic,
)
from hbserve.synthetic import generate_requests  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hbserve generate",
        description="HBServe: generate a Poisson/lognormal request trace",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        trace = generate_requests(load_synthetic_request_config(args.config))
        write_json_atomic(args.output, trace.canonical())
    except (OSError, HBServeError) as error:
        print(f"HBServe request generation failed: {error}", file=sys.stderr)
        return 2
    print(
        f"HBServe generated {len(trace.requests)} requests; "
        f"sha256={trace.digest}; "
        f"output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
