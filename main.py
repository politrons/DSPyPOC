from __future__ import annotations

import argparse
import subprocess
import sys

MODULE_BY_COMMAND = {
    "optimize": "src.poc_dspy_databricks.optimize_prompts",
    "index-kb": "src.poc_dspy_databricks.build_rag_index",
    "register": "src.poc_dspy_databricks.log_register_model",
    "deploy": "src.poc_dspy_databricks.deploy_mosaic_endpoint",
}


def run_module(module: str, module_args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *module_args]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launcher for DSPy + Databricks + Hugging Face PoC"
    )
    parser.add_argument(
        "command",
        choices=sorted(MODULE_BY_COMMAND.keys()),
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "module_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected stage",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module = MODULE_BY_COMMAND[args.command]

    forwarded = args.module_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    return run_module(module, forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
