from __future__ import annotations

import argparse
import time
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update a Mosaic AI Model Serving endpoint")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--uc-model-name", required=True, help="Unity Catalog model name: <catalog>.<schema>.<name>")
    parser.add_argument("--uc-model-version", required=True, help="Model version to deploy")
    parser.add_argument("--workload-size", default="Small")
    parser.add_argument("--workload-type", default=None, help="Optional: CPU/GPU enum as supported by your workspace")
    parser.add_argument("--scale-to-zero", action="store_true")
    parser.add_argument("--wait-timeout-sec", type=int, default=1800)
    parser.add_argument("--poll-interval-sec", type=int, default=20)
    return parser.parse_args()


def _endpoint_exists(w: WorkspaceClient, endpoint_name: str) -> bool:
    try:
        w.serving_endpoints.get(name=endpoint_name)
        return True
    except Exception as exc:
        if "RESOURCE_DOES_NOT_EXIST" in str(exc):
            return False
        raise


def _build_served_entity(args: argparse.Namespace) -> ServedEntityInput:
    kwargs: dict[str, Any] = {
        "name": f"{args.endpoint_name.replace('-', '_')}_entity",
        "entity_name": args.uc_model_name,
        "entity_version": str(args.uc_model_version),
        "workload_size": args.workload_size,
        "scale_to_zero_enabled": args.scale_to_zero,
    }
    if args.workload_type:
        kwargs["workload_type"] = args.workload_type
    return ServedEntityInput(**kwargs)


def _wait_until_ready(
    w: WorkspaceClient,
    endpoint_name: str,
    timeout_sec: int,
    poll_interval_sec: int,
) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        endpoint = w.serving_endpoints.get(name=endpoint_name)
        state = getattr(endpoint, "state", None)
        ready = str(getattr(state, "ready", "UNKNOWN"))
        config_update = str(getattr(state, "config_update", "UNKNOWN"))

        print(f"ready={ready} config_update={config_update}")
        if ready.endswith("READY") and config_update.endswith("NOT_UPDATING"):
            return

        time.sleep(poll_interval_sec)

    raise TimeoutError(f"Endpoint {endpoint_name} was not ready within {timeout_sec} seconds")


def main() -> None:
    args = parse_args()
    w = WorkspaceClient()

    served_entity = _build_served_entity(args)

    if _endpoint_exists(w, args.endpoint_name):
        w.serving_endpoints.update_config(
            name=args.endpoint_name,
            served_entities=[served_entity],
        )
        print(f"Updated endpoint: {args.endpoint_name}")
    else:
        w.serving_endpoints.create(
            name=args.endpoint_name,
            config=EndpointCoreConfigInput(served_entities=[served_entity]),
        )
        print(f"Created endpoint: {args.endpoint_name}")

    _wait_until_ready(
        w,
        endpoint_name=args.endpoint_name,
        timeout_sec=args.wait_timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
    )
    print(f"Endpoint ready: {args.endpoint_name}")


if __name__ == "__main__":
    main()
