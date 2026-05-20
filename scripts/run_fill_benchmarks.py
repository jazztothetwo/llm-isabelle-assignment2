#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import argparse
import csv
import json
from pathlib import Path

from prover.isabelle_api import (
    start_isabelle_server,
    get_isabelle_client,
    graceful_terminate,
)
from planner.fill import fill_script


def load_benchmarks(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "category",
        "difficulty",
        "success",
        "elapsed_s",
        "attempts",
        "used_methods",
        "message",
        "filled_script",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow({
                "name": row.get("name", ""),
                "category": row.get("category", ""),
                "difficulty": row.get("difficulty", ""),
                "success": row.get("success", False),
                "elapsed_s": round(row.get("elapsed_s", 0.0), 3),
                "attempts": row.get("attempts", 0),
                "used_methods": "; ".join(row.get("used_methods", [])),
                "message": row.get("message", ""),
                "filled_script": row.get("filled_script", ""),
            })


def write_json(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def extract_session_id(session_start_responses):
    """
    Your isabelle_client version returns response objects, not a raw session id.
    Extract the session_id from the FINISHED response.
    """
    responses = list(session_start_responses)

    for r in responses:
        response_type = getattr(r, "response_type", None)
        response_type_value = getattr(response_type, "value", str(response_type))

        if str(response_type_value).upper() == "FINISHED":
            return r.response_body.session_id

    raise RuntimeError(f"Could not extract Isabelle session_id from: {responses}")


def main():
    parser = argparse.ArgumentParser(description="Run Fill benchmarks.")

    parser.add_argument(
        "--benchmarks",
        default="benchmarks/fill_benchmarks.json",
        help="Path to Fill benchmark JSON file.",
    )

    parser.add_argument(
        "--csv-out",
        default="results/fill_benchmark_results.csv",
        help="Path to output CSV file.",
    )

    parser.add_argument(
        "--json-out",
        default="results/fill_benchmark_results.json",
        help="Path to output JSON file.",
    )

    parser.add_argument("--model", default="qwen3-coder:30b")
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")

    args = parser.parse_args()

    benchmarks = load_benchmarks(Path(args.benchmarks))

    if args.limit is not None:
        benchmarks = benchmarks[: args.limit]

    print(f"Loaded {len(benchmarks)} Fill benchmarks from {args.benchmarks}")
    print()

    server_info, proc = start_isabelle_server(name="isabelle", log_file="server.log")
    print(server_info.strip())

    try:
        isabelle = get_isabelle_client(server_info)
        session_id = extract_session_id(isabelle.session_start(session="HOL"))
        print(f"session_id: {session_id}")
        print()

        results = []

        for i, benchmark in enumerate(benchmarks, start=1):
            name = benchmark["name"]
            script = benchmark["script"]

            print(f"[{i}/{len(benchmarks)}] {name}")

            fill_result = fill_script(
                isabelle=isabelle,
                session_id=session_id,
                script=script,
                model_name=args.model,
                beam_w=args.beam,
                max_depth=args.max_depth,
                timeout_s=args.timeout,
                trace=args.trace,
                enable_reranker=not args.no_reranker,
            )

            row = {
                **benchmark,
                "success": fill_result.success,
                "filled_script": fill_result.filled_script,
                "used_methods": fill_result.used_methods,
                "elapsed_s": fill_result.elapsed_s,
                "attempts": fill_result.attempts,
                "message": fill_result.message,
            }

            results.append(row)

            status = "PASS" if fill_result.success else "FAIL"
            print(f"  result:   {status}")
            print(f"  methods:  {', '.join(fill_result.used_methods) if fill_result.used_methods else '-'}")
            print(f"  attempts: {fill_result.attempts}")
            print(f"  elapsed:  {fill_result.elapsed_s:.2f}s")
            if fill_result.message:
                print(f"  message:  {fill_result.message}")
            print()

        write_csv(results, Path(args.csv_out))
        write_json(results, Path(args.json_out))

        solved = sum(1 for r in results if r["success"])
        total = len(results)
        success_rate = (solved / total * 100) if total else 0.0

        print("Fill benchmark summary")
        print("----------------------")
        print(f"Solved:       {solved}/{total}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"CSV saved to: {args.csv_out}")
        print(f"JSON saved to:{args.json_out}")

    finally:
        graceful_terminate(proc)


if __name__ == "__main__":
    main()