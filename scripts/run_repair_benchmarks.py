#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import json

from prover.isabelle_api import (
    start_isabelle_server,
    get_isabelle_client,
    graceful_terminate,
)
from planner.repair_driver import repair_with_fill


def load_benchmarks(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_session_id(session_start_responses):
    responses = list(session_start_responses)

    for r in responses:
        response_type = getattr(r, "response_type", None)
        response_type_value = getattr(response_type, "value", str(response_type))

        if str(response_type_value).upper() == "FINISHED":
            return r.response_body.session_id

    raise RuntimeError(f"Could not extract Isabelle session_id from: {responses}")


def write_csv(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "category",
        "difficulty",
        "success",
        "stage_used",
        "attempts",
        "elapsed_s",
        "message",
        "repaired_script",
        "log",
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
                "stage_used": row.get("stage_used", ""),
                "attempts": row.get("attempts", 0),
                "elapsed_s": round(row.get("elapsed_s", 0.0), 3),
                "message": row.get("message", ""),
                "repaired_script": row.get("repaired_script", ""),
                "log": " | ".join(row.get("log", [])),
            })


def write_json(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run integrated repair benchmarks.")

    parser.add_argument("--benchmarks", default="benchmarks/repair_benchmarks.json")
    parser.add_argument("--csv-out", default="results/repair_benchmark_results.csv")
    parser.add_argument("--json-out", default="results/repair_benchmark_results.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--repair-budget", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--trace", action="store_true")

    args = parser.parse_args()

    benchmarks = load_benchmarks(Path(args.benchmarks))

    if args.limit is not None:
        benchmarks = benchmarks[: args.limit]

    print(f"Loaded {len(benchmarks)} Repair benchmarks from {args.benchmarks}")
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
            print(f"[{i}/{len(benchmarks)}] {benchmark['name']}")

            result = repair_with_fill(
                isabelle=isabelle,
                session_id=session_id,
                script=benchmark["script"],
                goal_text=benchmark["goal"],
                model=args.model,
                repair_budget_s=args.repair_budget,
                fill_timeout_s=args.timeout,
                trace=args.trace,
            )

            row = {
                **benchmark,
                "success": result.success,
                "repaired_script": result.repaired_script,
                "elapsed_s": result.elapsed_s,
                "stage_used": result.stage_used,
                "attempts": result.attempts,
                "log": result.log,
                "message": result.message,
            }

            results.append(row)

            status = "PASS" if result.success else "FAIL"
            print(f"  result:   {status}")
            print(f"  stage:    {result.stage_used}")
            print(f"  attempts: {result.attempts}")
            print(f"  elapsed:  {result.elapsed_s:.2f}s")
            print(f"  message:  {result.message}")
            print()

        write_csv(results, Path(args.csv_out))
        write_json(results, Path(args.json_out))

        solved = sum(1 for r in results if r["success"])
        total = len(results)
        rate = (solved / total * 100) if total else 0.0

        print("Repair benchmark summary")
        print("------------------------")
        print(f"Solved:       {solved}/{total}")
        print(f"Success rate: {rate:.1f}%")
        print(f"CSV saved to: {args.csv_out}")
        print(f"JSON saved to:{args.json_out}")

    finally:
        graceful_terminate(proc)


if __name__ == "__main__":
    main()