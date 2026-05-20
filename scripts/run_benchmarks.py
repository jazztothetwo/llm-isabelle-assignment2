#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def run_goal(goal: str, max_depth: int, timeout: int, trace: bool, no_reranker: bool):
    """
    Runs one benchmark goal using the existing CLI.

    Returns a dictionary containing:
    - success: bool
    - elapsed_s: float
    - stdout: str
    - stderr: str
    - proof: str
    """

    cmd = [
        sys.executable,
        "-m",
        "prover.cli",
        "--goal",
        goal,
        "--max-depth",
        str(max_depth),
    ]

    if trace:
        cmd.append("--trace")

    if no_reranker:
        cmd.append("--no-reranker")

    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        elapsed_s = time.time() - start
        stdout = result.stdout
        stderr = result.stderr

        success = any(line.startswith("SUCCESS") for line in stdout.splitlines())

        proof = extract_proof(stdout)

        return {
            "success": success,
            "elapsed_s": elapsed_s,
            "stdout": stdout,
            "stderr": stderr,
            "proof": proof,
            "returncode": result.returncode,
            "timeout": False,
        }

    except subprocess.TimeoutExpired as e:
        elapsed_s = time.time() - start

        return {
            "success": False,
            "elapsed_s": elapsed_s,
            "stdout": e.stdout or "",
            "stderr": e.stderr or "",
            "proof": "",
            "returncode": None,
            "timeout": True,
        }


def extract_proof(stdout: str) -> str:
    """
    Extracts the final proof lines from the CLI output.

    Example output section:

    SUCCESS | depth: 1
    lemma "A \\<longrightarrow> A"
    by simp
    """

    lines = stdout.splitlines()

    for i, line in enumerate(lines):
        if line.startswith("SUCCESS"):
            return "\n".join(lines[i + 1 :]).strip()

    return ""


def load_benchmarks(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "category",
        "difficulty",
        "goal",
        "expected_method",
        "success",
        "timeout",
        "elapsed_s",
        "returncode",
        "proof",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow({
                "name": row.get("name", ""),
                "category": row.get("category", ""),
                "difficulty": row.get("difficulty", ""),
                "goal": row.get("goal", ""),
                "expected_method": row.get("expected_method", ""),
                "success": row.get("success", False),
                "timeout": row.get("timeout", False),
                "elapsed_s": round(row.get("elapsed_s", 0.0), 3),
                "returncode": row.get("returncode", ""),
                "proof": row.get("proof", ""),
            })


def write_json(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run Isabelle prover benchmarks.")

    parser.add_argument(
        "--benchmarks",
        default="benchmarks/testbenchmarks.json",
        help="Path to benchmark JSON file.",
    )

    parser.add_argument(
        "--csv-out",
        default="results/testbenchmark_results.csv",
        help="Path to output CSV results file.",
    )

    parser.add_argument(
        "--json-out",
        default="results/testbenchmark_results.json",
        help="Path to output JSON results file.",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum proof search depth.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout per benchmark goal in seconds.",
    )

    parser.add_argument(
        "--trace",
        action="store_true",
        help="Show detailed prover trace output.",
    )

    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable reranker.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N benchmarks.",
    )

    args = parser.parse_args()

    benchmark_path = Path(args.benchmarks)
    csv_out = Path(args.csv_out)
    json_out = Path(args.json_out)

    benchmarks = load_benchmarks(benchmark_path)

    if args.limit is not None:
        benchmarks = benchmarks[: args.limit]

    results = []

    print(f"Loaded {len(benchmarks)} benchmarks from {benchmark_path}")
    print()

    for i, benchmark in enumerate(benchmarks, start=1):
        name = benchmark["name"]
        goal = benchmark["goal"]
        category = benchmark.get("category", "")
        difficulty = benchmark.get("difficulty", "")

        print(f"[{i}/{len(benchmarks)}] {name}")
        print(f"  category:   {category}")
        print(f"  difficulty: {difficulty}")
        print(f"  goal:       {goal}")

        run_result = run_goal(
            goal=goal,
            max_depth=args.max_depth,
            timeout=args.timeout,
            trace=args.trace,
            no_reranker=args.no_reranker,
        )

        row = {
            **benchmark,
            **run_result,
        }

        results.append(row)

        status = "PASS" if run_result["success"] else "FAIL"
        if run_result["timeout"]:
            status = "TIMEOUT"

        print(f"  result:     {status}")
        print(f"  elapsed:    {run_result['elapsed_s']:.2f}s")

        if run_result["proof"]:
            print("  proof:")
            for line in run_result["proof"].splitlines():
                print(f"    {line}")

        print()

    write_csv(results, csv_out)
    write_json(results, json_out)

    solved = sum(1 for r in results if r["success"])
    total = len(results)
    success_rate = (solved / total * 100) if total else 0.0

    print("Benchmark summary")
    print("-----------------")
    print(f"Solved:       {solved}/{total}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"CSV saved to: {csv_out}")
    print(f"JSON saved to:{json_out}")


if __name__ == "__main__":
    main()