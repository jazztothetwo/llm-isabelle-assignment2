from __future__ import annotations

import re
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from prover.isabelle_api import run_theory, finished_ok
from prover.prover import prove_goal


SIMPLE_METHODS = [
    "by simp",
    "by auto",
    "by clarsimp",
    "by fastforce",
    "by blast",
]


@dataclass
class FillResult:
    success: bool
    filled_script: str
    used_methods: List[str]
    elapsed_s: float
    attempts: int
    message: str = ""


def wrap_theory(script: str) -> str:
    return textwrap.dedent(f"""
    theory Scratch
      imports Main
    begin

    {script}

    end
    """)

def extract_assumption_labels_before_first_sorry(script: str) -> List[str]:
    r"""
    Collect assumption labels before the first sorry.

    Example:
      assume h: "A \<and> B"    -> h
      assume ab: "A \<longrightarrow> B" -> ab
      assume a: A              -> a
    """
    sorry_pos = script.find("sorry")
    if sorry_pos < 0:
        return []

    before = script[:sorry_pos]

    labels = []
    for m in re.finditer(
        r'\bassume\s+([A-Za-z_][A-Za-z0-9_]*)\s*:',
        before,
        flags=re.MULTILINE,
    ):
        labels.append(m.group(1).strip())

    seen = set()
    unique = []
    for label in labels:
        if label not in seen:
            unique.append(label)
            seen.add(label)

    return unique

def check_script(isabelle, session_id: str, script: str, timeout_s: int = 10) -> bool:
    """
    Check a full Isabelle script.
    A script is only considered complete if Isabelle accepts it and it contains no sorry.
    """
    if "sorry" in script:
        return False

    theory_text = wrap_theory(script)
    responses = run_theory(isabelle, session_id, theory_text, timeout_s=timeout_s)
    ok, _ = finished_ok(responses)
    return ok


def check_script_allow_sorry(isabelle, session_id: str, script: str, timeout_s: int = 10) -> bool:
    """
    Check whether a partial script is syntactically/type valid even if it still has sorry.
    Useful while filling one hole at a time.
    """
    theory_text = wrap_theory(script)
    responses = run_theory(isabelle, session_id, theory_text, timeout_s=timeout_s)
    ok, _ = finished_ok(responses)
    return ok


def replace_first_sorry(script: str, replacement: str) -> str:
    return script.replace("sorry", replacement, 1)


def proof_tail_from_steps(steps: List[str]) -> str:
    """
    prove_goal returns steps like:

        lemma "A ..."
        by simp

    For filling a sorry hole, we only need:

        by simp
    """
    if not steps:
        return ""

    if steps[0].strip().startswith("lemma"):
        tail = steps[1:]
    else:
        tail = steps

    return "\n".join(tail).strip()


def extract_goal_before_first_sorry(script: str) -> Optional[str]:
    """
    Extract the nearest show/have statement before the first sorry.

    This version considers both quoted and unquoted show/have statements
    and chooses whichever occurs closest to the sorry.
    """
    sorry_pos = script.find("sorry")
    if sorry_pos < 0:
        return None

    before = script[:sorry_pos]
    candidates = []

    # Quoted show/have:
    #   show "B \<and> A"
    #   have foo: "P"
    for m in re.finditer(
        r'\b(?:show|have)\b(?:\s+[A-Za-z_][A-Za-z0-9_]*\s*:)?\s*"([^"]+)"',
        before,
        flags=re.MULTILINE,
    ):
        candidates.append((m.start(), m.group(1).strip()))

    # Plain show/have:
    #   show C
    #   have foo: C
    for m in re.finditer(
        r'\b(?:show|have)\b(?:\s+[A-Za-z_][A-Za-z0-9_]*\s*:)?\s+([^\n"]+)',
        before,
        flags=re.MULTILINE,
    ):
        raw = m.group(1).strip()
        raw = raw.split(" by ")[0].strip()
        raw = raw.split(" proof")[0].strip()

        if raw and raw not in {"proof", "qed", "sorry"}:
            candidates.append((m.start(), raw))

    if not candidates:
        return None

    # Choose nearest show/have before sorry.
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def extract_assumptions_before_first_sorry(script: str) -> List[str]:
    """
    Collects assumptions before the first sorry.

    Handles:

        assume h: "A"
        assume "A"
        assume a: A

    This is a conservative first version. It does not fully understand Isar scoping,
    but it works for simple nested benchmark scripts.
    """
    sorry_pos = script.find("sorry")
    if sorry_pos < 0:
        return []

    before = script[:sorry_pos]
    assumptions: List[str] = []

    # Quoted assumptions: assume h: "A \<and> B"
    for m in re.finditer(
        r'\bassume\b(?:\s+[A-Za-z_][A-Za-z0-9_]*\s*:)?\s*"([^"]+)"',
        before,
        flags=re.MULTILINE,
    ):
        assumptions.append(m.group(1).strip())

    # Plain assumptions: assume a: A
    for m in re.finditer(
        r'\bassume\b(?:\s+[A-Za-z_][A-Za-z0-9_]*\s*:)?\s+([A-Za-z][A-Za-z0-9_]*)\s*$',
        before,
        flags=re.MULTILINE,
    ):
        assumptions.append(m.group(1).strip())

    # Remove duplicates while preserving order.
    seen = set()
    unique = []
    for a in assumptions:
        if a not in seen:
            unique.append(a)
            seen.add(a)

    return unique


def build_subgoal_from_context(assumptions: List[str], target: str) -> str:
    """
    Convert local Isar context into a standalone Isabelle goal.

    Example:

        assumptions = ["A \\<and> B"]
        target = "B \\<and> A"

    becomes:

        A \\<and> B \\<Longrightarrow> B \\<and> A
    """
    if assumptions:
        return " \\<Longrightarrow> ".join(assumptions + [target])
    return target


def try_simple_fill_methods(isabelle, session_id: str, script: str, timeout_s: int) -> Tuple[Optional[str], Optional[str], int]:
    """
    First try replacing sorry with direct methods.

    For local Isar proof holes, also try:
      using h by simp
      using h by auto
      using ab bc a by simp
      etc.
    """
    attempts = 0

    labels = extract_assumption_labels_before_first_sorry(script)

    candidate_methods = []

    # Plain methods.
    candidate_methods.extend(SIMPLE_METHODS)

    # Context-aware methods using local assumption labels.
    if labels:
        using_prefix = "using " + " ".join(labels) + " "
        for method in SIMPLE_METHODS:
            candidate_methods.append(using_prefix + method)

    for method in candidate_methods:
        attempts += 1
        candidate = replace_first_sorry(script, method)

        if check_script_allow_sorry(isabelle, session_id, candidate, timeout_s=timeout_s):
            return candidate, method, attempts

    return None, None, attempts


def try_stepwise_fill(
    isabelle,
    session_id: str,
    script: str,
    model_name: str,
    beam_w: int,
    max_depth: int,
    timeout_s: int,
    trace: bool = False,
    enable_reranker: bool = False,
) -> Tuple[Optional[str], Optional[str], int, str]:
    """
    Extract the local hole context and call the existing stepwise prover.
    """
    target = extract_goal_before_first_sorry(script)
    if not target:
        return None, None, 0, "Could not extract show/have target before sorry."

    assumptions = extract_assumptions_before_first_sorry(script)
    subgoal = build_subgoal_from_context(assumptions, target)

    if trace:
        print("  extracted target:     ", target)
        print("  extracted assumptions:", assumptions)
        print("  stepwise subgoal:     ", subgoal)

    result = prove_goal(
        isabelle=isabelle,
        session_id=session_id,
        goal=subgoal,
        model_name_or_ensemble=model_name,
        beam_w=beam_w,
        max_depth=max_depth,
        hint_lemmas=5,
        timeout=timeout_s,
        trace=trace,
        enable_reranker=enable_reranker,
        do_minimize=False,
        use_sledge=False,
        use_qc=False,
        use_np=False,
    )

    if not result.get("success"):
        return None, None, 1, f"Stepwise prover failed on subgoal: {subgoal}"

    proof_fragment = proof_tail_from_steps(result.get("steps", []))

    labels = extract_assumption_labels_before_first_sorry(script)

    fragments_to_try = [proof_fragment]

    if labels and proof_fragment.startswith("by "):
        fragments_to_try.append("using " + " ".join(labels) + " " + proof_fragment)

    for frag in fragments_to_try:
        candidate = replace_first_sorry(script, frag)

        if check_script_allow_sorry(isabelle, session_id, candidate, timeout_s=timeout_s):
            return candidate, frag, 1, f"Filled using stepwise prover on subgoal: {subgoal}"

    if not proof_fragment:
        return None, None, 1, "Stepwise prover succeeded but returned no proof fragment."

    return None, None, 1, f"Stepwise proof fragment did not verify in local script. Subgoal was: {subgoal}"


def fill_script(
    isabelle,
    session_id: str,
    script: str,
    model_name: str = "qwen3-coder:30b",
    beam_w: int = 3,
    max_depth: int = 5,
    timeout_s: int = 30,
    max_holes: int = 20,
    trace: bool = False,
    enable_reranker: bool = False,
) -> FillResult:
    """
    Fill all sorry holes in a script.

    Strategy:
    1. Try direct local replacement: sorry -> by simp/by auto/...
    2. If that fails, extract local context and call prove_goal().
    3. Verify after every replacement.
    """
    start = time.time()
    current = script
    used_methods: List[str] = []
    attempts = 0

    for _ in range(max_holes):
        if "sorry" not in current:
            final_ok = check_script(isabelle, session_id, current, timeout_s=timeout_s)
            return FillResult(
                success=final_ok,
                filled_script=current,
                used_methods=used_methods,
                elapsed_s=time.time() - start,
                attempts=attempts,
                message="Filled all holes." if final_ok else "No sorry remains, but final script did not verify.",
            )

        # 1. Fast path: try local direct proof methods.
        candidate, method, n_attempts = try_simple_fill_methods(
            isabelle,
            session_id,
            current,
            timeout_s=timeout_s,
        )
        attempts += n_attempts

        if candidate is not None and method is not None:
            current = candidate
            used_methods.append(method)
            if trace:
                print(f"  filled sorry with direct method: {method}")
            continue

        # 2. Stepwise prover path.
        candidate, proof_fragment, n_attempts, msg = try_stepwise_fill(
            isabelle=isabelle,
            session_id=session_id,
            script=current,
            model_name=model_name,
            beam_w=beam_w,
            max_depth=max_depth,
            timeout_s=timeout_s,
            trace=trace,
            enable_reranker=enable_reranker,
        )
        attempts += n_attempts

        if candidate is not None and proof_fragment is not None:
            current = candidate
            used_methods.append(f"stepwise: {proof_fragment}")
            if trace:
                print(f"  filled sorry with stepwise proof: {proof_fragment}")
            continue

        return FillResult(
            success=False,
            filled_script=current,
            used_methods=used_methods,
            elapsed_s=time.time() - start,
            attempts=attempts,
            message=msg,
        )

    return FillResult(
        success=False,
        filled_script=current,
        used_methods=used_methods,
        elapsed_s=time.time() - start,
        attempts=attempts,
        message=f"Stopped after max_holes={max_holes}.",
    )