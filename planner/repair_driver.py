from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from planner.fill import fill_script, check_script
from planner.repair import try_cegis_repairs, regenerate_whole_proof


DIRECT_METHODS = [
    "by simp",
    "by auto",
    "by clarsimp",
    "by fastforce",
    "by blast",
]


@dataclass
class RepairDriverResult:
    success: bool
    repaired_script: str
    elapsed_s: float
    stage_used: str
    attempts: int
    log: List[str]
    message: str = ""


def _first_sorry_span(script: str) -> Optional[Tuple[int, int]]:
    pos = script.find("sorry")
    if pos < 0:
        return None
    return (pos, pos + len("sorry"))


def _extract_assumption_labels_before(script: str, pos: int) -> List[str]:
    """
    Extract local assumption labels before a position.

    Example:
      assume h: "A ..."  -> h
      assume ab: "A ..." -> ab
    """
    before = script[:pos]

    labels: List[str] = []
    for m in re.finditer(
        r"\bassume\s+([A-Za-z_][A-Za-z0-9_]*)\s*:",
        before,
        flags=re.MULTILINE,
    ):
        labels.append(m.group(1).strip())

    seen = set()
    out: List[str] = []
    for label in labels:
        if label not in seen:
            out.append(label)
            seen.add(label)

    return out


def _extract_lemma_goal(script: str) -> Optional[str]:
    """
    Extract the first quoted lemma goal from a script.

    Example:
      lemma "rev (rev xs) = xs"
    """
    m = re.search(r'\blemma\b\s*"([^"]+)"', script, flags=re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip()


def _deterministic_local_repairs(script: str) -> List[Tuple[str, str]]:
    """
    Generate simple local method repairs.

    Examples:
      by simp  -> using h by simp
      by arith -> using h by simp
    """
    candidates: List[Tuple[str, str]] = []

    method_re = re.compile(
        r"(?P<indent>^[ \t]*)(?P<method>(?:using\s+[A-Za-z0-9_ ]+\s+)?by\s+[A-Za-z_][A-Za-z0-9_]*(?:[ \t]+[^\n]*)?)",
        flags=re.MULTILINE,
    )

    for m in method_re.finditer(script):
        old_method = m.group("method")
        start, end = m.span("method")

        labels = _extract_assumption_labels_before(script, start)

        replacement_methods = list(DIRECT_METHODS)

        if labels:
            prefix = "using " + " ".join(labels) + " "
            replacement_methods.extend(prefix + method for method in DIRECT_METHODS)

        for new_method in replacement_methods:
            if new_method.strip() == old_method.strip():
                continue

            candidate = script[:start] + new_method + script[end:]
            desc = f"deterministic local repair: {old_method} -> {new_method}"
            candidates.append((desc, candidate))

    return candidates


def _deterministic_whole_proof_repairs(script: str) -> List[Tuple[str, str]]:
    """
    Generate simple whole-proof replacements.

    Example:
      lemma "rev (rev xs) = xs"
      proof
        ...
      qed

    becomes:
      lemma "rev (rev xs) = xs"
        by simp
    """
    goal = _extract_lemma_goal(script)
    if not goal:
        return []

    candidates: List[Tuple[str, str]] = []

    for method in DIRECT_METHODS:
        candidate = f'lemma "{goal}"\n  {method}'
        desc = f"deterministic whole-proof repair: {method}"
        candidates.append((desc, candidate))

    return candidates


def _check_candidate(
    *,
    isabelle,
    session_id: str,
    candidate: str,
    timeout_s: int,
) -> bool:
    """
    Small wrapper so the main loop reads more clearly.
    """
    return check_script(isabelle, session_id, candidate, timeout_s=timeout_s)


def repair_with_fill(
    *,
    isabelle,
    session_id: str,
    script: str,
    goal_text: str,
    model: Optional[str] = None,
    repair_budget_s: float = 30.0,
    fill_timeout_s: int = 30,
    beam_k: int = 1,
    trace: bool = False,
) -> RepairDriverResult:
    """
    Integrated CEGIS-style repair driver.

    Flow:
    1. If the script already verifies, return success.
    2. If it contains sorry, run Fill first.
    3. Try deterministic local repairs, e.g. by simp -> using h by simp.
    4. Call the existing CEGIS repair implementation.
    5. If CEGIS output contains sorry, call Fill again.
    6. Try deterministic whole-proof repair, e.g. lemma "goal" by simp.
    7. Fall back to the existing whole-proof regeneration.
    8. If whole-proof regeneration introduces sorry, call Fill again.
    """
    start = time.time()
    log: List[str] = []
    attempts = 0
    current = script

    # 1. Already valid?
    if _check_candidate(
        isabelle=isabelle,
        session_id=session_id,
        candidate=current,
        timeout_s=fill_timeout_s,
    ):
        return RepairDriverResult(
            success=True,
            repaired_script=current,
            elapsed_s=time.time() - start,
            stage_used="none",
            attempts=attempts,
            log=["Initial script already verifies."],
            message="Initial script already verifies.",
        )

    # 2. Fill first if there are existing holes.
    if "sorry" in current:
        attempts += 1
        log.append("Initial Fill attempt on existing sorry holes.")

        fill_result = fill_script(
            isabelle=isabelle,
            session_id=session_id,
            script=current,
            model_name=model or "qwen3-coder:30b",
            beam_w=3,
            max_depth=5,
            timeout_s=fill_timeout_s,
            trace=trace,
            enable_reranker=False,
        )

        if fill_result.success:
            return RepairDriverResult(
                success=True,
                repaired_script=fill_result.filled_script,
                elapsed_s=time.time() - start,
                stage_used="fill",
                attempts=attempts,
                log=log + fill_result.used_methods,
                message="Repaired by Fill.",
            )

        current = fill_result.filled_script
        log.append(f"Fill failed: {fill_result.message}")

    # 3. Deterministic local repair before expensive CEGIS/LLM repair.
    for desc, candidate in _deterministic_local_repairs(current):
        attempts += 1
        log.append(desc)

        if trace:
            print(f"[repair-driver] trying {desc}")

        if _check_candidate(
            isabelle=isabelle,
            session_id=session_id,
            candidate=candidate,
            timeout_s=fill_timeout_s,
        ):
            return RepairDriverResult(
                success=True,
                repaired_script=candidate,
                elapsed_s=time.time() - start,
                stage_used="deterministic_local",
                attempts=attempts,
                log=log,
                message=desc,
            )

    # 4. Existing CEGIS repair needs a hole_span. If no sorry exists,
    # use the start of the script as a dummy anchor.
    hole_span = _first_sorry_span(current)
    if hole_span is None:
        hole_span = (0, 0)

    attempts += 1
    log.append("Calling existing try_cegis_repairs().")

    patched, ok, reason = try_cegis_repairs(
        full_text=current,
        hole_span=hole_span,
        goal_text=goal_text,
        model=model,
        isabelle=isabelle,
        session=session_id,
        repair_budget_s=repair_budget_s,
        max_ops_to_try=3,
        beam_k=beam_k,
        allow_whole_fallback=False,
        trace=trace,
        resume_stage=0,
    )

    log.append(f"try_cegis_repairs result: ok={ok}, reason={reason}")

    # 5. If CEGIS verified directly, done.
    if ok and "sorry" not in patched:
        if _check_candidate(
            isabelle=isabelle,
            session_id=session_id,
            candidate=patched,
            timeout_s=fill_timeout_s,
        ):
            return RepairDriverResult(
                success=True,
                repaired_script=patched,
                elapsed_s=time.time() - start,
                stage_used=f"cegis:{reason}",
                attempts=attempts,
                log=log,
                message=reason,
            )

    # 6. If CEGIS output has sorry, Fill it.
    if "sorry" in patched:
        attempts += 1
        log.append("Running Fill after CEGIS repair.")

        fill_result = fill_script(
            isabelle=isabelle,
            session_id=session_id,
            script=patched,
            model_name=model or "qwen3-coder:30b",
            beam_w=3,
            max_depth=5,
            timeout_s=fill_timeout_s,
            trace=trace,
            enable_reranker=False,
        )

        if fill_result.success:
            return RepairDriverResult(
                success=True,
                repaired_script=fill_result.filled_script,
                elapsed_s=time.time() - start,
                stage_used=f"cegis_then_fill:{reason}",
                attempts=attempts,
                log=log + fill_result.used_methods,
                message="CEGIS repair followed by Fill succeeded.",
            )

        log.append(f"Fill after CEGIS failed: {fill_result.message}")

    # 7. Deterministic whole-proof fallback before LLM whole-proof regeneration.
    for desc, candidate in _deterministic_whole_proof_repairs(current):
        attempts += 1
        log.append(desc)

        if trace:
            print(f"[repair-driver] trying {desc}")

        if _check_candidate(
            isabelle=isabelle,
            session_id=session_id,
            candidate=candidate,
            timeout_s=fill_timeout_s,
        ):
            return RepairDriverResult(
                success=True,
                repaired_script=candidate,
                elapsed_s=time.time() - start,
                stage_used="deterministic_whole",
                attempts=attempts,
                log=log,
                message=desc,
            )

    # 8. Existing whole-proof regeneration fallback.
    attempts += 1
    log.append("Calling existing regenerate_whole_proof().")

    regen_text, regen_ok, regen_reason = regenerate_whole_proof(
        full_text=current,
        goal_text=goal_text,
        model=model,
        isabelle=isabelle,
        session=session_id,
        budget_s=repair_budget_s,
        trace=trace,
        prior_outline_text=current,
    )

    log.append(f"regenerate_whole_proof result: ok={regen_ok}, reason={regen_reason}")

    if regen_ok and "sorry" not in regen_text:
        if _check_candidate(
            isabelle=isabelle,
            session_id=session_id,
            candidate=regen_text,
            timeout_s=fill_timeout_s,
        ):
            return RepairDriverResult(
                success=True,
                repaired_script=regen_text,
                elapsed_s=time.time() - start,
                stage_used=f"whole:{regen_reason}",
                attempts=attempts,
                log=log,
                message=regen_reason,
            )

    # 9. If whole-proof regeneration introduced sorry, Fill it.
    if "sorry" in regen_text:
        attempts += 1
        log.append("Running Fill after whole-proof regeneration.")

        fill_result = fill_script(
            isabelle=isabelle,
            session_id=session_id,
            script=regen_text,
            model_name=model or "qwen3-coder:30b",
            beam_w=3,
            max_depth=5,
            timeout_s=fill_timeout_s,
            trace=trace,
            enable_reranker=False,
        )

        if fill_result.success:
            return RepairDriverResult(
                success=True,
                repaired_script=fill_result.filled_script,
                elapsed_s=time.time() - start,
                stage_used=f"whole_then_fill:{regen_reason}",
                attempts=attempts,
                log=log + fill_result.used_methods,
                message="Whole-proof regeneration followed by Fill succeeded.",
            )

        log.append(f"Fill after whole-proof regeneration failed: {fill_result.message}")

    return RepairDriverResult(
        success=False,
        repaired_script=regen_text if regen_ok else patched,
        elapsed_s=time.time() - start,
        stage_used="failed",
        attempts=attempts,
        log=log,
        message="Repair driver failed.",
    )