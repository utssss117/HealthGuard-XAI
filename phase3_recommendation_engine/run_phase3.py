"""
run_phase3.py
─────────────
Main entry point for the Phase 3 Risk-Weighted Adaptive Wellness
Recommendation Model (RW-AWRM).

Runs:
    1. Full hybrid pipeline on 10 synthetic patient profiles
    2. Rule-only vs Hybrid comparison table
    3. Ranking consistency validation
    4. Per-patient formatted console output

Usage:
    python run_phase3.py
"""

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phase3_recommendation_engine.utils import (
    generate_synthetic_patients,
    run_evaluation,
    compare_rule_vs_hybrid,
    validate_ranking_consistency,
    print_patient_report,
)


def main():
    patients = generate_synthetic_patients()

    # ── 1. Full hybrid evaluation ────────────────────────────────────────────
    print("\n" + "█" * 80)
    print("  PHASE 3 — Risk-Weighted Adaptive Wellness Recommendation Model (RW-AWRM)")
    print("  Hybrid Evaluation on 10 Synthetic Patients")
    print("█" * 80)

    results = run_evaluation(patients, use_llm=True)
    for res in results:
        print_patient_report(res)

    # ── 2. Rule-only vs Hybrid comparison ───────────────────────────────────
    print("\n" + "─" * 80)
    print("  COMPARISON: Rule-Only vs Hybrid (Top-1 Recommendation Priority Score)")
    print("─" * 80)
    comparisons = compare_rule_vs_hybrid(patients)
    print(f"  {'ID':<5} {'Label':<42} {'Rule Score':>10} {'Hybrid Score':>12}")
    print("  " + "-" * 70)
    for c in comparisons:
        print(
            f"  {c['patient_id']:<5} "
            f"{c['label'][:42]:<42} "
            f"{c['rule_top1']['priority_score']:>10.4f} "
            f"{c['hybrid_top1']['priority_score']:>12.4f}"
        )

    # ── 3. Ranking consistency ───────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  RANKING CONSISTENCY VALIDATION")
    print("─" * 80)
    consistency = validate_ranking_consistency(results)
    print(f"  Total Patients : {consistency['total']}")
    print(f"  Passed         : {consistency['passed']}")
    print(f"  Failed         : {consistency['failed']}")
    for pid, status in consistency["details"].items():
        print(f"    {pid}: {status}")

    # ── 4. Sample structured JSON output for P04 (highest risk) ─────────────
    print("\n" + "─" * 80)
    print("  SAMPLE STRUCTURED JSON OUTPUT — P04 (Multi-Comorbid High Risk)")
    print("─" * 80)
    p04_result = next(r for r in results if r["patient_id"] == "P04")
    print(json.dumps(p04_result["output"], indent=2))
    print("─" * 80)


if __name__ == "__main__":
    main()
