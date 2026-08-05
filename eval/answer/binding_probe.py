"""Price the binding check before anyone wires it into the guard.

Four populations, counts only:
  1. the relation-class DEV mutants -- the wrongs this check exists to catch;
  2. their paired controls (original answers) -- harness-shaped false alarms;
  3. every human-settled CORRECT answer of run2 and run2_duz -- the real
     false-review cost, the number that killed eksik_sayfa;
  4. the settled-wrong answers of both runs -- few, but free to look at.

DEVELOPMENT measurement: locked clusters never reach this module (the
harness's own gate enforces it), and nothing here decides a gate.
"""
import argparse
from pathlib import Path

from eval.answer.adversarial_feasibility import load_cases
from eval.answer.adversarial_mutate import (
    LABEL_VALUE_SWAP,
    QUESTION_ANSWER_MISMATCH,
    WRONG_ROW,
    build_mutants,
    development_cases,
)
from eval.answer.adversarial_feasibility import _selected_result_paths
from eval.answer.guard_floor import legacy_context
from pipeline.validation.rag.binding_guard import check_binding

RELATION_CLASSES = (WRONG_ROW, LABEL_VALUE_SWAP, QUESTION_ANSWER_MISMATCH)


def _handles(evidence):
    handles = [claim["pasaj"] for claim in evidence or []]
    return handles or None


def measure_mutants(run_dir: Path, question_dir: Path):
    cases, _ = load_cases(run_dir, question_dir)
    produced = build_mutants(development_cases(cases))
    print("=== 1-2) DEV MUTANTLARI ve ESLESTIRILMIS KONTROLLER ===")
    print(f"{'sinif':<28}{'mutant':>7}{'yakalanan':>11}{'kontrol':>9}{'yanlis alarm':>14}")
    for name in RELATION_CLASSES:
        pairs = produced[name]
        caught = control_flagged = 0
        for case, mutant in pairs:
            if check_binding(case.question, mutant.answer, mutant.context,
                             _handles(list(mutant.evidence))):
                caught += 1
            original_handles = _handles(
                [{"pasaj": c.handle} for c in case.evidence])
            if check_binding(case.question, case.answer, case.context,
                             original_handles):
                control_flagged += 1
        print(f"{name:<28}{len(pairs):>7}{caught:>11}{len(pairs):>9}"
              f"{control_flagged:>14}")
    print("   (aday populasyon: sayilar tanisal, kapi degil)\n")


def measure_saved_run(run_dir: Path, label: str, timings=None, quiet=False):
    import json
    import time

    correct = wrong = flagged_correct = flagged_wrong = 0
    for set_name, path in sorted(_selected_result_paths(run_dir).items()):
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("sorular", []):
            answer = row.get("cevap") or ""
            context = legacy_context(row.get("baglam") or "")
            claims = row.get("dayanak") or []
            handles = [c.get("pasaj") for c in claims
                       if isinstance(c, dict) and isinstance(c.get("pasaj"), int)]
            started = time.process_time()
            flags = check_binding(row.get("soru") or "", answer, context,
                                  handles or None)
            if timings is not None:
                timings.append(time.process_time() - started)
            if row.get("cevap_dogru"):
                correct += 1
                flagged_correct += bool(flags)
            else:
                wrong += 1
                flagged_wrong += bool(flags)
    if not quiet:
        print(f"{label:<12} dogru {correct:>3} -> bayrakli {flagged_correct:>2}"
              f"   dogru-olmayan {wrong:>2} -> bayrakli {flagged_wrong}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="output/RAG_Outputs/run2/native")
    parser.add_argument("--question-dir", default="data/rag_eval")
    parser.add_argument("--time", action="store_true",
                        help="CPU sureleri (process_time) -- kalici olcum "
                             "izlenen koddan uretilsin diye buradadir; bir "
                             "scratch scriptin sayisi denetlenemez")
    parser.add_argument("--time-repeats", type=int, default=5,
                        help="zamanlama gecis sayisi; Windows zamanlayici "
                             "taneciği ~15.6 ms oldugundan tek gecislik p95 "
                             "kararli degildir")
    args = parser.parse_args()

    measure_mutants(Path(args.run_dir), Path(args.question_dir))

    timings = [] if args.time else None
    print("=== 3-4) KAYITLI GERCEK KOSUMLAR (yanlis-inceleme maliyeti) ===")
    runs = ("run2/native", "run2_duz/native")
    for run in runs:
        measure_saved_run(Path("output/RAG_Outputs") / run,
                          run.split("/")[0], timings)
    if timings:
        per_pass = len(timings)
        for _ in range(max(args.time_repeats, 1) - 1):
            for run in runs:
                measure_saved_run(Path("output/RAG_Outputs") / run,
                                  run.split("/")[0], timings, quiet=True)
        import platform

        ordered = sorted(second * 1000 for second in timings)
        p95 = ordered[int(len(ordered) * 0.95)]
        # the environment travels with the number: a p95 measured on another
        # machine, load or timer is a different measurement, not a check
        print(f"\nCPU sure: {len(ordered)} cagri "
              f"({max(args.time_repeats, 1)} gecis x {per_pass}) -> "
              f"ortalama {sum(ordered) / len(ordered):.1f} ms · "
              f"p95 {p95:.1f} ms · en yavas {ordered[-1]:.1f} ms")
        print(f"  ortam: {platform.system()} {platform.release()} · "
              f"Python {platform.python_version()} · process_time")


if __name__ == "__main__":
    main()
