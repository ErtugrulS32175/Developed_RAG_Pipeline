"""How often does the guard flag a saved CORRECT answer with ideal evidence?

    python -m eval.answer.guard_floor output/RAG_Outputs/run1/native

The false-review rate a live run would produce mixes two very different
failures: a model that quotes badly, and a guard that is too strict. A run
against the real model cannot separate them -- an answer sent to review could
be either. This separates them by removing the model from the question.

For every answer already confirmed correct, the evidence a perfectly obedient
model WOULD have produced is synthesised -- everything the prompt actually
demands: the line carrying the expected answer, a line for every other figure
the answer states, and a passage for every page it cites. That reply is then
put through the guard. Anything it objects to is the guard objecting to ideal
input, which is a property of the guard alone.

This is a guard-only BASELINE for the fixed saved answers, not a mathematical
lower bound on a future live run. A real model will usually quote less tidily,
which pushes review upward, but the structured prompt can also change the
answer wording or abstain, which can move the live rate in either direction.
The baseline is still useful before renting a GPU: if ideal evidence cannot
make the saved correct answers pass, the guard or answer contract needs work
before model compliance is measured.

Counts only. The figures a check objects to are document content.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from eval.answer.judge import accepts_without_similarity
from eval.retrieval.rag_eval import QUESTION_DIR
from pipeline.lang.tr_notation import normalize, number_forms, numbers
from pipeline.retrieval.context import Passage, RagContext
from pipeline.validation.rag.answer_guard import check_structured

BLOCK = "\n\n---\n\n"
_PAGE = re.compile(r"Sayfa\s*(\d+)", re.IGNORECASE)
_ANSWER_PAGE = re.compile(r"sayfa\s*\d+", re.IGNORECASE)
_ANSWER_PAGE_NO = re.compile(r"sayfa\s*(\d+)", re.IGNORECASE)


def confirmed_correct(run_dir):
    """(question, answer, context) for answers the scorer confirmed itself.

    Deliberately tiers 1-2 only, never the embedding tier: this population must
    be answers known correct WITHOUT a judgement call, and it also keeps the
    measurement runnable with no services up.
    """
    for f in sorted(Path(run_dir).glob("rag_answers_*.json")):
        name = f.stem.replace("rag_answers_", "")
        if "_k" in name:
            continue
        qfile = QUESTION_DIR / f"{name}.json"
        if not qfile.exists():
            continue
        gt = {g["q"]: g for g in json.loads(qfile.read_text(encoding="utf-8"))}
        for r in json.loads(f.read_text(encoding="utf-8"))["sorular"]:
            g = gt.get(r["soru"])
            if not g or not g.get("key"):
                continue
            answer, context = r["cevap"], r.get("baglam") or ""
            if accepts_without_similarity(g["key"], answer):
                yield name, g, answer, context


def legacy_context(context):
    """Rebuild a ``RagContext`` from a saved pre-provenance run.

    This parser is an eval-only compatibility adapter, not production
    provenance.  The saved runs contain only rendered context strings, so their
    original chunk records cannot be recovered. Re-retrieving would be worse:
    it would fetch a context the saved answer was never produced from.
    """
    blocks = [b for b in (context or "").split(BLOCK) if b.strip()]
    passages = []
    for i, block in enumerate(blocks, start=1):
        head, _, body = block.partition("\n")
        page_match = _PAGE.search(head)
        page = int(page_match.group(1)) if page_match else None
        page = page if page and page > 0 else None
        passages.append(Passage(
            handle=i,
            page=page,
            text=body,
            citation=head,
        ))
    return RagContext(
        passages=tuple(passages),
        numbered=True,
    )


def _lines(context):
    """(handle, page, line) for every non-empty line of every passage."""
    for passage in context.passages:
        for line in passage.text.splitlines():
            if line.strip():
                yield passage.handle, passage.page, line.strip()


def ideal_evidence(key, answer, context):
    """Everything an obedient model would have quoted, or None if it could not.

    The prompt does not ask for one line -- it asks that EVERY figure in the
    answer appear in the quoted lines, and that the cited page be a page the
    answer used. So ideal evidence is: the line carrying the expected answer,
    plus a line for every other figure the answer states, plus a passage for
    every page it cites.

    The first version of this quoted only the key line, and the guard then
    objected to figures and pages the answer had legitimately drawn from
    elsewhere. That was the measurement blaming the guard for its own
    shortcut -- 21.4% of it -- which is why the synthesis has to match what the
    prompt actually demands before any number here means anything.
    """
    rows = list(_lines(context))
    if not rows:
        return None

    chosen = {}                                    # handle -> [line, ...]

    def take(handle, line):
        chosen.setdefault(handle, [])
        if line not in chosen[handle]:
            chosen[handle].append(line)

    key_line = next(((h, l) for h, _, l in rows
                     if accepts_without_similarity(key, l)), None)
    if not key_line:
        return None
    take(*key_line)

    body = _ANSWER_PAGE.sub(" ", answer or "")
    for forms in number_forms(normalize(body)):
        if not forms:
            continue
        quoted = numbers(normalize(" ".join(
            l for lines in chosen.values() for l in lines)))
        if forms & quoted:
            continue
        hit = next(((h, l) for h, _, l in rows if forms & numbers(normalize(l))), None)
        if hit:
            take(*hit)

    cited = {int(n) for n in _ANSWER_PAGE_NO.findall(answer or "")}
    covered = {p for h, p, _ in rows if h in chosen}
    for page in cited - covered:
        hit = next(((h, l) for h, p, l in rows if p == page), None)
        if hit:
            take(*hit)

    return [{"pasaj": h, "alinti": l} for h, lines in chosen.items() for l in lines]


def measure(run_dirs, derive):
    seen = Counter()
    flagged = Counter()
    total = 0
    for run_dir in run_dirs:
        for _, g, answer, context in confirmed_correct(run_dir):
            total += 1
            rag_context = legacy_context(context)
            dayanak = ideal_evidence(g["key"], answer, rag_context)
            if not dayanak:
                # the key is in no passage: nothing an obedient model could
                # have cited, so the guard is not on trial here
                seen["kanit_bulunamadi"] += 1
                continue
            seen["alinti_sayisi"] += len(dayanak)
            seen["pasaj_sayisi"] += len({d["pasaj"] for d in dayanak})
            seen["olculen"] += 1
            reply = {"dayanak": dayanak, "cevap": answer}
            names = {
                name
                for name, _ in check_structured(
                    reply,
                    rag_context,
                    derive=derive,
                )
            }
            if names:
                flagged["_herhangi"] += 1
                for name in names:
                    flagged[name] += 1
    return total, seen, flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    args = ap.parse_args()

    for derive in (True, False):
        total, seen, flagged = measure(args.run_dirs, derive)
        usable = seen["olculen"]
        print(f"\n=== oran turetme: {'acik' if derive else 'KAPALI (planlanan varsayilan)'} ===")
        print(f"  dogrulanmis dogru cevap        : {total}")
        print(f"    anahtar hicbir pasajda yok   : {seen['kanit_bulunamadi']}  (guard yargilanmiyor)")
        if usable:
            print(f"    cevap basina ideal alinti    : {seen['alinti_sayisi']/usable:.2f}"
                  f"  ({seen['pasaj_sayisi']/usable:.2f} pasaj)")
        if not usable:
            continue
        print(f"\n  IDEAL kanitla isaretlenen      : {flagged['_herhangi']}/{usable}"
              f" = {flagged['_herhangi'] / usable:.1%}   <-- guard-only TABAN")
        for name, n in sorted(flagged.items()):
            if name != "_herhangi":
                print(f"      {name:20} {n}")


if __name__ == "__main__":
    main()
