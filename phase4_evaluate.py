"""Phase 4 - Evaluation.

Runs Baseline and Engineered end-to-end over the 40-question eval set and
computes the full scorecard at K=5:

  Retriever (file-level ground truth, computed locally, no API):
    Hit Rate@5  = questions with >=1 correct file in top-5 / total
    MRR         = mean 1 / rank of first correct file
    Recall@5    = mean (correct files retrieved / correct files required)

  Generator (Claude as grader/judge, identical prompts for both systems):
    Factual Accuracy   = answers correct within +/-1% / total
    Groundedness       = supported claims / total claims
    Hallucination Rate = unsupported claims / total claims   (= 1 - Groundedness)

Generation and judge calls are cached to outputs/cache/ by content hash, so a
re-run is nearly free and never re-bills tokens for unchanged inputs.

Run (with ANTHROPIC_API_KEY set):  python phase4_evaluate.py
"""
import csv
import hashlib
import json
import re

import config
import rag

CACHE_DIR = config.OUT_DIR / "cache"


# --------------------------------------------------------------------------
# Disk cache (so re-runs don't re-bill tokens)
# --------------------------------------------------------------------------
def cached(key: str, produce):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = CACHE_DIR / (hashlib.sha256(key.encode()).hexdigest()[:20] + ".json")
    if fp.exists():
        return json.load(open(fp))
    val = produce()
    json.dump(val, open(fp, "w"))
    return val


def judge_json(prompt: str, max_tokens: int = 1500) -> dict:
    """One judge call that must return a JSON object; parse the first {...}."""
    resp = rag.messages_create(
        model=config.JUDGE_MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    txt = rag.resp_text(resp)
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0))


# --------------------------------------------------------------------------
# Retriever metrics (local)
# --------------------------------------------------------------------------
def retriever_scores(retrieved: list[dict], correct_files: set):
    files = [c["source_file"] for c in retrieved]
    ranks = [i + 1 for i, f in enumerate(files) if f in correct_files]
    hit = 1 if ranks else 0
    rr = 1 / ranks[0] if ranks else 0.0
    found = len({f for f in files if f in correct_files})
    recall = found / len(correct_files)
    return hit, rr, recall


# --------------------------------------------------------------------------
# Generator metrics (Claude judge)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Factual accuracy: deterministic +/-1% numeric check in Python (the rubric
# gives us the tolerance, so no LLM is needed -- this is transparent and free).
# Handles $ , % and scale words, and tolerates unit mismatches by powers of
# 1000 (e.g. answering in billions when the source reports millions).
# --------------------------------------------------------------------------
SCALE = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "thousand": 1e3}
NUM_RE = re.compile(r"-?\$?\s*[\d,]*\d(?:\.\d+)?\s*(trillion|billion|million|thousand)?", re.I)


def extract_numbers(text: str) -> list[float]:
    nums = []
    for m in NUM_RE.finditer(text):
        raw = re.sub(r"[,$\s]", "", m.group(0))
        raw = re.sub(r"(?i)(trillion|billion|million|thousand)", "", raw)
        try:
            v = float(raw)
        except ValueError:
            continue
        if m.group(1):
            v *= SCALE[m.group(1).lower()]
        nums.append(v)
    return nums


def parse_gold(gold: str):
    m = re.search(r"-?[\d,]*\d(?:\.\d+)?", gold.replace("$", ""))
    return float(m.group(0).replace(",", "")) if m else None


def numeric_correct(pred: str, gold: str, tol: float = 0.01) -> bool:
    gv = parse_gold(gold)
    if gv is None:                                   # non-numeric answer key
        return gold.strip().lower() in pred.lower()
    for n in extract_numbers(pred):
        for s in (1, 1e3, 1e-3, 1e6, 1e-6):          # tolerate unit scale
            denom = abs(gv) if gv else 1.0
            if abs(n * s - gv) / denom <= tol:
                return True
    return False


GND_PROMPT = """You verify whether an ANSWER is grounded in the provided CONTEXT.

CONTEXT:
{context}

ANSWER:
{pred}

Consider each distinct factual claim the ANSWER makes (each number or asserted
fact). Count how many claims it makes in total, and how many of those are
supported (directly stated in or derivable from the CONTEXT). If the ANSWER
makes no factual claims (e.g. it refuses or says "Not found"), both counts are 0.
Respond with ONLY a JSON object of two integers:
{{"total_claims": 0, "supported_claims": 0}}"""


def grade_groundedness(q, preset, pred, context):
    key = f"gnd|{config.JUDGE_MODEL}|{preset}|{q['uid']}|{hashlib.sha256(pred.encode()).hexdigest()[:12]}"
    data = cached(key, lambda: judge_json(
        GND_PROMPT.format(context=context, pred=pred)))
    total = int(data.get("total_claims", 0))
    supported = min(int(data.get("supported_claims", 0)), total)
    return supported, total


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def get_answer(q, preset, retrieved):
    key = (f"gen|{config.GEN_MODEL}|{preset}|{q['uid']}|"
           + ",".join(c["chunk_id"] for c in retrieved))
    return cached(key, lambda: {"answer": rag.generate(q["question"], retrieved, preset)})["answer"]


def evaluate(preset: str, questions: list[dict]) -> dict:
    rows = []
    for i, q in enumerate(questions, 1):
        print(f"  [{preset}] {i}/{len(questions)} {q['uid']} ... retrieving/generating",
              flush=True)
        correct = set(q["correct_files"])
        # retrieve GEN_K for the reader; score the retriever on the top TOP_K only
        retrieved = rag.retrieve(q["question"], preset, k=config.GEN_K)
        hit, rr, recall = retriever_scores(retrieved[: config.TOP_K], correct)

        pred = get_answer(q, preset, retrieved)
        correct_ans = numeric_correct(pred, q["answer"])
        supported, total = grade_groundedness(
            q, preset, pred, rag.format_context(retrieved))

        rows.append({
            "uid": q["uid"], "difficulty": q["difficulty"],
            "hit": hit, "rr": round(rr, 3), "recall": round(recall, 3),
            "correct": int(correct_ans), "supported": supported, "claims": total,
            "answer": pred.replace("\n", " ")[:300],
        })
        print(f"  [{preset}] {i}/{len(questions)} {q['uid']} "
              f"hit={hit} correct={int(correct_ans)} claims={supported}/{total}")

    n = len(rows)
    tot_claims = sum(r["claims"] for r in rows)
    tot_supported = sum(r["supported"] for r in rows)
    grounded = tot_supported / tot_claims if tot_claims else float("nan")
    return {
        "rows": rows,
        "hit_rate": sum(r["hit"] for r in rows) / n,
        "mrr": sum(r["rr"] for r in rows) / n,
        "recall": sum(r["recall"] for r in rows) / n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "groundedness": grounded,
        "hallucination": 1 - grounded,
        "answered": sum(1 for r in rows if r["claims"] > 0),
        "total_claims": tot_claims,
    }


def main():
    questions = [json.loads(l) for l in open(config.OUT_DIR / "questions.jsonl")]
    results = {}
    for preset in ("baseline", "engineered"):
        print(f"\n=== Evaluating {preset} ({len(questions)} questions) ===")
        rag.build_index(preset)                 # no-op if already built
        results[preset] = evaluate(preset, questions)

    # per-question CSV
    with open(config.OUT_DIR / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["preset", "uid", "difficulty", "hit", "rr", "recall",
                    "correct", "supported", "claims", "answer"])
        for preset, res in results.items():
            for r in res["rows"]:
                w.writerow([preset, r["uid"], r["difficulty"], r["hit"], r["rr"],
                            r["recall"], r["correct"], r["supported"], r["claims"], r["answer"]])

    # scorecard
    b, e = results["baseline"], results["engineered"]

    def pct(x):
        return "n/a" if x != x else f"{x*100:.1f}%"   # x!=x is True for NaN

    lines = [
        "# Financial RAG Scorecard",
        f"\nCorpus window: {config.START_YEAR}-{config.END_YEAR}  |  Questions: {len(questions)}"
        f"  |  metrics@K={config.TOP_K}  |  reader sees top-{config.GEN_K}\n",
        "| Metric | Baseline (Simple) | Engineered (Improved) |",
        "|---|---|---|",
        f"| Hit Rate@5 | {pct(b['hit_rate'])} | {pct(e['hit_rate'])} |",
        f"| MRR | {b['mrr']:.3f} | {e['mrr']:.3f} |",
        f"| Recall@5 | {pct(b['recall'])} | {pct(e['recall'])} |",
        f"| Groundedness | {pct(b['groundedness'])} | {pct(e['groundedness'])} |",
        f"| Factual Accuracy | {pct(b['accuracy'])} | {pct(e['accuracy'])} |",
        f"| Hallucination Rate | {pct(b['hallucination'])} | {pct(e['hallucination'])} |",
        f"\n_Answered (non-refusal): baseline {b['answered']}/{len(questions)}, "
        f"engineered {e['answered']}/{len(questions)}._",
    ]
    scorecard = "\n".join(lines)
    (config.OUT_DIR / "scorecard.md").write_text(scorecard + "\n")
    print("\n" + scorecard)
    print(f"\nWrote {config.OUT_DIR/'results.csv'} and {config.OUT_DIR/'scorecard.md'}")


if __name__ == "__main__":
    main()
