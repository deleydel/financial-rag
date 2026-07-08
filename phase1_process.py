"""Phase 1 - Data processing.

Turns the raw OfficeQA download into the two artifacts the rest of the
pipeline needs:

  outputs/questions.jsonl        - the filtered evaluation set (with the
                                   ground-truth source files per question)
  outputs/chunks_<preset>.jsonl  - every corpus chunk, tagged with metadata

Run:  python phase1_process.py
"""
import json
import re
from collections import Counter
from functools import lru_cache

import tiktoken

import config

ENC = tiktoken.get_encoding("cl100k_base")
FILE_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})")


def ntok(text: str) -> int:
    return len(ENC.encode(text))


# --------------------------------------------------------------------------
# CSV / question handling
# --------------------------------------------------------------------------
def parse_source_files(raw: str) -> list[str]:
    """source_files may hold several filenames separated by newlines/commas."""
    return [m.group(0) + ".txt" for m in FILE_RE.finditer(raw or "")]


def file_year_month(filename: str) -> tuple[int, int]:
    m = FILE_RE.search(filename)
    return int(m.group(1)), int(m.group(2))


def load_questions() -> list[dict]:
    """Keep questions whose answer lives ENTIRELY inside the corpus window,
    and whose ground-truth files are actually present on disk. 'Entirely
    inside' guarantees every relevant file is retrievable, so Recall is fair.
    """
    import csv

    present = {p.name for p in config.TRANSFORMED_DIR.glob("*.txt")}
    kept = []
    with open(config.CSV_PATH, newline="") as fh:
        for row in csv.DictReader(fh):
            files = parse_source_files(row["source_files"])
            if not files:
                continue
            years = {file_year_month(f)[0] for f in files}
            if not all(config.START_YEAR <= y <= config.END_YEAR for y in years):
                continue
            if not all(f in present for f in files):
                continue  # ground-truth file missing from download -> unfair
            kept.append(
                {
                    "uid": row["uid"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "correct_files": files,
                    "years": sorted(years),
                    "difficulty": row["difficulty"],
                }
            )
    return kept


# --------------------------------------------------------------------------
# Text cleaning + chunking
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.replace("\x0c", "\n")          # form feeds -> newline
    text = re.sub(r"[ \t]+\n", "\n", text)      # trailing whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse big gaps
    return text.strip()


def chunk_by_tokens(text: str, target: int, overlap: int) -> list[str]:
    """Baseline chunker: blind sliding window over tokens. Fast and naive;
    will happily cut a table in half - that is the point of the baseline."""
    toks = ENC.encode(text)
    step = max(1, target - overlap)
    out = []
    for i in range(0, len(toks), step):
        window = toks[i : i + target]
        if not window:
            break
        out.append(ENC.decode(window).strip())
        if i + target >= len(toks):
            break
    return out


SEP_RE = re.compile(r"^\s*\|?[\s:|-]*-{2,}[\s:|-]*\|")   # markdown table separator row


def token_windows(text: str, target: int, overlap: int) -> list[str]:
    """Split a long string into token windows (used for oversized prose/rows)."""
    toks = ENC.encode(text)
    if len(toks) <= target:
        return [text.strip()]
    step = max(1, target - overlap)
    out = []
    for i in range(0, len(toks), step):
        window = toks[i : i + target]
        if not window:
            break
        out.append(ENC.decode(window).strip())
        if i + target >= len(toks):
            break
    return out


def parse_elements(text: str) -> list[tuple]:
    """Split a document into ordered ('prose', str) and
    ('table', header, sep, [rows]) elements. A table is detected by its
    '| --- | --- |' separator row; the header is the line directly above it,
    and rows are the '|'-bearing lines that follow."""
    lines = text.split("\n")
    elements, prose, i = [], [], 0

    def flush_prose():
        if prose:
            block = "\n".join(prose).strip()
            if block:
                elements.append(("prose", block))
            prose.clear()

    while i < len(lines):
        if SEP_RE.match(lines[i]) and prose:
            header = prose.pop()                # line above the separator
            flush_prose()
            sep = lines[i]
            rows, j = [], i + 1
            while j < len(lines) and "|" in lines[j]:
                rows.append(lines[j])
                j += 1
            elements.append(("table", header, sep, rows))
            i = j
        else:
            prose.append(lines[i])
            i += 1
    flush_prose()
    return elements


@lru_cache(maxsize=1)
def _hf_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.EMBED_MODEL)


@lru_cache(maxsize=500_000)
def wp_body(text: str) -> int:
    """WordPiece length of a piece WITHOUT specials, cached. Sizing is done by
    summing these per-piece bodies (fast: each atomic piece is tokenized once)
    rather than re-tokenizing growing strings. Chunk fit = body_sum + 2 specials
    <= target. Cross-piece merges only shrink the true count, so summing is a
    safe over-estimate; with target=210 vs a 256 window there is ample margin."""
    return len(_hf_tokenizer()(text, add_special_tokens=False,
                               truncation=False)["input_ids"])


def wp_windows(text: str, target: int, overlap: int) -> list[str]:
    """Split an oversized unit so each piece fits `target` WordPiece tokens.
    Packs whole words and returns the ORIGINAL substrings -- we never decode
    WordPiece back to text (the tokenizer is uncased and would mangle it)."""
    words = text.split()
    if len(words) <= 1:
        return [text.strip()] if text.strip() else []
    budget = target - 2
    out, cur, lens, csum = [], [], [], 0
    for w in words:
        wl = wp_body(w)
        if cur and csum + wl > budget:
            out.append(" ".join(cur))
            keep, klen, ksum = [], [], 0            # word-level overlap tail
            for x, xl in zip(reversed(cur), reversed(lens)):
                if keep and ksum + xl > overlap:
                    break
                keep.insert(0, x); klen.insert(0, xl); ksum += xl
            cur, lens, csum = keep, klen, ksum
        cur.append(w); lens.append(wl); csum += wl
    if cur:
        out.append(" ".join(cur))
    return out


def split_table(header, sep, rows, target, overlap, cap, blen, windows, spec) -> list[str]:
    """Group table rows so each header+sep+rows piece fits `target`, repeating
    the header + separator on every piece so each chunk is self-describing.
    Row lengths are summed (each row measured once) via `blen`."""
    emit = lambda g: "\n".join([header, sep, *g])
    base = blen(header) + blen(sep)
    budget, hardcap = target - spec, cap - spec
    out, group, gsum = [], [], 0
    for row in rows:
        rl = blen(row)
        if group and base + gsum + rl > budget:
            out.append(emit(group)); group, gsum = [], 0
        if base + rl > hardcap:                 # pathologically wide single row
            out.extend(windows(emit([row]), cap, overlap)); continue
        group.append(row); gsum += rl
    if group:
        out.append(emit(group))
    return out


def chunk_table_aware(text, target, overlap, cap, blen, windows, spec) -> list[str]:
    """Structure-aware engineered chunker: parse the doc into prose and tables,
    pack prose paragraphs up to the token target, and split big tables row-wise
    with header repetition. Sizes summed from the embedder's own tokenizer, so
    nothing is truncated."""
    budget = target - spec
    units: list[str] = []
    for el in parse_elements(text):
        if el[0] == "table":
            _, header, sep, rows = el
            whole = "\n".join([header, sep, *rows])
            if blen(whole) + spec <= target:
                units.append(whole)
            else:
                units.extend(split_table(header, sep, rows, target, overlap,
                                          cap, blen, windows, spec))
        else:                                   # prose
            for para in re.split(r"\n\s*\n", el[1]):
                para = para.strip()
                if not para:
                    continue
                units.extend(windows(para, target, overlap)
                             if blen(para) + spec > target else [para])

    # pack prose units together; table units stand alone (their repeated
    # header is the continuity, so overlap would be redundant).
    out, cur, csum = [], [], 0
    for u in units:
        if u.startswith("|") or SEP_RE.search(u) is not None:   # table unit
            if cur:
                out.append("\n\n".join(cur)); cur, csum = [], 0
            out.append(u)
            continue
        ul = blen(u)
        if cur and csum + ul > budget:
            out.append("\n\n".join(cur))
            last, ll = cur[-1], blen(cur[-1])   # carry one small unit as overlap
            cur, csum = ([last], ll) if ll <= overlap else ([], 0)
        cur.append(u); csum += ul
    if cur:
        out.append("\n\n".join(cur))
    return [c.strip() for c in out if c.strip()]


def chunk_document(text: str, preset: dict) -> list[str]:
    if preset["table_aware"]:
        if preset.get("tokenizer") == "minilm":
            blen, windows, spec = wp_body, wp_windows, 2
        else:
            blen, windows, spec = ntok, token_windows, 0
        return chunk_table_aware(text, preset["target_tokens"],
                                 preset["overlap_tokens"], preset["max_tokens"],
                                 blen, windows, spec)
    return chunk_by_tokens(text, preset["target_tokens"], preset["overlap_tokens"])


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def corpus_files() -> list:
    files = []
    for p in sorted(config.TRANSFORMED_DIR.glob("treasury_bulletin_*.txt")):
        y, _ = file_year_month(p.name)
        if config.START_YEAR <= y <= config.END_YEAR:
            files.append(p)
    return files


def build_chunks(preset_name: str, files: list) -> list[dict]:
    preset = config.CHUNK_PRESETS[preset_name]
    chunks = []
    for path in files:
        year, month = file_year_month(path.name)
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        for i, piece in enumerate(chunk_document(text, preset)):
            if not piece.strip():
                continue
            chunks.append(
                {
                    "chunk_id": f"{path.stem}_chunk_{i:03d}",
                    "text": piece,
                    "source_file": path.name,
                    "year": year,
                    "month": month,
                }
            )
    return chunks


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def main():
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_questions()
    write_jsonl(config.OUT_DIR / "questions.jsonl", questions)

    files = corpus_files()
    coverage = {f for q in questions for f in q["correct_files"]}

    print("=" * 64)
    print(f"Corpus window : {config.START_YEAR}-{config.END_YEAR}")
    print(f"Corpus files  : {len(files)}")
    print(f"Questions kept: {len(questions)}")
    print(f"  difficulty  : {dict(Counter(q['difficulty'] for q in questions))}")
    print(f"  multi-file  : {sum(1 for q in questions if len(q['correct_files']) > 1)}")
    print("=" * 64)

    for name in config.CHUNK_PRESETS:
        chunks = build_chunks(name, files)
        write_jsonl(config.OUT_DIR / f"chunks_{name}.jsonl", chunks)
        toks = [ntok(c["text"]) for c in chunks]
        produced = {c["source_file"] for c in chunks}
        missing = coverage - produced
        print(f"\n[{name}]  chunks={len(chunks)}  "
              f"avg_tokens={sum(toks)//len(toks)}  "
              f"max_tokens={max(toks)}  chunks/file={len(chunks)/len(files):.1f}")
        print(f"  ground-truth files with 0 chunks: "
              f"{len(missing)}  ({'OK' if not missing else sorted(missing)})")
        sample = chunks[len(chunks) // 2]
        print(f"  sample chunk_id: {sample['chunk_id']} "
              f"(year={sample['year']}, month={sample['month']})")
        print(f"  sample text[:180]: {sample['text'][:180]!r}")

    print(f"\nWrote outputs to {config.OUT_DIR}")


if __name__ == "__main__":
    main()
