# Financial RAG Challenge — U.S. Treasury Bulletins

A Retrieval-Augmented Generation system that answers financial questions over U.S.
Treasury Bulletins ([Databricks OfficeQA](https://huggingface.co/datasets/databricks/officeqa)).
It compares a **Baseline** (naive) pipeline against an **Engineered** (improved)
one to quantify how chunking, metadata filtering, and prompting affect both the
retriever and the generator.

## Results (K = 5, 40 questions, corpus 2010–2025)

| Metric | Baseline | Engineered |
|---|---|---|
| Hit Rate@5 | 22.5% | **52.5%** |
| MRR | 0.100 | **0.292** |
| Recall@5 | 11.9% | **38.6%** |
| Groundedness | 71.9% | **93.3%** |
| Factual Accuracy | 0.0% | **2.5%** |
| Hallucination Rate | 28.1% | **6.7%** |



## Architecture

| Component | Choice |
|---|---|
| Vector DB | ChromaDB (cosine, persistent) |
| Embedder | `all-MiniLM-L6-v2` (identical in both systems) |
| Generator / Judge | Anthropic `claude-sonnet-5` |
| Metadata | `year`, `month`, `source_file` on every chunk |

**Baseline vs Engineered levers** (one config, no code fork — see `config.py`):

| Lever | Baseline | Engineered |
|---|---|---|
| Chunking | 600-tok blind windows | structure-aware, table-row splitting, sized to the embedder (≤256 tok) |
| Metadata filter | none | soft Year filter (falls back to plain search) |
| Prompt | plain | strict, grounded, computes from tables |
| Reader context | top-5 | top-25 (metrics still scored @5) |

## Pipeline

```
config.py            # all knobs: corpus window, chunk presets, RAG presets, models
phase1_process.py    # filter answer key -> chunk corpus -> tag metadata
rag.py               # shared engine: build_index / retrieve / generate / answer
phase4_evaluate.py   # runs both systems, writes results.csv + scorecard.md
outputs/             # questions.jsonl, chunks_*.jsonl, chroma/, results.csv, scorecard.md
```

## How to run

```bash
pip install chromadb sentence-transformers anthropic tiktoken pandas
export ANTHROPIC_API_KEY="sk-ant-..."

python3 phase1_process.py            # build the eval set + tagged chunks
python3 rag.py build baseline        # embed + index (local, cached)
python3 rag.py build engineered
python3 phase4_evaluate.py           # full scorecard (cached; safe to re-run)
```

Change the corpus window in one line (`START_YEAR` in `config.py`).

## Key findings
- The **retriever is the bottleneck**: high groundedness / low hallucination but
  near-zero accuracy means the model reads faithfully; the search hands it the
  wrong pages.
- **Metadata + structure-aware chunking more than doubled retrieval** (Hit@5,
  MRR, Recall@5) — the biggest single win.
- Absolute accuracy is low because these OfficeQA questions require aggregating
  many values across bulletins for advanced statistics; it is retrieval-bound.

## Notes / limitations
- MiniLM is a sentence embedder and is weak on dense numeric tables — a
  table/number-aware or hybrid lexical+vector retriever would be the next upgrade.
- Corpus widened to 2010–2025 because the answer key has only 3 questions inside
  2022–2025.
