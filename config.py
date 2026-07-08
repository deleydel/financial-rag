"""Central configuration for the Financial RAG project.

Everything that defines an experiment lives here so Baseline vs Engineered
is a change of preset, never a fork of the code.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_ROOT       = Path.home() / "Desktop" / "course" / "officeqa_data"
TRANSFORMED_DIR = DATA_ROOT / "treasury_bulletins_parsed" / "transformed"
CSV_PATH        = DATA_ROOT / "officeqa_full.csv"

PROJECT_ROOT = Path.home() / "Desktop" / "course" / "financial_rag"
OUT_DIR      = PROJECT_ROOT / "outputs"

# --------------------------------------------------------------------------
# Corpus window  --  THE knob.
# officeqa_full only has 3 questions entirely inside 2022-2025, which is too
# few for meaningful metrics. 2010-2025 gives 40 questions / 63 quarterly
# files while staying "recent" and fast to embed locally. Change START_YEAR
# to 2022 to reproduce the spec-literal (tiny) run.
# --------------------------------------------------------------------------
START_YEAR = 2010
END_YEAR   = 2025

# --------------------------------------------------------------------------
# Chunking presets (token-based; cl100k tokenizer)
# --------------------------------------------------------------------------
CHUNK_PRESETS = {
    # baseline: blind token windows, oversized on purpose -> will be truncated
    # by the embedder, which is part of why the baseline underperforms.
    "baseline":   {"target_tokens": 600, "overlap_tokens": 50,  "table_aware": False,
                   "max_tokens": None},
    # engineered: sized to MiniLM's ~256-token window so nothing is truncated;
    # big tables are split by rows with the header repeated on each piece.
    # sizes are WordPiece tokens (MiniLM's own tokenizer, 256-token window).
    "engineered": {"target_tokens": 210, "overlap_tokens": 30,  "table_aware": True,
                   "max_tokens": 256, "tokenizer": "minilm"},
}

# --------------------------------------------------------------------------
# Models (used in later phases)
# --------------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # same in both -> clean attribution
GEN_MODEL   = "claude-sonnet-5"
JUDGE_MODEL = "claude-sonnet-5"
TOP_K       = 5     # cutoff for the retrieval metrics (Hit/MRR/Recall)
GEN_K       = 25    # chunks fed to the generator (reader sees more than we score at)
GEN_MAX_TOKENS = 1024

# --------------------------------------------------------------------------
# RAG presets  --  Baseline vs Engineered is a change of preset, not of code.
# Every lever that differs between the two systems is a field here.
# --------------------------------------------------------------------------
CHROMA_DIR    = OUT_DIR / "chroma"
EMB_CACHE_DIR = OUT_DIR / "emb_cache"

RAG_PRESETS = {
    "baseline": {
        "chunks":              "chunks_baseline.jsonl",
        "use_metadata_filter": False,      # flat search, no Year/Month filtering
        "prompt_style":        "simple",   # plain "answer from context"
    },
    "engineered": {
        "chunks":              "chunks_engineered.jsonl",
        "use_metadata_filter": True,       # soft Year/Month pre-filter (Phase 3)
        "prompt_style":        "strict",   # grounded + "say Not found" (Phase 3)
    },
}
