# Financial RAG Challenge — U.S. Treasury Bulletins

This project implements a Retrieval-Augmented Generation (RAG) system for answering financial questions using U.S. Treasury Bulletins from the Databricks OfficeQA dataset. A baseline RAG pipeline is compared with an engineered version to evaluate how chunking strategy, metadata-aware retrieval, and prompt engineering affect both retrieval and answer generation.

**Dataset:** https://huggingface.co/datasets/databricks/officeqa

---

# Results

The evaluation was conducted on a filtered OfficeQA subset covering Treasury Bulletins from **2010–2025**, resulting in **40 evaluation questions**. Retrieval metrics were computed using **K = 5**.

| Metric | Baseline | Engineered |
|:---|---:|---:|
| Hit Rate@5 | 22.5% | **52.5%** |
| MRR | 0.100 | **0.292** |
| Recall@5 | 11.9% | **38.6%** |
| Groundedness | 71.9% | **93.3%** |
| Factual Accuracy | 0.0% | **2.5%** |
| Hallucination Rate | 28.1% | **6.7%** |

---

# Architecture

| Component | Choice |
|:---|:---|
| Vector Database | ChromaDB (persistent cosine similarity search) |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | Anthropic Claude Sonnet 5 |
| Metadata | `year`, `month`, `source_file` |

---

# Baseline vs. Engineered Design

| Component | Baseline | Engineered |
|:---|:---|:---|
| Chunking | Fixed 600-token windows | Structure-aware chunking with table-aware splitting and embedder-aware chunk sizing |
| Metadata | None | Soft Year filtering with fallback to unrestricted semantic search |
| Prompt | Simple answer-from-context prompt | Strict grounded prompt with numerical reasoning guidance |
| Generation Context | Top 5 retrieved chunks | Top 25 retrieved chunks (retrieval metrics still evaluated at K = 5) |
| Embedding Model | MiniLM | MiniLM (unchanged for fair comparison) |

The embedding model was intentionally kept identical in both systems so that performance improvements can be attributed to engineering decisions rather than to a stronger embedding model.

---

# Prompt Design

The baseline and engineered systems used the same retrieved context format and the same final answer format so that both could be evaluated fairly.

The baseline prompt simply instructed the model to answer using the retrieved Treasury Bulletin excerpts and end with:

```text
Final answer: <value>
```

The engineered prompt introduced several additional constraints:

- Use **only** the retrieved excerpts.
- Perform simple numerical reasoning when required (e.g., differences, growth rates, percentages, or sums).
- Preserve the original units reported in the source.
- Cite the source file(s) used.
- Return **"Not found"** only when the retrieved evidence genuinely does not contain the required information.

These prompt modifications substantially improved answer quality, increasing **Groundedness** from **71.9%** to **93.3%** while reducing the **Hallucination Rate** from **28.1%** to **6.7%**.

---

# Evaluation Methodology

## Retriever Metrics

Retriever performance was evaluated using the ground-truth source files provided in `officeqa_full.csv`.

- **Hit Rate@5:** Fraction of questions where at least one correct source file appears among the top five retrieved results.
- **MRR (Mean Reciprocal Rank):** Average reciprocal rank of the first correctly retrieved source file.
- **Recall@5:** Fraction of required source files retrieved within the top five results. This is particularly important because many OfficeQA questions require information from multiple Treasury Bulletins.

These metrics are computed deterministically in Python without using an LLM.

## Generator Metrics

**Factual Accuracy** is computed deterministically in Python.

- Numeric answers are considered correct when they match the ground truth within **±1%**, after normalizing currency symbols, commas, percentages, and scale words such as *million* and *billion*.
- Non-numeric answers are evaluated using normalized text matching.

**Groundedness** and **Hallucination Rate** are evaluated using Claude Sonnet 5 as an LLM judge.

The judge compares each generated answer against the retrieved context and reports:

- Total factual claims
- Number of supported claims

The final metrics are computed as

- **Groundedness = Supported Claims / Total Claims**
- **Hallucination Rate = 1 − Groundedness**

---

# Project Structure

```text
config.py
    Configuration for corpus window, chunking presets, retrieval settings,
    prompts, and model selection.

phase1_process.py
    Data filtering, preprocessing, chunk generation, and metadata tagging.

rag.py
    Shared RAG engine for both Baseline and Engineered systems:
    build_index(), retrieve(), generate(), and answer().

phase4_evaluate.py
    Runs the complete evaluation and generates the final scorecard.

outputs/
    Processed questions, chunk files, Chroma indexes,
    results.csv, and scorecard.md.
```

---

# How to Run

Install the required packages:

```bash
pip install chromadb sentence-transformers anthropic huggingface_hub pandas tiktoken
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="YOUR_API_KEY"
```

Run the pipeline:

```bash
python3 phase1_process.py
python3 rag.py build baseline
python3 rag.py build engineered
python3 phase4_evaluate.py
```

The corpus window can be changed by modifying `START_YEAR` in `config.py`.

---

# Key Findings

- The engineered pipeline substantially improved retrieval performance, increasing **Hit Rate@5** from **22.5%** to **52.5%**, **MRR** from **0.100** to **0.292**, and **Recall@5** from **11.9%** to **38.6%**.

- Metadata-aware retrieval and structure-aware chunking produced the largest improvements in retrieval quality.

- The engineered prompt substantially improved answer reliability, increasing **Groundedness** from **71.9%** to **93.3%** while reducing the **Hallucination Rate** from **28.1%** to **6.7%**.

- Despite these improvements, **Factual Accuracy** remained low because many OfficeQA questions require multi-step statistical reasoning and aggregation across multiple Treasury Bulletins. This indicates that the remaining limitation lies in evidence coverage and numerical reasoning rather than unsupported generation.

---

# Limitations

- The evaluation corpus was expanded to **2010–2025** because the OfficeQA answer key contains only **three** usable questions within the 2022–2025 period. Expanding the window produced a statistically meaningful benchmark of **40 questions**.

- `all-MiniLM-L6-v2` performs well for lightweight semantic retrieval but is less effective for dense numerical tables.


---
