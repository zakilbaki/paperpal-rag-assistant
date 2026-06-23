# PaperPal RAG Architecture Decisions

## Purpose

This document records the reasoning and preliminary experiments used to design
PaperPal's retrieval-augmented generation roadmap. It distinguishes approved decisions
from recommendations that still require validation. The current application remains a
document-intelligence system with indexing and retrieval foundations; this record does
not claim that answer generation is implemented.

## Existing Baseline

PaperPal currently extracts PDF text, stores papers in MongoDB, generates summaries,
extracts keywords, and compares papers. MongoDB is accessed through a shared asynchronous
client, and the complete local application runs through Docker Compose.

## Approved Decisions

### MongoDB remains the source of truth

MongoDB will continue to store paper text, metadata, summaries, keywords, future RAG
chunks, citation metadata, and question-and-answer history. Semantic retrieval will use
a separate vector index containing embeddings, chunk IDs, paper IDs, and only the
metadata required for filtering.

This separation keeps application records durable and inspectable while allowing the
retrieval index to be rebuilt when the embedding model or chunking strategy changes.

### Initial embedding model: all-MiniLM-L6-v2

The initial retrieval implementation will use
`sentence-transformers/all-MiniLM-L6-v2`. It runs locally, requires no API key, produces
compact 384-dimensional vectors, and was faster than BGE-small in the preliminary local
experiment. Both candidates retrieved all expected passages in that experiment, so
speed and implementation simplicity were the differentiators rather than measured
retrieval-quality superiority.

MiniLM's effective input limit is approximately 256 word-piece tokens. PaperPal must
therefore avoid the previously considered 500-token chunks. The initial chunking target
is 200–220 tokens with 30–40 tokens of overlap, while preserving sentence and page
boundaries where possible. Section inference is intentionally not part of the RAG
baseline because heading heuristics proved brittle across papers and domains.

Changing the embedding model later will require a complete vector-index rebuild. The
model ID and index version must be stored with indexing metadata.

### Local persistent vector store: ChromaDB

ChromaDB is approved as PaperPal's initial vector store. MongoDB stores complete chunks
and citation metadata; Chroma stores normalized MiniLM embeddings, `chunk_id`, `paper_id`,
the index version, and minimal filter metadata. Chroma is persisted through a dedicated
Docker volume and can be reconstructed entirely from MongoDB.

The adapter is hidden behind a `VectorStore` interface. This keeps Atlas Vector Search or
another backend possible later without rewriting chunking or application persistence.
Embedded Chroma is appropriate for the current single-backend local deployment; a
multi-instance deployment would require server-mode or managed vector infrastructure.

## Preliminary Embedding Experiment

### Question

Could MiniLM or `BAAI/bge-small-en-v1.5` provide a viable local retrieval baseline before
choosing the production vector store?

### Method

- Twelve original synthetic scientific passages covered distinct technical subjects.
- Twelve answerable and three unanswerable questions were evaluated.
- Each model embedded the same passages and questions on CPU.
- Exact cosine similarity ranked every passage, avoiding vector-store configuration as
  a confounding variable.
- Recall@1, Recall@3, Recall@5, mean reciprocal rank, and execution time were recorded.

### Results

| Model | Recall@1 | Recall@3 | Recall@5 | MRR | Index time | Query time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM-L6-v2 | 1.000 | 1.000 | 1.000 | 1.000 | 0.045 s | 0.017 s |
| BGE-small-en-v1.5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.061 s | 0.036 s |

Times came from one small local cached-model run and are not production benchmarks.
Both models achieved perfect retrieval, showing that both are viable but also that the
synthetic corpus was too easy to establish a retrieval-quality winner. Raw cosine scores
from different embedding models are not directly comparable and cannot define a shared
relevance threshold.

### Interpretation

MiniLM was selected as the simpler and faster initial baseline, not because the experiment
proved it retrieves scientific evidence better. The choice must be revisited if evaluation
on representative papers shows inadequate recall or if longer chunks are necessary.

The temporary benchmark implementation, generated reports, synthetic dataset, model
downloads, and benchmark-specific tests are intentionally excluded from the final
repository. This decision record preserves the method and evidence without shipping
experimental scaffolding as a product feature.

## Decisions Still Pending

| Decision | Current recommendation | Status |
| --- | --- | --- |
| Answer generation | Optional API provider with retrieval-only fallback | Requires approval |
| Reranking | Exclude from the first version | Requires approval |
| Retrieval evaluation | Human-labelled questions with expected pages or chunks | Requires real papers |
| Retriever fine-tuning | Postpone until a measured baseline exists | Requires approval |

## Implemented Indexing Slice

PaperPal now preserves page text and character offsets during upload. The indexing
endpoint creates page-aware, sentence-aware, MiniLM-sized chunks, stores their complete
records in MongoDB, and replaces the paper's vectors in persistent Chroma. Chunk
identifiers are deterministic and the operation removes previous records, making retries
idempotent.

## Implemented Retrieval Slice

The retrieval endpoint accepts a question for one indexed paper and a configurable
`top_k` between 1 and 10, defaulting to 5. It embeds the question with the same MiniLM
model, filters Chroma strictly by `paper_id`, converts cosine distance to similarity,
and hydrates the ranked chunk IDs from MongoDB. Results include text, score, page,
nullable section metadata, and character offsets.

The server always filters by `paper_id` and `index_version`; retrieval searches all
indexed chunks in the selected paper. Section metadata remains nullable in the API for a
future robust parser, but the current RAG baseline does not infer or filter by sections.

Retrieval is intentionally limited to one paper in the first version to keep provenance
and evaluation clear. No fixed relevance threshold is applied yet because MiniLM scores
must be calibrated on representative answerable and unanswerable questions.

Answer generation and RAG evaluation are not yet implemented. The retrieval endpoint
returns ranked evidence rather than presenting that evidence as a generated answer.

## Implemented Context Builder

The retrieval endpoint now also returns a deterministic context packet for the future
answer model. It processes the ranked top-five results without another model call or a
similarity threshold. Chunks are merged only when their exact page-local character
ranges overlap and their page and section metadata match. Each resulting passage has a
stable citation label (`C1`, `C2`, ...), retains every contributing chunk ID, and includes
page, nullable section metadata, score, and character offsets.

The packet is capped at five passages and 1,200 MiniLM tokenizer tokens, including the
citation headers. It exposes `evidence_available`, which means only that retrieval
returned usable text. It deliberately does not claim that the evidence answers the
question; the unsupported-question experiment showed that cosine similarity cannot make
that decision safely. A future answer model must use its own tokenizer to validate the
final prompt budget.

The builder was evaluated with the same 24 real-paper questions used for retrieval:

| Context metric | Result |
| --- | ---: |
| Relevant-page retention (18 answerable questions) | 1.000 |
| Raw top-five source chunks | 120 |
| Resulting cited passages after overlap merging | 101 |
| Source chunks retained in cited passages | 120 |
| Mean context size | 859.4 tokens |
| Maximum context size | 1,073 tokens |
| Contexts exceeding 1,200 tokens | 0 |

This validates provenance preservation and budget enforcement on the current corpus; it
does not evaluate generated-answer correctness. Unsupported questions still receive a
context packet because no reliable abstention mechanism has been approved yet.

## Required Production Evaluation

Before claiming retrieval quality, PaperPal should be evaluated with three to five
representative scientific PDFs and at least twenty human-labelled questions. Expected
evidence should identify the relevant paper, page, or chunk. The evaluation
should report Recall@k, mean reciprocal rank, citation validity, latency, and behavior on
questions unsupported by the documents.

## Real-Paper Retrieval Evaluation

A preliminary production-pipeline evaluation used three original arXiv papers:
[Attention Is All You Need](https://arxiv.org/abs/1706.03762),
[BERT](https://arxiv.org/abs/1810.04805), and
[LoRA](https://arxiv.org/abs/2106.09685). The corpus contained 57 PDF pages and produced
332 chunks using the configured 220-token limit and 40-token overlap. Eighteen
human-labelled answerable questions and six document-unsupported questions were run
through the actual MiniLM and persistent Chroma implementation.

| Metric | Aggregate result |
| --- | ---: |
| Page Recall@1 | 0.778 |
| Page Recall@3 | 0.944 |
| Page Recall@5 | 1.000 |
| Page Recall@10 | 1.000 |
| MRR@10 | 0.854 |
| Mean warm query latency | 7.1 ms |
| P95 warm query latency | 7.7 ms |

Per-paper Page Recall@1 was 0.833 for Transformer, 0.833 for BERT, and 0.667 for LoRA.
Every labelled question found a relevant page within the first five chunks. Four questions
needed more than the first result, with BERT task-specific architecture requiring rank
five. These cases motivate keeping `top_k=5` and later testing reranking or hybrid
retrieval rather than assuming the first chunk is always sufficient.

Similarity scores did not cleanly separate answerable from unsupported questions.
Answerable top-1 scores ranged from 0.395 upward, while an unsupported BERT question
scored 0.711. A threshold of 0.571 optimized on this tiny set reached 0.861 balanced
accuracy, 0.889 answerable recall, and 0.833 unsupported-question rejection. This is
overfit diagnostic evidence, not an approved production threshold. Refusal should not
rely on cosine score alone.

The evaluation also exposed a metadata limitation: canonical section detection does not
fully represent every paper-specific heading and can become domain-specific quickly.
Section inference was therefore removed from the RAG baseline. Section remains nullable
metadata in the API, not a retrieval prefilter.
The downloaded PDFs, labelled questions, temporary Chroma index, and evaluation script
were intentionally kept outside the repository; this document preserves the method and
results without shipping benchmark scaffolding as a product feature.

## Chunk Size and Overlap Experiment

The current sentence/token chunker was tested unchanged on the 15-page *Attention Is All
You Need* PDF with two configurations: the production baseline of 220 tokens with 40-token
overlap, and a larger 250-token configuration with 20-token overlap. MiniLM's verified
maximum sequence length is 256 tokens, so larger values would truncate indexed content.

Five fixed questions covered the abstract, the problem addressed by the paper, encoder
layer count, scaled-attention rationale, and positional encoding. Ground-truth pages were
verified against rendered PDF pages before evaluation. Both configurations used the same
MiniLM embeddings, Chroma search, and top-five ranking.

| Configuration | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 220 tokens / 40 overlap | 65 | 0.600 | 0.800 | 0.800 | 0.700 |
| 250 tokens / 20 overlap | 54 | 0.600 | 0.800 | 0.800 | 0.700 |

Both settings ranked the abstract page second, failed to retrieve pages 1 or 2 for the
broad problem question, and ranked the correct page first for the other three questions.
The larger setting therefore reduced the index by 11 chunks but produced no retrieval
quality gain. Its lower overlap also separated part of the scaled-attention rationale
from the sentence stating the scaling action. The 220/40 baseline remains unchanged.

This experiment indicates that chunk size is not the primary cause of the two observed
failures. Future work should evaluate PDF-layout cleanup and retrieval/reranking changes
independently rather than increasing chunks beyond MiniLM's useful input length. The PDF,
rendered pages, temporary Chroma indexes, and experiment script remain outside the repo.

## PDF Cleaning Experiment

Before implementation, a coordinate-based cleaning prototype was tested on two PDFs:
*Attention Is All You Need* and BERT. The chunker, MiniLM model, Chroma retrieval, and
220-token/40-overlap configuration stayed fixed. The experiment compared raw PyMuPDF
plain text, geometry-only cleaning, and geometry plus aggressive line/hyphen normalization.

Geometry-only cleaning removed only high-confidence artifacts: narrow side arXiv
watermarks, standalone bottom page numbers, and repeated margin blocks when present. It
did not change retrieval metrics on either paper.

| Paper | Variant | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | raw | 65 | 0.600 | 0.800 | 0.800 | 0.700 |
| Transformer | geometry-only clean | 65 | 0.600 | 0.800 | 0.800 | 0.700 |
| Transformer | geometry + line normalization | 64 | 0.600 | 0.800 | 0.800 | 0.700 |
| BERT | raw | 109 | 0.800 | 0.800 | 0.800 | 0.800 |
| BERT | geometry-only clean | 109 | 0.800 | 0.800 | 0.800 | 0.800 |
| BERT | geometry + line normalization | 99 | 0.600 | 0.800 | 0.800 | 0.700 |

The aggressive normalization repaired 326 line-end hyphenations in BERT but changed chunk
boundaries enough to move the architecture answer from rank one to rank two. This is a
regression, so it should not be implemented as-is. Geometry-only cleaning is safe but did
not address the known abstract and broad-problem failures. Those failures remain ranking
issues, not primarily PDF-noise issues.

No production code was changed by this experiment. The temporary scripts, rendered pages,
and Chroma indexes remain outside the repository.

## Interview Summary

The embedding model converts questions and paper chunks into vectors; cosine similarity
ranks chunks by semantic proximity. Exact cosine search was used for model comparison so
the vector store did not influence the result. MongoDB remains the source of truth, while
the future vector store acts as a rebuildable retrieval index. MiniLM is a measured local
baseline whose limitations and replacement conditions are explicit.
