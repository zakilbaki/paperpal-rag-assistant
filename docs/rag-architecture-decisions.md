# Document retrieval design

PaperPal separates durable document records, a rebuildable vector index, and deterministic evidence assembly. This document describes the implementation in this repository and the recorded experiments supporting its design.

## Document storage and indexing

MongoDB stores paper text, page boundaries, summaries, keywords, complete chunks, and citation metadata. Chroma stores normalized embeddings, chunk and paper identifiers, index version, and retrieval metadata in a persistent Docker volume.

The indexer uses page-aware, sentence-aware chunks with a 220-token limit and 40-token overlap. It preserves character offsets. Deterministic chunk identifiers and replacement of previous records make re-indexing idempotent.

The embedding model is `sentence-transformers/all-MiniLM-L6-v2`, producing 384-dimensional vectors. Its effective 256-word-piece input limit constrains chunk size. Changing the embedding model requires rebuilding the index.

Section metadata is nullable. The baseline does not infer or filter by section because heading heuristics varied across scientific papers.

## Retrieval and reranking

A request identifies one paper and supplies a question. The same MiniLM model embeds the question. Chroma filters by `paper_id` and `index_version`, returning up to 20 vector candidates by default. MongoDB supplies the complete passages for those chunk identifiers.

A deterministic reranker combines **60% vector similarity and 40% lexical coverage** of meaningful question terms. The API returns the blended score alongside `vector_score`, `keyword_score`, and `vector_rank`. The final `top_k` ranges from 1 to 10, defaulting to 5.

The 60/40 blend followed a five-question *Attention Is All You Need* comparison in which MRR@5 increased from 0.607 to 0.707 relative to a 75/25 blend. This is a small selection experiment, not an independent retrieval benchmark.

## Citation context

The context builder merges chunks only when their exact page-local character ranges overlap and their page and section metadata match. It assigns stable citation labels, retains contributing chunk IDs and offsets, and records passage scores.

The default budget is **five passages and 1,200 MiniLM tokenizer tokens**, including citation headers. `evidence_available` means that usable text was returned. It does not establish that the passages answer the question.

This endpoint returns cited context. Generated-answer correctness is a separate evaluation question.

## Recorded retrieval evaluation

The documented evaluation used the original [Transformer](https://arxiv.org/abs/1706.03762), [BERT](https://arxiv.org/abs/1810.04805), and [LoRA](https://arxiv.org/abs/2106.09685) papers: 57 PDF pages, 332 chunks, 18 labelled answerable questions, and six unsupported questions.

| Metric | Recorded result |
| --- | ---: |
| Page Recall@1 | 0.778 |
| Page Recall@3 | 0.944 |
| Page Recall@5 | 1.000 |
| Page Recall@10 | 1.000 |
| MRR@10 | 0.854 |
| Mean warm query latency | 7.1 ms |
| P95 warm query latency | 7.7 ms |

Page Recall@1 was 0.833 for Transformer, 0.833 for BERT, and 0.667 for LoRA. One BERT question required rank five. These measurements come from the recorded local experiment; the labelled questions, temporary index, and evaluation script are not committed, so they are not an independently rerunnable benchmark package.

Similarity scores overlapped between answerable and unsupported questions. An unsupported BERT question reached 0.711, while answerable top scores started at 0.395. No cosine-only abstention threshold is applied.

## Context preservation

The context builder was evaluated on the same 24 questions.

| Metric | Recorded result |
| --- | ---: |
| Relevant-page retention, 18 answerable questions | 1.000 |
| Raw top-five source chunks | 120 |
| Cited passages after overlap merging | 101 |
| Source chunks retained | 120 |
| Mean context size | 859.4 tokens |
| Maximum context size | 1,073 tokens |
| Contexts over 1,200 tokens | 0 |

The measurement tests provenance retention and budgeting. Unsupported questions can still receive context.

## Representation experiments

A cached-model CPU comparison on twelve synthetic passages and twelve answerable questions gave Recall@1, Recall@3, Recall@5, and MRR of 1.000 for both MiniLM and BGE-small. Recorded index/query times were 0.045/0.017 seconds for MiniLM and 0.061/0.036 seconds for BGE-small. The small corpus did not establish a retrieval-quality winner.

A five-question chunking comparison on the 15-page Transformer PDF produced:

| Chunk size / overlap | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 220 / 40 tokens | 65 | 0.600 | 0.800 | 0.800 | 0.700 |
| 250 / 20 tokens | 54 | 0.600 | 0.800 | 0.800 | 0.700 |

Larger chunks reduced index size without improving this comparison. The 220/40 configuration was retained.

A separate PDF-cleaning experiment kept the chunker, model, and retrieval configuration fixed:

| Paper | Extraction variant | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | Raw | 65 | 0.600 | 0.800 | 0.800 | 0.700 |
| Transformer | Geometry-only cleaning | 65 | 0.600 | 0.800 | 0.800 | 0.700 |
| Transformer | Geometry and line normalization | 64 | 0.600 | 0.800 | 0.800 | 0.700 |
| BERT | Raw | 109 | 0.800 | 0.800 | 0.800 | 0.800 |
| BERT | Geometry-only cleaning | 109 | 0.800 | 0.800 | 0.800 | 0.800 |
| BERT | Geometry and line normalization | 99 | 0.600 | 0.800 | 0.800 | 0.700 |

Aggressive normalization repaired 326 BERT hyphenations but changed chunk boundaries and reduced Recall@1. It was not adopted. Geometry-only cleaning did not improve retrieval in these comparisons.

## Implementation references

[Chunking](../backend/app/rag/chunking.py) · [Reranking](../backend/app/rag/reranking.py) · [Context builder](../backend/app/rag/context.py) · [Tests](../backend/tests/)
