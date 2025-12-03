# MVP vs Subgraph Injection — Retrieval Comparison

A short, visual comparison of retrieval performance for the MVP pipeline and the enhanced pipeline with subgraph injection.

---

## Summary

- Dataset: 30 queries
- MVP accuracy: 43.33% (13/30)
- Subgraph injection accuracy: 40.00% (12/30)
- Average retrieval time (MVP): 2.39 s
- Average retrieval time (Subgraph injection): 5.22 s

Conclusion (summary): subgraph injection currently lowers accuracy and ~doubles average retrieval time — not an improvement for this run.

---

## Method Comparison

The MVP (baseline) pipeline performs standard retrieval based on chunk similarity. In contrast, the subgraph injection pipeline retrieves the subgraph associated with the most similar chunks and injects new triplets, allowing the injector to delete, merge, or create links between the new chunk and the retrieved subgraph for richer context.

![Subgraph injection illustration](/home/viture-ai/Documents/Real-Time-Brain-Inspired-Video-Memory/data/images/subgraph.png)

*Figure: Illustration of subgraph retrieval and injection process.*

---

## Graph topology

Below are full-graph visualizations for each run. Note the node counts and edge density — the enhanced graph shows higher node / edge counts.

### MVP (baseline)

![MVP graph](/home/viture-ai/Documents/Real-Time-Brain-Inspired-Video-Memory/data/graph_images/mvp_93e9c82e-95d6-4864-8ac1-2ae70edfd961.png)

*Figure: MVP graph (graph layout).*

### Subgraph injection

![Subgraph injection graph](/home/viture-ai/Documents/Real-Time-Brain-Inspired-Video-Memory/data/graph_images/sub_graph_4fd0a841-a1f1-4bee-9359-4a5982bc3df9.png)

*Figure: Subgraph injection — noticeably denser and more connected.*

---

## Retrieval metrics comparison

![Metrics comparison](/home/viture-ai/Documents/Real-Time-Brain-Inspired-Video-Memory/data/metrics/metrics_comparison.png)

*Figure: Multiple network science metrics and a small caption with average total times.*

Key takeaways from the metrics plot and data:

- Node/edge counts: Subgraph injection produced more nodes and relationships overall.
- Density & avg degree: Subgraph injection graphs show slightly higher density and degree distribution.
 - Graph connectivity: Subgraph injection increases the number of (potentially relevant) connections, making the KG denser.
- Retrieval time: Subgraph injection average retrieval time is significantly higher (5.22s vs 2.39s), consistent with additional graph traversal and reranking overhead.
- Downstream correctness: MVP performs slightly better in retrieval accuracy (43.33% vs 40.00%).

---

## Short analysis

- What we expected: injecting subgraphs and richer relations should help retrieval by surfacing more context.
- What happened: the enriched graph increased retrieval cost and slightly decreased quality of answers for this test set.
- Why this matters: the added graph complexity is not free — it increases traversal and reranking cost and can introduce noise that reduces precision in downstream answers.

- Retrieval mismatch: while the injected subgraph increases relevant relations and paths, the current (naive) retrieval flow largely relies on chunk/entity similarity rather than leveraging relationship paths or multi-hop traversal patterns, so the added graph information is not fully exploited.

---


## Result (final verdict)

# **FAILURE — enhanced pipeline with subgraph injection is slower and slightly less accurate**

For the tested run: the enhanced pipeline with subgraph injection is a failure so far: it's slower and slightly less accurate.

*Report generated using final benchmark files and retrieval logs in the repository.*
