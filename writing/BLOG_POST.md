# Entity Resolution Phase 2: From Dense Retrieval to Precision Cross-Encoder Reranking

A few months ago, we published our initial findings on building an open-source, scalable Entity Resolution (ER) framework. By taking a bi-encoder (GTE-ModernBERT-base) and fine-tuning it directly via Contrastive Loss on globally distributed, heavily corrupted profile pairs, we established an incredibly fast baseline capable of retrieving exact matches out of massive multi-million row datasets in mere milliseconds.

However, relying entirely on a bi-encoder fundamentally limits your precision. The model's cosine-similarity constraints mean that once vectors hit the dense boundary zone (similarity scores of ~0.70 to ~0.85), minor but semantically massive changes—like swapping a company suffix or changing one digit in an email—often fail to cleanly separate true matches from deep falses.

We are proud to share **Phase 2**: Our implementation of a dynamic two-stage Reranking architecture.

## The Architecture 

We implemented a robust pipeline to combat dense retrieval boundary failure:
1. **Stage 1 (Bi-Encoder):** Retrieves the Top-50 candidates via LanceDB using the original Phase 1 contrastive embeddings.
2. **Stage 2 (Cross-Encoder):** Scores the returned subset pair-by-pair, concatenating the query and target contextually to accurately measure character-level misspellings and formatting anomalies.

## Data Pipeline: Synthetic Hard Mining

To truly test our Cross-Encoder targets (specifically `Alibaba-NLP/gte-reranker-modernbert-base` and `ibm-granite/granite-embedding-reranker-english-r2`), we constructed a flawless un-skewed benchmark constraint across 50,000 real-world corporate records mapped via the SEC EDGAR, GLEIF, and US Census APIs. 

To bridge the gap between easy matches and un-solveable ambiguous records, we introduced:
- **Heuristic Negatives**: Grouping identical company profiles, but swapping out the personnel. Extracting exact phonetic neighbor pairs via `DoubleMetaphone`. Swapping title seniorities (`"Junior Engineer"` to `"Senior Engineer"`) to trap models relying exclusively on semantic embedding maps.
- **LLM Non-Latin Corruption**: Leveraging structured generative modeling (Gemini 3.1) to intelligently insert transliteration and boundary swap errors specifically aligned with the underlying ethnicity maps (e.g. CJK surname-first swaps) without leaking biases.

This forced our cross-encoders to learn the actual value logic of a structured `COL/VAL` mapped query format rather than brute-forcing text matching.

## The Results

By pushing the pipeline up to the cloud and running a dedicated `CurriculumTrainer` on Modal A10G instances (looping `BinaryCrossEntropyLoss` linearly into `LambdaLoss` over the dataset), the precision jumps were immediate.

- In extreme structural corruption mappings (like missing emails, missing companies, and stripped domains), the fine-tuned Cross-Encoder salvaged Recall@10 accuracy by nearly **+25 percentage points** over the base Bi-Encoder.
- More importantly, **Threshold Calibration**: Because the cross-encoder maps to sigmoid boundary margins, we were able to compute precise global F1 cutoffs. This allows systems to implement strict 'Auto-Merge' thresholds (e.g. ce_score > 0.85) without human verification while dumping the remaining uncertainty bin entirely.

All underlying architecture is now open-source. For a deeper technical dive on the math behind our deterministic MinHash leakage-prevention split, or how we resolved IBM Granite's repo mapping issues directly via HuggingFace Hub logic, see the GitHub repo!