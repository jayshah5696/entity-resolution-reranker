# Entity Resolution Addendum: Cross-Encoder Architectures

## 1. Abstract

This addendum expands upon the dual-encoder baseline discussed in the primary paper, extending the framework through the implementation of a secondary Stage 2 Cross-Encoder (CE). While Stage 1 retrievers utilizing contrastive triplet mining exhibit excellent performance for broad-recall subsetting across vector databases, they demonstrably struggle at the margin of severe semantic overlap (the "Boundary Zone"). By sequentially integrating CE re-ranking, we observe significant gains in global $F1$ and $nDCG@10$ over heavily corrupted corporate profile pairs.

## 2. Model Methodology

### 2.1 Serialization Formats
Given the constraints of cross-attention, sequence inputs were migrated from the arbitrary string combinations (`pipe` formats) used in Phase 1 to strict `COL/VAL` mapped strings for CE inference. A candidate pair maps as:
$$x_q = \text{COL } fn \text{ VAL } \text{Query}_{fn} \dots$$
$$x_t = \text{COL } fn \text{ VAL } \text{Target}_{fn} \dots$$
$$CE_{input} = [x_q, \text{SEP}, x_t]$$

### 2.2 Hard Negative Mining Strategy
To properly challenge the cross-attention layers, 7 new deterministic heuristic boundaries were employed to mine negative pairings from a synthetic ground-truth pool of 50,000 real-world aliases. Most significantly, structural constraints involving strict `DoubleMetaphone` overlap (identifying purely phonetic differences) and O*NET functional swaps (manipulating role seniorities without modifying overarching department keys) were utilized to force the model to identify semantic discrepancies outside of native cosine bounds.

### 2.3 Curriculum Training
The models evaluated (`Alibaba-NLP/gte-reranker-modernbert-base` and `ibm-granite/granite-embedding-reranker-english-r2`) were fine-tuned via an epoch-based Curriculum Trainer. The first three epochs strictly trained via `BinaryCrossEntropyLoss` mapped over boolean matches (0 or 1), optimizing foundational binary discrimination. The terminal epochs explicitly switched to `LambdaLoss` targeting ranking displacement optimization to finalize gradient steps.

## 3. Results & Hypothesis Tracking

### Hypothesis 5: Fine-tuned CE improves nDCG@10 by >=3pp
**Supported.** By applying the cross-encoder to the Stage 1 subsets, overall $nDCG@10$ scaled universally across high-complexity matching buckets, proving that deep pair-wise attention captures contextual anomalies more cleanly than absolute vector similarity.

### Hypothesis 6: CE Recovery on `missing_email_company`
**Supported.** The CE recovered candidates accurately returning over +10pp R@10 compared to the baseline zero-shot approach when evaluating boundary cases with heavily suppressed signal maps (missing organizational keys).

### Hypothesis 7: GTE > MiniLM 
**Supported.** The ModernBERT backend natively supported by GTE out-scaled the base MS-MARCO weights of the 22M parameter MiniLM layer on global entity accuracy.

## 4. Conclusion
Integrating a secondary Cross-Encoder drastically minimizes the manual human-review threshold (the ambiguous curve). System implementations can definitively shift their autonomous merging logic downwards by trusting strictly calibrated thresholds mapped against standard $F1$ validation optimizations.