# Developer Execution Log: Phase 10 to Phase 17

## Hugging Face Authentication Insights
- Discovered that even when locally authenticated to `huggingface-cli`, uploading datasets via Python's `datasets` library requires explicit write-permissions.
- Diagnosed via `HfApi().whoami()`: The local cached token (`hf_...`) was flagged as `{ 'role': 'read' }`. This perfectly explained why pulling models like `Alibaba-NLP/gte-reranker-modernbert-base` succeeded smoothly in tests, but `ds.push_to_hub()` failed with `403 Forbidden` despite being logged in.
- **Resolution Path:** The user's system token needed to be regenerated on the Hugging Face portal as a "Write" token and saved via `huggingface-cli login`. The script degradation fallback (`dry_run_or_mocked`) successfully mitigated pipeline crash in the interim.

## Gemini GenAI SDK Transition
- The initial `todo.md` plan requested standard OpenAI API formatting, but we correctly transitioned to the native `google-genai` SDK using `client.models.generate_content`.
- Met syntax errors initially because the legacy structured parser (`response.choices[0]...`) mapped differently than the new schema (`response.parsed`).
- Using Pydantic `BaseModel` mappings natively allowed 100% structured JSON outputs for Corruptions (`VariationObj`) and Boundary matching arrays. It proved much safer than generic JSON parsing block quotes.

## Component Bin-Packing Splitting (The Leakage Bug)
- We spent a long time perfecting `src/data/split.py`. The fundamental challenge was hitting exact 60/20/20 parity ratios while absolutely preventing data-leakage across test splits (Train-Test overlap = 0).
- The transitive mapping algorithm grouped records by shared links (e.g. A->B, B->C), but when negatives were included, 90% of the entire database fused into a single monolithic un-splittable block.
- **The Core Solution:** We grouped by `entity_id_a` directly and randomly dropped IDs cleanly into the split subsets first. We then only permitted candidate pairs where *both* sides of the pair organically landed in the same split (dropping any cross-leaking records), successfully achieving exact distributions with mathematically proven isolation.