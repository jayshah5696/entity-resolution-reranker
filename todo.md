# Entity Resolution Phase 2 -- Cross-Encoder Reranker
## Implementation TODO (TDD, Phased with Gate Checks)

**Author:** Jay Shah
**Date:** 2026-03-15
**Source plan:** `plan.md` (Obsidian vault)
**Phase 1 repo:** `/Users/jshah/Documents/GitHub/entity-resolution-poc` (read-only reference)

---

## Critical Issues Found During Plan Review

Before starting, the engineer must be aware of these discrepancies between the plan and Phase 1 reality:

1. **Phase 1 has NO `title` field.** Phase 1 profiles are `(first_name, last_name, company, email, country)`. The plan's Phase 2 data pipeline adds `title` and `middle_name` fields. This means Phase 2's eval against Phase 1 indexes will use queries that include title, but the index records do NOT have title. The eval pipeline must handle this: serialize queries in Phase 1 pipe format (5 fields, no title) when querying Phase 1 indexes, but use full COL/VAL (with title) for cross-encoder reranking. This is actually fine architecturally -- the bi-encoder retrieval uses Phase 1 format, the CE sees full records -- but the engineer must not accidentally include title in Stage 1 queries.

2. **Phase 1 experiment 007 (ablation dims) was never run.** The `experiments/007_ablation_dims/` directory in Phase 1 has only `config.json` and `notes.md` (status: pending). There are NO pre-built indexes there. The plan references "Phase 1 indexes on local disk" -- the engineer needs to either: (a) run Phase 1's index build for the GTE FT model, or (b) use one of the existing indexes from other Phase 1 experiments if they exist. Check `results/` directory for `gte_modernbert_base_ft_*.json` files to find which indexes were actually built.

3. **Dependency name errors in plan's pyproject.toml:**
   - `minHash~=1.0` -- no such PyPI package. Use `datasketch>=1.6` (provides MinHash/LSH).
   - `hf-xet>=1.3` -- verify this exists; plan says `HF_HUB_DISABLE_XET=1` anyway.
   - `doublemetaphone>=1.1` -- verify PyPI name is exactly this (might be `metaphone` or `doublemetaphone`).

4. **sentence-transformers v5 CrossEncoder API:**
   - The plan says `LambdaRankLoss`. The actual class in ST v5 is `LambdaLoss` (under `sentence_transformers.cross_encoder.losses`). Verify import path.
   - `CrossEncoderTrainer` is the correct trainer class (not `SentenceTransformerTrainer`).
   - ST v5 `CrossEncoder.predict()` applies softmax by default -- scores are [0,1]. The plan's threshold calibration must account for this (the HF model card confirms: "Sentence Transformers calls Softmax over the outputs by default").

5. **LLM model name:** `google/gemini-3.1-flash-lite-preview` -- verify this is still the correct OpenRouter model ID at implementation time. Gemini model names change frequently. Check https://openrouter.ai/models before using.

6. **Phase 1 serialization format mismatch:** Phase 1 uses `pipe` format (`Jay | Smith | Google Inc | jay@google.com | USA`). Phase 2 introduces `COL/VAL` format for cross-encoder input. The engineer must keep both formats: pipe for querying Phase 1 indexes, COL/VAL for CE input pairs.

---

## Phase Structure

Each phase has:
- **Inputs:** what must exist before starting
- **Tasks:** ordered implementation steps (TDD -- tests first)
- **Gate check:** what must pass before moving to next phase
- **Verification command:** exact command to run

---

## PHASE 0: Repo Init + Dependency Lock
**Goal:** Working repo, all dependencies resolve, empty test suite runs.
**Time estimate:** 2-3 hours

### Tasks

- [ ] **0.1** Create repo structure (all directories from plan Section 1)
  - `src/data/`, `src/models/`, `src/eval/`, `configs/`, `tests/data/`, `tests/models/`, `tests/eval/`, `tests/configs/`, `experiments/`, `data/raw/`, `data/pool/`, `data/pairs/`, `data/manifests/`, `notebooks/`, `writing/`
  - All `__init__.py` files in `src/` and `tests/` directories

- [ ] **0.2** Create `pyproject.toml`
  - Use dependencies from plan Section 2, but fix:
    - `minHash~=1.0` -> `datasketch>=1.6`
    - Verify `doublemetaphone` PyPI name
    - Verify `hf-xet` PyPI name (remove if not needed since we use `HF_HUB_DISABLE_XET=1`)
  - Pin `transformers==4.57.6`
  - `requires-python = ">=3.12,<3.13"`

- [ ] **0.3** Create `.python-version` -> `3.12`

- [ ] **0.4** Create `.gitignore`
  - `data/raw/`, `data/pool/`, `data/pairs/`, `configs/eval.yaml`, `*.parquet`, `*.pkl`, `__pycache__/`, `.venv/`, `*.egg-info/`, `.ruff_cache/`, `.pytest_cache/`, `.mypy_cache/`, `wandb/`, `*.onnx`

- [ ] **0.5** Create `configs/eval.yaml.example` with placeholder paths

- [ ] **0.6** Create `AGENTS.md` with all 15 items from plan Section 10

- [ ] **0.7** Create minimal `README.md` (architecture diagram reference, Phase 1 results table)

- [ ] **0.8** Create `conftest.py` in `tests/` with basic fixtures (sample records for each ethnicity group, small pool fixture)

- [ ] **0.9** Run `uv sync` -- must complete without errors

### Gate Check 0
```bash
# ALL must pass:
uv sync
uv run pytest tests/ -q                    # 0 tests collected, 0 failures
uv run python -c "import sentence_transformers; print(sentence_transformers.__version__)"  # >=5.0
uv run python -c "from sentence_transformers import CrossEncoder; print('CrossEncoder OK')"
uv run python -c "import polars; import lancedb; import modal; print('core deps OK')"
uv run python -c "from transformers import AutoModelForSequenceClassification; print('transformers OK')"
```

---

## PHASE 1: Configs + Config Tests
**Goal:** All YAML configs exist, load correctly, have required keys.
**Time estimate:** 1-2 hours
**Inputs:** Phase 0 gate passed

### Tasks

- [ ] **1.1** Write `tests/configs/test_configs.py` FIRST
  - Test: all 4 YAML files load without error
  - Test: `models.yaml` has exactly 4 model keys: `minilm_reranker`, `gte_reranker`, `bge_reranker_m3`, `granite_reranker`
  - Test: each model entry has `hf_id`, `license`, `params`, `context`, `fine_tuned` fields
  - Test: `training.yaml` has `training.epochs`, `training.batch_size`, `training.lr`, `training.seed`, `loss.phase1_loss`, `loss.phase2_loss`
  - Test: `data.yaml` has `sources`, `ethnicity_distribution`, `corruption_codes`
  - Test: `eval.yaml.example` has `phase1.index_root`, `eval.top_k_stage1`, `metrics`

- [ ] **1.2** Create `configs/models.yaml` (4 CE models from plan Section 4)

- [ ] **1.3** Create `configs/training.yaml` (from plan Section 6)

- [ ] **1.4** Create `configs/data.yaml` (source URLs, ethnicity targets, corruption code registry)

- [ ] **1.5** Update `configs/eval.yaml.example` with full schema from plan Section 7

### Gate Check 1
```bash
uv run pytest tests/configs/test_configs.py -v  # all pass
```

---

## PHASE 2: Serialization
**Goal:** COL/VAL serialization working, round-trip tested, pair format tested.
**Time estimate:** half day
**Inputs:** Phase 1 gate passed

### Tasks

- [ ] **2.1** Write `tests/data/test_serialize.py` FIRST
  - Test: `colval_serialize` produces correct format with all fields
  - Test: missing fields are omitted (not empty COL/VAL pairs)
  - Test: field order is deterministic: fn, ln, org, title, country
  - Test: `colval_pair` joins two serialized records with ` [SEP] `
  - Test: None/empty string fields are omitted
  - Test: whitespace-only fields are omitted
  - Test: special characters in values are preserved (no escaping)
  - Test: consistency -- same input always produces same output

- [ ] **2.2** Implement `src/data/serialize.py`
  - `colval_serialize(record: dict) -> str`
  - `colval_pair(record_a: dict, record_b: dict) -> str`
  - `pipe_serialize(record: dict) -> str` (Phase 1 compat -- for querying Phase 1 indexes)
  - Field map constant: `COLVAL_FIELD_MAP`

### Gate Check 2
```bash
uv run pytest tests/data/test_serialize.py -v  # all pass
```

---

## PHASE 3: Data Sources
**Goal:** All raw data sources download, parse, validate. Real data on disk.
**Time estimate:** 1-2 days
**Inputs:** Phase 2 gate passed

### Tasks

- [ ] **3.1** Write `tests/data/test_sources.py` FIRST
  - Test: `names-dataset` package loads, `get_top_names(n=50, country_alpha2="IN")` returns non-empty
  - Test: `nicknames` package loads, `NickNamer().nicknames_of("William")` contains "Bill"
  - Test: Census surnames CSV loads, has >160K rows, `name` and `count` columns, Smith is rank 1
  - Test: SSA names loads year files, has `name, sex, count` columns
  - Test: GLEIF CSV parses, has >1M rows, has `LegalName` column
  - Test: GLEIF `OtherEntityNames` is parseable (test extraction of aliases)
  - Test: O*NET alternate titles loads, >27K rows
  - Test: O*NET reported titles loads, >35K rows
  - Test: SEC EDGAR `company_tickers_exchange.json` loads, >5K entries
  - Test: `cleanco.basename("Acme Corporation")` returns `"Acme"`
  - **NOTE:** Some tests require downloaded data. Mark download tests with `@pytest.mark.slow`. Fast tests use the Python packages only (names-dataset, nicknames, cleanco).

- [ ] **3.2** Implement `src/data/sources.py`
  - `download_gleif(output_dir: Path) -> Path`
  - `parse_gleif(path: Path) -> pl.DataFrame` -- columns: `legal_name, other_names, country, legal_form, status`
  - `parse_gleif_aliases(df: pl.DataFrame) -> dict[str, list[str]]` -- `{legal_name: [alias1, ...]}`
  - `download_onet(output_dir: Path) -> Path`
  - `parse_onet_alternates(path: Path) -> dict[str, list[str]]`
  - `parse_onet_reported(path: Path) -> list[str]`
  - `download_census_surnames(output_dir: Path) -> Path`
  - `load_census_surnames(path: Path) -> pl.DataFrame`
  - `download_ssa_names(output_dir: Path) -> Path`
  - `load_ssa_names(path: Path, min_year: int = 1980) -> pl.DataFrame`
  - `load_names_dataset(country_alpha2: str, n: int = 500) -> tuple[list[str], list[int]]`
  - `load_nicknames() -> dict[str, set[str]]`
  - `download_edgar_tickers(output_dir: Path) -> Path`
  - `parse_edgar_tickers(path: Path) -> pl.DataFrame`

- [ ] **3.3** Run full downloads: `uv run python -m src.data.sources --download-all --output-dir data/raw/`

### Gate Check 3
```bash
uv run pytest tests/data/test_sources.py -v  # all fast tests pass
uv run pytest tests/data/test_sources.py -v -m slow  # all download/parse tests pass
# Verify files exist:
ls data/raw/gleif_golden_copy.csv  # or .zip
ls data/raw/onet_alternate_titles.txt
ls data/raw/onet_reported_titles.txt
ls data/raw/census_surnames.csv
ls data/raw/ssa_names/
```

---

## PHASE 4: Global Entity Pool
**Goal:** 50K base entity records with realistic global distribution.
**Time estimate:** 1-2 days
**Inputs:** Phase 3 gate passed

### Tasks

- [ ] **4.1** Write `tests/data/test_pool.py` FIRST
  - Test: pool has exactly 50K records
  - Test: all required columns present (`entity_id, first_name, last_name, company, title, email, country, ethnicity_group, name_script`)
  - Test: no duplicate `entity_id` values
  - Test: no duplicate emails
  - Test: name frequency is weighted -- Smith appears >=3x more than a rare surname (sample 1000 last names, check distribution is not uniform)
  - Test: ethnicity group distribution within +/-5% of targets (e.g., US/UK English = 25% +/- 5%)
  - Test: company names include real GLEIF names (at least 1000 unique real company names from GLEIF in pool)
  - Test: titles include O*NET titles (at least 500 unique O*NET-sourced titles)
  - Test: no Phase 1 entity_id overlap -- load Phase 1 `data/processed/triplet_source.parquet` and `data/eval/eval_profiles.parquet` entity_ids, assert zero intersection
  - Test: email patterns match expected distribution (>50% `firstname.lastname@` pattern)
  - Test: `name_script` column has at least 3 distinct values (latin, cjk, arabic, etc.)

- [ ] **4.2** Implement `src/data/pool.py`
  - `build_name_pool(ethnicity_targets: dict, census_df, ssa_df, names_data) -> dict[str, NamePool]`
  - `build_company_pool(gleif_df, edgar_df, n_companies: int = 40_000) -> list[CompanyRecord]`
  - `build_title_pool(onet_alternates, onet_reported) -> TitlePool`
  - `generate_email(first: str, last: str, company: str, rng) -> str`
  - `assemble_base_pool(n: int = 50_000, seed: int = 42) -> pl.DataFrame`
  - `validate_pool(df: pl.DataFrame) -> None` -- raises on any invariant violation

- [ ] **4.3** Generate pool: `uv run python -m src.data.pool --output data/pool/base_pool.parquet`

- [ ] **4.4** Generate `data/pool/pool_stats.json` (row count, ethnicity distribution, script distribution, company source counts)

### Gate Check 4
```bash
uv run pytest tests/data/test_pool.py -v  # all pass
# Manual inspection:
uv run python -c "import polars as pl; df = pl.read_parquet('data/pool/base_pool.parquet'); print(df.shape); print(df['ethnicity_group'].value_counts())"
```

---

## PHASE 5: Corruption Functions (Latin-Script)
**Goal:** All 28 rule-based corruption types implemented, one test per type.
**Time estimate:** 2-3 days
**Inputs:** Phase 4 gate passed

### Tasks

- [ ] **5.1** Write `tests/data/test_corrupt.py` FIRST -- one test per corruption code
  - Test: `test_all_corruption_codes_have_handlers()` -- every code in registry has a function
  - **Company (C1-C8):**
    - Test C1: `legal_suffix_swap` -- "Acme LLC" produces "Acme Ltd" or "Acme Inc" (not unchanged)
    - Test C2: `suffix_drop` -- "Microsoft Corporation" -> "Microsoft"
    - Test C3: `the_prefix_drop` -- "The Home Depot" -> "Home Depot"
    - Test C4: `ampersand_normalize` -- "Johnson & Johnson" -> "Johnson and Johnson"
    - Test C5: `company_abbreviation` -- known GLEIF alias pair (requires aliases dict fixture)
    - Test C6: `word_truncation` -- "Goldman Sachs Group" -> "Goldman Sachs"
    - Test C7: `rebrand` -- known rebrand pair (requires EDGAR formerNames fixture)
    - Test C8: `shorten_with_abbrev` -- abbreviation + suffix drop
  - **Name Latin (N1-N16):**
    - Test N1: `diacritic_strip` -- "Garcia" (with accent) -> "Garcia"
    - Test N2: `single_char_delete` -- result is 1 char shorter
    - Test N3: `keyboard_sub` -- substituted char is in `QWERTY_NEIGHBORS[original_char]`
    - Test N4: `ocr_sub` -- substitution is from `OCR_PAIRS` table
    - Test N5: `char_transposition` -- two adjacent chars swapped
    - Test N6: `name_field_swap` -- fn and ln are swapped
    - Test N7: `east_asian_order_swap` -- surname/given swap
    - Test N8: `first_initial` -- "Jay Smith" -> "J. Smith"
    - Test N9: `first_middle_initial` -- "Jay Michael Smith" -> "J. M. Smith"
    - Test N10: `drop_middle` -- "Jay Michael Smith" -> "Jay Smith"
    - Test N11: `middle_initial_only` -- "Jay Michael Smith" -> "Jay M. Smith"
    - Test N12: `last_initial` -- "Jay Smith" -> "Jay S."
    - Test N13: `nickname_sub` -- "William" -> one of {"Bill", "Will", "Billy", "Liam"}
    - Test N14: `phonetic_sub_english` -- known substitution from `PHONETIC_PAIRS`
    - Test N15: `hyphen_add_remove` -- "Mary Jane" <-> "Mary-Jane"
    - Test N16: `prefix_suffix_drop` -- "Dr. Smith" -> "Smith"
  - **Title (T1-T6):**
    - Test T1: `title_abbreviation` -- uses O*NET lookup
    - Test T2: `title_expansion` -- reverse lookup
    - Test T3: `title_reorder` -- token order changes
    - Test T4: `seniority_drop` -- "Senior" prefix removed
    - Test T5: `seniority_synonym` -- "Sr." <-> "Senior"
    - Test T6: `dept_abbreviation` -- "Engineering" -> "Eng"
  - **Email (E1-E2):**
    - Test E1: `email_format_variant` -- format pattern changes
    - Test E2: `domain_swap` -- domain changes to personal
  - **Cross-cutting:**
    - Test: every corruption produces text DIFFERENT from input
    - Test: `corrupt_record(record, codes)` applies all codes in sequence
    - Test: corrupted record preserves `entity_id`

- [ ] **5.2** Implement `src/data/corrupt.py`
  - Constants: `QWERTY_NEIGHBORS`, `OCR_PAIRS`, `PHONETIC_PAIRS`, `SENIORITY_MAP`, `DEPT_ABBREV`, `CORRUPTION_CODES`, `CORRUPTION_HANDLERS`
  - One function per code: `corrupt_c1()`, ..., `corrupt_e2()`
  - `corrupt_record(record: dict, codes: list[str], rng=None) -> dict`
  - `CORRUPTION_HANDLERS: dict[str, Callable]` -- registry mapping code to function

### Gate Check 5
```bash
uv run pytest tests/data/test_corrupt.py -v  # all 28+ tests pass, 0 failures
```

---

## PHASE 6: LLM Corruption (Non-Latin) + Quality Filter
**Goal:** LLM-generated corruptions for non-Latin names, CE quality filter working.
**Time estimate:** 1-2 days
**Inputs:** Phase 5 gate passed

### Tasks

- [ ] **6.1** Write `tests/data/test_corrupt_llm.py` FIRST
  - Test: prompt builder produces correct format for each ethnicity hint
  - Test: response parser extracts JSON array of strings
  - Test: malformed response (non-JSON) returns empty list, no crash
  - Test: CE quality filter removes pairs below threshold (mock CE returning known scores)
  - Test: batching produces correct number of API calls (mock client)
  - Test: output includes `corruption_code` = `NL1`-`NL7` as appropriate

- [ ] **6.2** Implement `src/data/corrupt_llm.py`
  - `_build_prompt(record: dict) -> str`
  - `_parse_response(response_text: str) -> list[str]`
  - `generate_nonlatin_corruptions(records: list[dict], client, model: str, batch_size: int = 20) -> list[dict]`
  - `filter_by_ce_score(pairs: list[dict], stock_ce, min_score: float = 0.35) -> list[dict]`
  - Use OpenRouter client pattern from plan Section 5.4

- [ ] **6.3** Verify OpenRouter API works:
  ```bash
  uv run python -c "
  from openai import OpenAI; import os
  c = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.environ['OPENROUTER_API_KEY'])
  r = c.chat.completions.create(model='google/gemini-2.5-flash-lite-preview', messages=[{'role':'user','content':'hello'}])
  print(r.choices[0].message.content)
  "
  ```
  **NOTE:** Verify correct model name at implementation time. May need to update from plan's `gemini-3.1-flash-lite-preview`.

### Gate Check 6
```bash
uv run pytest tests/data/test_corrupt_llm.py -v  # all pass (uses mocked LLM client)
# Manual: run one real LLM call to verify API connectivity
```

---

## PHASE 7: Negative Mining
**Goal:** All 7 negative strategies produce valid label=0 pairs. Deterministic pre-filter works.
**Time estimate:** 1-2 days
**Inputs:** Phase 5 gate passed (can run parallel with Phase 6)

### Tasks

- [ ] **7.1** Write `tests/data/test_negatives.py` FIRST
  - Test: each of 7 strategies (NEG1-NEG7) produces pairs where anchor and negative have different `entity_id`
  - Test NEG1: same company, different person -- company matches, name differs
  - Test NEG2: phonetic neighbor -- Double Metaphone codes match, strings differ
  - Test NEG3: common name -- name from top-500 Census, different company
  - Test NEG4: title function swap -- same seniority level, different O*NET group
  - Test NEG5: title level swap -- same function, different seniority
  - Test NEG6: random from pool -- entity_ids differ
  - Test NEG7: BM25-style hard negatives (mock BM25 scores)
  - Test: `is_false_negative()` returns True when email domains match
  - Test: `is_false_negative()` returns True when normalized company names match AND Jaro-Winkler > 0.92
  - Test: `apply_deterministic_filter()` removes false negatives from negative set

- [ ] **7.2** Implement `src/data/negatives.py`
  - `mine_same_company_diff_person(pool: pl.DataFrame) -> list[dict]`  (NEG1)
  - `mine_phonetic_neighbor(pool: pl.DataFrame) -> list[dict]`  (NEG2)
  - `mine_common_name_diff_company(pool: pl.DataFrame, census_df: pl.DataFrame, top_n: int = 500) -> list[dict]`  (NEG3)
  - `mine_title_function_swap(pool: pl.DataFrame, onet_groups: dict) -> list[dict]`  (NEG4)
  - `mine_title_level_swap(pool: pl.DataFrame) -> list[dict]`  (NEG5)
  - `mine_random(pool: pl.DataFrame) -> list[dict]`  (NEG6)
  - `is_false_negative(anchor: dict, candidate: dict) -> bool`
  - `apply_deterministic_filter(pairs: list[dict]) -> list[dict]`

### Gate Check 7
```bash
uv run pytest tests/data/test_negatives.py -v  # all pass
```

---

## PHASE 8: Boundary Mining + LLM Labeling
**Goal:** Bi-encoder boundary zone pairs labeled by LLM, AMBIGUOUS discarded.
**Time estimate:** 1 day
**Inputs:** Phase 4, 6, 7 gates passed

### Tasks

- [ ] **8.1** Write `tests/data/test_boundary.py` FIRST
  - Test: `find_boundary_candidates` returns pairs with cosine in [0.60, 0.90]
  - Test: LLM labeling call uses correct prompt format (mock client)
  - Test: `parse_llm_labels` extracts MATCH/NON-MATCH/AMBIGUOUS correctly
  - Test: AMBIGUOUS pairs are discarded from output
  - Test: output contains both MATCH and NON-MATCH labeled pairs
  - Test: malformed LLM response triggers retry (up to 2 retries, then skip)

- [ ] **8.2** Implement `src/data/boundary.py`
  - `load_phase1_biencoder() -> SentenceTransformer` -- loads `jayshah5696/er-gte-modernbert-base-pipe-ft`
  - `encode_pool(pool: pl.DataFrame, model, batch_size: int = 512) -> np.ndarray`
  - `find_boundary_candidates(embeddings: np.ndarray, pool: pl.DataFrame, low: float = 0.60, high: float = 0.90) -> list[tuple]`
  - `label_with_llm(pairs: list[tuple], client, model: str, batch_size: int = 50) -> list[dict]`
  - `parse_llm_labels(response_text: str) -> list[str]`
  - `discard_ambiguous(labeled_pairs: list[dict]) -> list[dict]`

### Gate Check 8
```bash
uv run pytest tests/data/test_boundary.py -v  # all pass (mocked bi-encoder and LLM)
```

---

## PHASE 9: Dataset Assembly + Split
**Goal:** Train/val/test parquets created with all invariants enforced.
**Time estimate:** half day
**Inputs:** Phases 5-8 gates passed

**THIS IS THE MOST CRITICAL GATE. If splits are wrong, everything downstream is invalid.**

### Tasks

- [ ] **9.1** Write `tests/data/test_split.py` FIRST -- CRITICAL INVARIANT TESTS
  - Test: `test_zero_entity_id_overlap_with_phase1` -- Phase 2 pool entity_ids share ZERO overlap with Phase 1 triplet entity_ids AND Phase 1 eval entity_ids
  - Test: `test_test_split_locked_from_training` -- ce_test.parquet entity_ids do NOT appear in ce_train.parquet or ce_val.parquet
  - Test: `test_label_distribution_balanced` -- train set positive:negative ratio is 45-55%
  - Test: `test_split_ratios_correct` -- train ~60%, val ~20%, test ~20% (within 2%)
  - Test: `test_all_corruption_types_represented_in_each_split` -- stratified by corruption type
  - Test: `test_no_near_duplicates_after_dedup` -- MinHash dedup actually removes near-dups
  - Test: `test_deterministic_with_seed` -- same seed produces identical splits
  - Test: `test_split_manifest_matches_files` -- manifest JSON row counts match parquet row counts

- [ ] **9.2** Implement `src/data/split.py`
  - `assemble_pairs(rule_positives, llm_positives, negatives, boundary_pairs) -> pl.DataFrame`
  - `minhash_dedup(pairs: pl.DataFrame, threshold: float = 0.90) -> pl.DataFrame`
  - `deterministic_split(pairs: pl.DataFrame, seed: int = 42, ratios: tuple = (0.60, 0.20, 0.20)) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]`
  - `validate_split(train, val, test) -> None` -- raises if any invariant violated

- [ ] **9.3** Run full data pipeline end-to-end
  ```bash
  uv run python -m src.data.pool
  uv run python -m src.data.corrupt --pool data/pool/base_pool.parquet
  uv run python -m src.data.corrupt_llm --pool data/pool/base_pool.parquet
  uv run python -m src.data.negatives --pool data/pool/base_pool.parquet
  uv run python -m src.data.boundary --pool data/pool/base_pool.parquet
  uv run python -m src.data.split
  ```

- [ ] **9.4** Generate `data/manifests/split_manifest.json` and `data/manifests/data_stats.json`

### Gate Check 9 -- HARD GATE (DO NOT PROCEED IF ANY FAIL)
```bash
uv run pytest tests/data/test_split.py -v  # ALL MUST PASS
# Manual verification:
uv run python -c "
import polars as pl
train = pl.read_parquet('data/pairs/ce_train.parquet')
val = pl.read_parquet('data/pairs/ce_val.parquet')
test = pl.read_parquet('data/pairs/ce_test.parquet')
print(f'Train: {train.shape}, Val: {val.shape}, Test: {test.shape}')
print(f'Train label dist: {train[\"label\"].value_counts()}')
# Verify no entity_id overlap:
train_ids = set(train['entity_id_a'].to_list() + train['entity_id_b'].to_list())
test_ids = set(test['entity_id_a'].to_list() + test['entity_id_b'].to_list())
print(f'Train-Test overlap: {len(train_ids & test_ids)} (MUST BE 0)')
"
```

---

## PHASE 10: HF Dataset Upload
**Goal:** Training pairs uploaded to HuggingFace Hub.
**Time estimate:** half day
**Inputs:** Phase 9 gate passed

### Tasks

- [ ] **10.1** Implement `src/models/upload_dataset.py`
  - Loads `ce_train.parquet` + `ce_val.parquet` ONLY (NEVER `ce_test.parquet`)
  - Sets `HF_HUB_DISABLE_XET=1`
  - Pushes to `jayshah5696/entity-resolution-ce-pairs-v2`
  - Saves `data/manifests/upload_manifest.json`

- [ ] **10.2** Run upload and verify
  ```bash
  export HF_HUB_DISABLE_XET=1
  uv run python -m src.models.upload_dataset
  ```

### Gate Check 10
```bash
# Verify dataset is accessible:
uv run python -c "
from datasets import load_dataset
ds = load_dataset('jayshah5696/entity-resolution-ce-pairs-v2')
print(ds)
print(ds['train'][0])
"
```

---

## PHASE 11: Cross-Encoder Inference Wrapper
**Goal:** CE loading, scoring, reranking, threshold calibration all working.
**Time estimate:** 1 day
**Inputs:** Phase 2 gate passed (serialization needed for input format)

### Tasks

- [ ] **11.1** Write `tests/models/test_crossencoder.py` FIRST
  - Test: `test_ce_loads_stock_minilm` -- `cross-encoder/ms-marco-MiniLM-L12-v2` loads without error
  - Test: `test_score_range_is_0_to_1` -- all scores in [0, 1] range (ST v5 applies softmax)
  - Test: `test_match_scores_higher_than_nonmatch` -- identical pair scores > clearly-different pair
  - Test: `test_colval_input_accepted` -- no error on COL/VAL formatted strings
  - Test: `test_batch_and_single_consistent` -- batch predict == single predict (within float tolerance)
  - Test: `test_rerank_returns_sorted_by_score` -- output sorted descending
  - Test: `test_calibrate_threshold_not_naive_05` -- calibrated F1 >= naive 0.5 threshold F1

- [ ] **11.2** Implement `src/models/crossencoder.py`
  ```python
  class CrossEncoderReranker:
      def __init__(self, model_key: str, cfg: dict, device: str = "cpu", model_path: str | None = None): ...
      def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray: ...
      def rerank(self, query: dict, candidates: list[dict], top_k: int) -> list[dict]: ...
      def calibrate_threshold(self, val_scores: np.ndarray, val_labels: np.ndarray) -> float: ...
  ```
  - Uses `sentence_transformers.CrossEncoder` internally
  - `predict()` handles COL/VAL serialization internally
  - `calibrate_threshold()` uses F1-maximizing sweep on val set (sklearn)

### Gate Check 11
```bash
uv run pytest tests/models/test_crossencoder.py -v  # all pass
```

---

## PHASE 12: Eval Metrics
**Goal:** All Phase 2 metrics implemented and tested with known values.
**Time estimate:** 1 day
**Inputs:** Phase 11 gate passed

### Tasks

- [ ] **12.1** Write `tests/eval/test_metrics.py` FIRST
  - Test: `compute_f1_at_threshold` returns correct F1 for hand-computed cases
  - Test: `compute_pr_curve` returns monotonically decreasing precision at higher recall
  - Test: `compute_recall_retention` returns 1.0 when true match stays in top-K after reranking
  - Test: `compute_recall_retention` returns <1.0 when true match drops out
  - Test: F1 at calibrated threshold >= F1 at naive 0.5
  - Test: edge cases -- all predictions correct (F1=1.0), all wrong (F1=0.0), empty input

- [ ] **12.2** Implement `src/eval/metrics.py`
  - `compute_f1_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float`
  - `compute_pr_curve(scores: np.ndarray, labels: np.ndarray) -> list[tuple[float, float, float]]`
  - `compute_recall_retention(stage1_ranking: list, stage2_ranking: list, true_id: str) -> float`
  - `calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]` -- returns (best_f1, best_threshold)

### Gate Check 12
```bash
uv run pytest tests/eval/test_metrics.py -v  # all pass
```

---

## PHASE 13: Phase 1 Index Loading + Eval Pipeline
**Goal:** Can load Phase 1 indexes, run Stage 1 retrieval, pipe into Stage 2 CE.
**Time estimate:** 1-2 days
**Inputs:** Phases 11, 12 gates passed

### Tasks

- [ ] **13.1** Write `tests/eval/test_load_phase1.py` FIRST
  - Test: Phase 1 index path configured and reachable (skip if not configured)
  - Test: loaded index returns results for a known query
  - Test: eval queries load with correct schema (query_id, entity_id, bucket, query_text_pipe)
  - Test: BM25 index loads (if configured)

- [ ] **13.2** Write `tests/eval/test_run_reranker.py` FIRST
  - Test: end-to-end Stage 1 -> Stage 2 produces valid results JSON
  - Test: results JSON matches expected schema (plan Section 7)
  - Test: latency fields populated (stage1_latency_ms, stage2_latency_ms)
  - Test: per_bucket results have all 6 bucket keys

- [ ] **13.3** Implement `src/eval/load_phase1.py`
  - `load_phase1_index(index_path: Path, model_key: str, device: str) -> tuple`
  - `load_phase1_eval_queries(eval_dir: Path) -> dict[str, pl.DataFrame]`
  - `load_bm25_index(bm25_path: Path) -> object`

- [ ] **13.4** Implement `src/eval/run_reranker.py` (argparse CLI)
  - Stage 1: query Phase 1 index -> top-K candidates
  - Stage 2: rerank with CE -> scored + reordered
  - Compute all metrics (recall@k, MRR, nDCG, F1@threshold, PR curve, recall_retention)
  - Output results JSON per plan schema
  - Report latency separately for Stage 1 and Stage 2
  - **CRITICAL:** Query Phase 1 indexes using Phase 1 pipe format. Feed CE using Phase 2 COL/VAL format. This format mismatch is intentional.

- [ ] **13.5** Implement `src/eval/aggregate.py`
  - Reads all `results/*.json`, validates schema
  - Writes `results/master.csv` and `results/report.md`

### Gate Check 13
```bash
uv run pytest tests/eval/ -v  # all pass
# Smoke test (requires Phase 1 index path configured in configs/eval.yaml):
uv run python -m src.eval.run_reranker \
  --stage1-model gte_modernbert_base \
  --stage1-index /path/to/phase1/index \
  --reranker minilm_reranker \
  --eval-queries /path/to/phase1/data/eval \
  --top-k-stage1 50 \
  --output results/smoke_test.json \
  --experiment-id smoke
```

---

## PHASE 14: Modal Training Script
**Goal:** CE fine-tuning runs on Modal A10G. Both targets train in parallel.
**Time estimate:** 1-2 days
**Inputs:** Phase 10 (dataset uploaded), Phase 11 (CE wrapper) gates passed

### Tasks

- [ ] **14.1** Write `tests/models/test_finetune_config.py` FIRST
  - Test: config schema has both fine-tune targets (`gte_reranker`, `granite_reranker`)
  - Test: GPU spec is `A10G`
  - Test: loss config has `phase1_loss: bce` and `phase2_loss: lambda_rank` (verify correct ST class name)
  - Test: curriculum ratios list has correct length for 5 epochs
  - Test: `HF_HUB_DISABLE_XET` is set in training script
  - Test: output repo names follow convention `jayshah5696/er2-ce-{model_key}-ft`

- [ ] **14.2** Implement `src/models/finetune_modal.py`
  - Mirror Phase 1 structure:
    - Modal image with pinned torch/transformers/flash-attn/sentence-transformers
    - `CurriculumTrainer` subclass of `CrossEncoderTrainer` (NOT SentenceTransformerTrainer)
    - Override `get_train_dataloader()` for hard negative curriculum
  - Loss schedule: BCE (epochs 1-3) -> LambdaLoss (epochs 4-5)
  - **CRITICAL API CHECK:** Verify `sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss` and `sentence_transformers.cross_encoder.losses.LambdaLoss` exist and work with `CrossEncoderTrainer`. Import these at the top and verify.
  - Push to HF Hub after training
  - W&B logging

- [ ] **14.3** Dry-run test
  ```bash
  uv run python -c "from src.models.finetune_modal import app; print('import OK')"
  modal run src/models/finetune_modal.py::finetune_one --model-key gte_reranker --dry-run True
  ```

### Gate Check 14
```bash
uv run pytest tests/models/test_finetune_config.py -v  # all pass
# Dry run succeeds without error
```

---

## PHASE 15: Training Execution
**Goal:** Both CE models fine-tuned, pushed to HF Hub, checkpoints on Modal Volume.
**Time estimate:** 1 day (mostly waiting for GPU)
**Inputs:** Phase 14 gate passed

### Tasks

- [ ] **15.1** Run training
  ```bash
  modal run src/models/finetune_modal.py::run_all
  ```

- [ ] **15.2** Verify models on HF Hub
  ```bash
  uv run python -c "
  from sentence_transformers import CrossEncoder
  ce = CrossEncoder('jayshah5696/er2-ce-gte-reranker-ft', automodel_args={'torch_dtype': 'auto'})
  scores = ce.predict([('COL fn VAL Jay COL ln VAL Smith', 'COL fn VAL Jay COL ln VAL Smith')])
  print(f'Self-match score: {scores}')  # should be high
  "
  ```

- [ ] **15.3** Verify Granite model similarly

### Gate Check 15
```bash
# Both models load and produce reasonable scores:
uv run python -c "
from sentence_transformers import CrossEncoder
for repo in ['jayshah5696/er2-ce-gte-reranker-ft', 'jayshah5696/er2-ce-granite-reranker-ft']:
    ce = CrossEncoder(repo, automodel_args={'torch_dtype': 'auto'})
    s = ce.predict([
        ('COL fn VAL Jay COL ln VAL Smith COL org VAL Acme', 'COL fn VAL Jay COL ln VAL Smith COL org VAL Acme'),
        ('COL fn VAL Jay COL ln VAL Smith COL org VAL Acme', 'COL fn VAL Mike COL ln VAL Jones COL org VAL Globex'),
    ])
    print(f'{repo}: match={s[0]:.3f}, nonmatch={s[1]:.3f}')
    assert s[0] > s[1], f'Match should score higher than non-match for {repo}'
"
```

---

## PHASE 16: Run All 7 Experiments
**Goal:** Full 7-system comparison matrix populated with results.
**Time estimate:** 1 day
**Inputs:** Phase 13 (eval pipeline), Phase 15 (trained models) gates passed

### Tasks

- [ ] **16.1** Set up experiment directories
  - Create `experiments/001_bm25_plus_minilm_zs/` through `experiments/007_gte_ft_plus_bge_m3_zs/`
  - Each has `config.json` and `notes.md` from plan Section 7

- [ ] **16.2** Run Experiment 001: BM25 + MiniLM ZS
  ```bash
  uv run python -m src.eval.run_reranker --stage1-model bm25 --reranker minilm_reranker --experiment-id 001
  git commit -m "exp(001): BM25 + MiniLM ZS -- {result}"
  ```

- [ ] **16.3** Run Experiment 002: BM25 + GTE CE FT

- [ ] **16.4** Run Experiment 003: GTE FT + MiniLM ZS

- [ ] **16.5** Run Experiment 004: GTE FT + GTE CE ZS

- [ ] **16.6** Run Experiment 005: GTE FT + GTE CE FT (MAIN RESULT)

- [ ] **16.7** Run Experiment 006: GTE FT + Granite FT

- [ ] **16.8** Run Experiment 007: GTE FT + BGE-M3 ZS

- [ ] **16.9** Run aggregation
  ```bash
  uv run python -m src.eval.aggregate
  ```

### Gate Check 16
```bash
# All result JSONs exist and are valid:
ls results/001_*.json results/002_*.json results/003_*.json results/004_*.json results/005_*.json results/006_*.json results/007_*.json
# Report has no empty cells:
uv run python -c "
import json, pathlib
for f in sorted(pathlib.Path('results').glob('0*.json')):
    d = json.loads(f.read_text())
    assert d['overall']['recall_at_10'] > 0, f'{f.name} has zero recall'
    print(f'{f.name}: R@10={d[\"overall\"][\"recall_at_10\"]:.3f}')
"
cat results/report.md
```

---

## PHASE 17: Final Verification + Writing
**Goal:** All results verified, blog post and paper addendum written.
**Time estimate:** 2-3 days
**Inputs:** Phase 16 gate passed

### Tasks

- [ ] **17.1** Run full test suite one last time
  ```bash
  uv run pytest tests/ -v --tb=short
  ```

- [ ] **17.2** Verify hypothesis outcomes (plan Section 11)
  - H5: Fine-tuned CE improves nDCG@10 by >=3pp on >=2 buckets? Compare exp 005 vs Phase 1 GTE FT alone
  - H6: Dense FT + CE FT beats BM25 + CE FT on `missing_email_company` by >=10pp R@10? Compare exp 005 vs exp 002
  - H7: GTE reranker > MiniLM reranker after FT by >=5pp nDCG@10? Compare exp 005 vs exp 003
  - H8: Real data > Faker data by >=3pp F1? (Requires Faker ablation -- may defer)
  - H9: Global ethnicity improves F1 on non-English subset by >=5pp? Check non-English test subset

- [ ] **17.3** Write `writing/BLOG_POST.md`

- [ ] **17.4** Write `writing/paper_addendum.md`

- [ ] **17.5** Final README.md update with results tables

### Gate Check 17
```bash
uv run pytest tests/ -v  # 0 failures
cat results/report.md  # complete
```

---

## Summary: Phase Dependencies

```
Phase 0 (repo init)
  |
Phase 1 (configs)
  |
Phase 2 (serialization)
  |
Phase 3 (data sources)
  |
Phase 4 (entity pool)
  |
  +------ Phase 5 (Latin corruptions)
  |         |
  |         +--- Phase 6 (LLM corruptions) ---+
  |         |                                   |
  |         +--- Phase 7 (negatives) ----------+
  |                                             |
  +--- Phase 8 (boundary mining) ------+       |
                                        |       |
                              Phase 9 (assembly + split) *** HARD GATE ***
                                        |
                              Phase 10 (HF upload)
                                        |
  Phase 11 (CE wrapper) ---- Phase 12 (metrics)
          |                          |
          +--- Phase 14 (Modal) ----+---- Phase 13 (eval pipeline)
                    |                         |
              Phase 15 (training) -----+-----+
                                       |
                              Phase 16 (run experiments)
                                       |
                              Phase 17 (verification + writing)
```

---

## Engineer Checklist: Before EVERY Commit

1. `uv run pytest tests/ -v` -- all pass
2. `uv run ruff check src/ tests/` -- no lint errors
3. `uv run ruff format --check src/ tests/` -- properly formatted
4. No `data/*.parquet` or `configs/eval.yaml` in staged files
5. No API keys or tokens in staged files
6. Commit message follows convention: `phase(N): {description}` or `exp(00N): {result}`

---

## Known Pitfalls Quick Reference

| # | Pitfall | Prevention |
|---|---------|-----------|
| 1 | Phase 1 has no `title` field | Use pipe format (no title) for Phase 1 index queries, COL/VAL (with title) for CE |
| 2 | ST v5 CrossEncoder.predict() returns [0,1] via softmax | Don't assume raw logits; threshold calibration accounts for this |
| 3 | LambdaLoss not LambdaRankLoss | Verify `from sentence_transformers.cross_encoder.losses import LambdaLoss` |
| 4 | `datasketch` not `minHash` | Use `datasketch>=1.6` for MinHash |
| 5 | Phase 1 exp007 indexes don't exist | Build indexes first or use existing ones from other experiments |
| 6 | Modal uses Python 3.11, local uses 3.12 | No 3.12-only features in Modal-run code |
| 7 | `HF_HUB_DISABLE_XET=1` always | Set before any HF Hub operation |
| 8 | `warmup_steps` not `warmup_ratio` | Deprecated in ST 3.4+ |
| 9 | ce_test.parquet is LOCKED | Never reference in training code; created before any training |
| 10 | LLM model names change | Verify OpenRouter model ID at implementation time |
| 11 | AMBIGUOUS labels | Discard -- never force into training |
| 12 | Frequency-flat name sampling | Always use `count` column as weight |
| 13 | Checkpoint sort lexicographic | Sort by integer extracted from dirname |
