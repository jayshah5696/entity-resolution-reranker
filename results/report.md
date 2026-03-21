# Phase 2 Experiment Results

Generated: 2026-03-21 02:21 UTC

## 1. Summary

| Experiment | R@10 Overall | MRR@10 | S1 (ms) | S2 (ms) |
|------------|:------------:|:------:|:-------:|:-------:|
| 001 | 0.944 | 0.899 | 0.1 | 4.2 |

## 2. Per-Bucket Recall@10

| Experiment | pristine | missing_firstname | missing_email_company | typo_name | domain_mismatch | swapped_attributes |
|------------|:---:|:---:|:---:|:---:|:---:|:---:|
| 001 | 0.977 | 0.978 | 0.757 | 0.993 | 0.981 | 0.980 |

## 3. Delta vs Baseline (R@10)

| Experiment | pristine | missing_firstname | missing_email_company | typo_name | domain_mismatch | swapped_attributes | Overall |
|------------|:---:|:---:|:---:|:---:|:---:|:---:|:-------:|

## 4. Phase 2 specific metrics

| Experiment | R@50 | F1 Best | F1 Threshold | Retention |
|------------|:----:|:-------:|:------------:|:---------:|
| 001 | 0.988 | 0.000 | 0.500 | 0.988 |