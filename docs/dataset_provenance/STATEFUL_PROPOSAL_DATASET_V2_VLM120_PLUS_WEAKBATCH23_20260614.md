# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/combined_queue_vlm120_plus_weakbatch23_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_fewerclaims820_noimg_flash_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_n0_aligned377_noimg_flash_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_claimreextract820_plus_negaligned377_vlm120_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_weak_after1500_batch2_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_weak_batch2_vlm_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_weak_after1500_batch3_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_weak_batch3_vlm_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_vlm120_plus_weakbatch23_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_vlm120_plus_weakbatch23_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_vlm120_plus_weakbatch23_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_vlm120_plus_weakbatch23_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `1650`
- `label_observed_rows`: `572`
- `contrastive_rows`: `321`
- `repair_rows`: `1078`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 1078, '1': 392, '0': 180}`
- `sample_role`: `{'repair_unlabeled': 780, 'supervised_main': 341, 'lowinfo_unlabeled': 298, 'supervised_silver_ambiguous': 40, 'supervised_silver_evidence_incomplete': 108, 'supervised_silver_repair_needed': 59, 'supervised_silver_guarded': 24}`
- `promotion_state`: `{'repair_missing_claim': 574, 'main_positive_refute': 216, 'lowinfo_no_aligned_comment': 271, 'repair_missing_evidence': 90, 'repair_insufficient_product_evidence': 164, 'silver_conflicting_comment_relation': 38, 'silver_refute_insufficient_product_evidence': 60, 'silver_refute_missing_product_evidence': 48, 'repair_numeric_value_judgment': 6, 'silver_subjective_eval_attribute': 15, 'silver_schema_meta_attribute': 19, 'main_negative_support': 125, 'repair_identity_claim_value': 7, 'silver_commercial_promise_attribute': 13, 'silver_attribute_semantic_drift': 1, 'silver_enumeration_evidence_extra_values': 3}`
- `claim_found`: `{False: 574, True: 1076}`
- `product_evidence_found`: `{False: 445, True: 1205}`
- `category_observed`: `{'apparel_and_underwear': 81, 'baby_kids_and_pets': 100, 'beauty_and_personal_care': 30, 'digital_and_electronics': 21, 'food_and_beverages': 98, 'general': 110, 'jewelry_and_collectibles': 8, 'shoes_and_bags': 45, 'smart_home': 38, 'sports_and_outdoor': 41}`
- `duplicate_claim_family_groups`: `73`
- `conflicting_claim_family_groups`: `6`
- `contrastive_masked_by_claim_family`: `20`
- `split`: `{'train': 401, 'test': 114, 'val': 57}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
