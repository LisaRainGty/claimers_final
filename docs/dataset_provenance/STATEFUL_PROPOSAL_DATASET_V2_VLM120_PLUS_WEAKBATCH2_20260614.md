# Stateful Proposal Dataset v2

This report separates consumer-perception labels from triplet completion status.
Silver refute rows keep positive perception labels with lower reliability instead of being forced to negative.

## Inputs

- queue: `data/final/repaired_v1/combined_queue_vlm120_plus_weakbatch2_v1_20260614.jsonl`
- reviews: `['data/final/repaired_v1/full_pair_reconstruction_llm_claimreextract_next1000_after500_fewerclaims820_noimg_flash_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_negative_control_n0_aligned377_noimg_flash_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_claimreextract820_plus_negaligned377_vlm120_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_weak_after1500_batch2_noimg_v1_20260614.jsonl', 'data/final/repaired_v1/full_pair_reconstruction_llm_evidence_repair_weak_batch2_vlm_v1_20260614.jsonl']`

## Outputs

- all stateful rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_vlm120_plus_weakbatch2_all_20260614.jsonl`
- observed supervised rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_vlm120_plus_weakbatch2_supervised_20260614.jsonl`
- contrastive-eligible rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_vlm120_plus_weakbatch2_contrastive_20260614.jsonl`
- repair/unobserved rows: `data/final/repaired_v1/stateful_proposal_dataset_v2_vlm120_plus_weakbatch2_repair_20260614.jsonl`

## Summary

- `reviewed_rows`: `1388`
- `label_observed_rows`: `505`
- `contrastive_rows`: `282`
- `repair_rows`: `883`
- `missing_queue_rows`: `0`
- `y_perception`: `{'None': 883, '1': 334, '0': 171}`
- `sample_role`: `{'repair_unlabeled': 633, 'supervised_main': 298, 'lowinfo_unlabeled': 250, 'supervised_silver_ambiguous': 36, 'supervised_silver_evidence_incomplete': 99, 'supervised_silver_repair_needed': 56, 'supervised_silver_guarded': 16}`
- `promotion_state`: `{'repair_missing_claim': 445, 'main_positive_refute': 178, 'lowinfo_no_aligned_comment': 230, 'repair_missing_evidence': 82, 'repair_insufficient_product_evidence': 151, 'silver_conflicting_comment_relation': 35, 'silver_refute_insufficient_product_evidence': 57, 'silver_refute_missing_product_evidence': 42, 'repair_numeric_value_judgment': 5, 'silver_subjective_eval_attribute': 11, 'silver_schema_meta_attribute': 11, 'main_negative_support': 120, 'repair_identity_claim_value': 7, 'silver_commercial_promise_attribute': 10, 'silver_attribute_semantic_drift': 1, 'silver_enumeration_evidence_extra_values': 3}`
- `claim_found`: `{False: 445, True: 943}`
- `product_evidence_found`: `{False: 373, True: 1015}`
- `category_observed`: `{'apparel_and_underwear': 74, 'baby_kids_and_pets': 93, 'beauty_and_personal_care': 18, 'digital_and_electronics': 19, 'food_and_beverages': 93, 'general': 94, 'jewelry_and_collectibles': 8, 'shoes_and_bags': 35, 'smart_home': 32, 'sports_and_outdoor': 39}`
- `duplicate_claim_family_groups`: `59`
- `conflicting_claim_family_groups`: `4`
- `contrastive_masked_by_claim_family`: `16`
- `split`: `{'train': 353, 'test': 101, 'val': 51}`
- `split_leakage`: `{'leaky_rooms': 0, 'leaky_rows': 0, 'examples': {}}`
