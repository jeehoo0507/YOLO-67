# Qualitative review and decision

All three frozen ViT-S models were actually executed on the same five images.
Input: 448px, CPU FP32, batch 1, one thread, explicit attention. Each image was repeated three times.
Input provenance and checksums are in provenance.json; checkpoints and revisions are in comparison.json.

| Model | Forward ms/image | Serial keys path ms/image | Serial tokens path ms/image |
|---|---:|---:|---:|
| v1 | 212.7 | 389.2 | 399.2 |
| v2 | 329.9 | 502.9 | 581.0 |
| v3 | 216.4 | 376.5 | 375.7 |

Serial paths sum decode, full forward, and the selected feature branch cut/box/atomic label write.
They exclude loading, downloads, PCA, visualizations, and process startup; not GPU throughput.
v2 uses 32x32 patches, versus 28x28 for v1/v3. This contributes to its higher forward latency.

## Observations (human-readable visual review, not scored ground truth)

- Raw keys at the existing tau=0.15: v1 follows the main dog/birds/ducks but has false positives.
- v2 raw keys hit the 12-box cap on all five samples, with obvious fragmentation/background boxes.
- v3 raw keys often produce broad image regions rather than individual objects.
- Final patch tokens improve some v2 examples relative to its raw keys; the bus scene still merges people.
- v3 final-token PCA maps visibly follow animal/person regions, but the same tau=0.15 yields broad boxes or misses.
- For v3 final tokens, the 5th percentile of pairwise cosine similarity is above 0.27 in all five samples.
  Thus more than 95% of patch pairs exceed tau=0.15: that graph threshold retains little discrimination.
- Fixed token thresholds 0.5, 0.7, 0.9 were tested identically on all models and all five images.
  They recover some objects but retain background/part false positives or merged instances.
  No one threshold was selected as an independent best-performance estimate.

## Decision

Keep v1 for the present reproducible pseudo-label baseline. Do not substitute v2/v3 raw keys without
calibrating the feature-to-mask stage. v3 final dense tokens remain a candidate for a different proposal
or few-shot matching stage, but this experiment did not test support matching, identity recognition,
YOLO accuracy, or a diverse crowd test set. It does NOT establish that v1 is generally better than v2/v3.

No mAP or accuracy ranking is provided because these five examples have no scored ground-truth annotations.
Do not infer instance correctness from box counts or colorful PCA maps.

The new command, weights, cache and outputs remain repository-local. Existing datasets, pseudo-labels,
production model settings and trained YOLO checkpoints were not changed.
