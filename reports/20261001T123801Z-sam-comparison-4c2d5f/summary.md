# DINO v1/MaskCut vs SAM 2.1 Hiera-Tiny

[Dense-grid findings and selection criteria](review.md)

Status: complete; images completed: 5/5
Command: `./solo compare-sam data/sam-check5 --limit 5 --points-per-side 32`
Source: {'commit_sha': '268c0459fd0b1638fee8f146d02a62bdd755d41a', 'branch': 'codex/milestone-1', 'dirty': False}

- Qualitative comparison; no scored ground truth, mAP, or precision/recall.
- Different architectures, pretraining supervision and resolutions: not a pure ViT test.
- SAM is mask-supervised. No target annotations or manual points/boxes are used here.
- AMG may retain background, nested parts and merged groups; mask IoU is not objectness.
- No crops, CUDA extension, connected-component cleanup or dynamic mask fallback.
- 16x16 grid is a CPU preview setting (official default 32x32); may miss small objects.

One unmeasured complete warmup per method. One timed run/image, torch threads=4, FP32. Common PIL decode timed separately. DINO includes its resize/normalization, feature extraction, MaskCut and boxes. SAM includes preprocessing, image encoder, 1024 automatic point prompts in batches of 16, decoder, AMG and shared geometric filters. Loading/download/render/report writes excluded. Not a sustained benchmark.

| Method | Total boxes | Mean seconds/image |
|---|---:|---:|
| dino | 18 | 0.329 |
| sam | 40 | 18.175 |

![DINO boxes, SAM boxes, SAM masks](comparison-1.jpg)
