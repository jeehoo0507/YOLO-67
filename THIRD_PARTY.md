# Upstream provenance

## DINO

- Source: [facebookresearch/dino](https://github.com/facebookresearch/dino)
- Commit: `7c446df5b9f45747937fb0d72314eb9f7b66930a`
- Vendored file: `src/solo/vendor/vision_transformer.py`, from `vision_transformer.py`.
- Copyright (c) Facebook, Inc. and its affiliates; Apache-2.0, retained in
  [licenses/DINO.txt](licenses/DINO.txt).
- Changes: retain the ViT-S inference architecture; remove unused larger models / DINO training
  head; replace upstream `utils.trunc_normal_` with `torch.nn.init.trunc_normal_`.
- The wrapper computes the last block's norm1/qkv directly and returns keys without CLS, matching
  CutLER's qkv hook extraction. Earlier transformer blocks and positional interpolation are unchanged.
- Official pretrained ViT-S/16 checkpoint:
  [DINO weights](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth)
- SHA256: `1566d50496f27f52f07fea6094fa29b2fdd6fae89da65bdd3ebc3b24ef6b7eb7`.
- Checkpoint is downloaded to ignored `weights/`, never committed. No torch.hub source download.

## MaskCut / CutLER

- Source: [facebookresearch/CutLER](https://github.com/facebookresearch/CutLER)
- Commit: `cca0a270cf68399efc8fd50df426b6d806e39416`
- `src/solo/maskcut.py` is an adaptation of
  [maskcut/maskcut.py](https://github.com/facebookresearch/CutLER/blob/cca0a270cf68399efc8fd50df426b6d806e39416/maskcut/maskcut.py).
- Copyright (c) Meta Platforms, Inc. and affiliates; **CC BY-NC-SA 4.0**. The adapted MaskCut
  implementation retains these terms. Full upstream license: [licenses/CutLER.txt](licenses/CutLER.txt).
- Changes: NumPy features and CPU-only spectral solve, configurable graph/extra-object rejection,
  direct seed connected component via SciPy, no CRF, patch masks converted directly to normalized
  bounding boxes. The generalized normalized-cut eigenproblem, corner reversal, seed choice,
  iterative feature masking and overlap rejection follow the original algorithm.
- CPU cosine affinity and eigensolve use float64; DINO keys transfer as float32. Affinity values
  and active-patch cosine diagonals are checked before thresholding. NumPy's matmul FP flags can
  be spurious on macOS BLAS ([NumPy issue](https://github.com/numpy/numpy/issues/29820)); actual
  non-finite/invalid affinity values still fail the image and cannot create completion receipts.
- This baseline uses ViT-S/16 at 384, max objects 2. It does not claim reproduction of published
  CutLER results using ViT-B/8 at 480 plus CRF and detector-specific training losses.

Local development previews may use upstream `maskcut/imgs/demo*.jpg` examples. These inputs are
stored only in ignored `data/`; representative annotated JPEG grids may be included in local
correctness reports with their source recorded. No ImageNet data is bundled.

## YOLO11n / Ultralytics

- Source: [Ultralytics](https://github.com/ultralytics/ultralytics), using its
  [`ultralytics-opencv-headless`](https://pypi.org/project/ultralytics-opencv-headless/8.3.242/)
  distribution pinned to `8.3.242` in `uv.lock`.
- [License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE): AGPL-3.0, with a separate
  commercial license offered by Ultralytics.
- `./solo train` constructs the packaged `yolo11n.yaml` architecture with random weights.
  It trains only on SOLO's DINO/MaskCut single-class pseudo boxes. COCO's original detection
  annotations and COCO-pretrained YOLO weights are not used.

## COCO images

- [COCO 2017](https://cocodataset.org/#download) train2017 and val2017 images only.
- Downloaded from the official `images.cocodataset.org` S3 bucket using HTTPS path-style URLs.
- Images and archives are ignored by Git. No COCO annotation archive is downloaded.

## Optional DINO backbone comparison

- [timm](https://github.com/huggingface/pytorch-image-models), Apache-2.0, pinned to `1.0.24`.
  Used only by `./solo compare-backbones`, through a project-local uv dependency group.
- [DINOv2 ViT-S/14](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m),
  Apache-2.0; timm revision `4610ca143709d58a633b6397a74412c2c3842454`.
- [DINOv3 ViT-S/16](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m),
  [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/);
  timm revision `3bf4720a82ec2066db88137180ff1f83a675cef0`.
  This is the timm conversion, not a claim of bit-identical Meta reference outputs.
  Its model card documents RoPE numerical differences and omission of zero QKV biases.
- Checkpoints are downloaded to ignored repository-local `.cache/huggingface/`, loaded with
  safetensors and strict state matching, and identified by SHA256 in comparison reports.
- These are feature-only inference experiments. No v2/v3 weights or source code are vendored,
  and the v1 production pseudo-label fingerprint/resume contract remains unchanged.

## Optional SAM comparison

- Official [SAM 2](https://github.com/facebookresearch/sam2), revision
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`, Apache-2.0. Optional CUDA component is
  disabled with `SAM2_BUILD_CUDA=0`; its connected-component cleanup is not used.
- SAM 2.1 Hiera-Tiny official checkpoint, Apache-2.0, SHA256
  `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.
  Downloaded from Meta's `dl.fbaipublicfiles.com/segment_anything_2/092824/` to
  `weights/`. SHA matches the LFS object in Meta's
  [official model repository](https://huggingface.co/facebook/sam2.1-hiera-tiny/tree/main).
- Installed only by `./solo compare-sam` in the repository-local `sam-comparison`
  uv group. No source/weights are vendored. Code, dependency versions and build
  constraints are locked by `pyproject.toml` and `uv.lock`.
- SAM is trained with segmentation supervision. This experiment does not use
  target COCO annotations or human prompts, but is not strictly unsupervised.
