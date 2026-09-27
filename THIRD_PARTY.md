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
