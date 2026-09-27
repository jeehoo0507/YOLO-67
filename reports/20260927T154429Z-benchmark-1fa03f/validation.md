# Local correctness validation

Code commit: `8da9b44`.

This is a **4-image CPU CLI check on macOS**, not a 10k sustained ImageNet/A5000 benchmark.
The reported ImageNet estimate and PASS/FAIL exercise report formatting only. No server
performance conclusion is supported by this sample, and this configuration was not activated
for full generation.

- `./solo test -q`: **31 passed**, including real pretrained DINO key equivalence, parallel
  producer/consumer batching, resume recovery, corrupt output retry, atomic write interruption,
  invalid/empty labels, smoke stop, simulated CUDA OOM and unstable candidate rejection.
- `./solo check src tests`: passed.
- `sh -n solo`: passed.
- CLI benchmark: all smoke/scaling/confirmation images succeeded; 4 worker/batch combinations,
  2 repeats per candidate. Representative boxes inspected in `pseudo_preview.jpg`.
- Single class 0; mean boxes/image 1.75; zero-box ratio 0.0.
- Full-generation precondition: reduced CPU benchmark is rejected as unqualified.
- Original example files were read as inputs; generated labels/receipts were kept in ignored
  `work/benchmarks/` and removed after reports were written. Model/cache/input images are not
  included in this commit.

Inputs: upstream CutLER demo1.jpg, demo2.jpg, demo3.jpg, demo4.jpg, downloaded from
[facebookresearch/CutLER at cca0a270](https://github.com/facebookresearch/CutLER/tree/cca0a270cf68399efc8fd50df426b6d806e39416/maskcut/imgs).
Source license and provenance: [THIRD_PARTY.md](../../THIRD_PARTY.md).

Required next experiment on the execution server:

```bash
git pull --ff-only
./solo benchmark /mnt/data/imagenet/train
git add reports
git commit -m "exp: A5000 ImageNet pseudo benchmark"
git push
```
