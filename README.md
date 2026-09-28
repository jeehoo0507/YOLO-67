# SOLO

Milestone 1: **unlabeled images → frozen DINO ViT-S/16 → CPU MaskCut → single-class YOLO boxes**.
ImageNet directory names are only file identifiers. No ImageNet labels or annotations are loaded.

## 서버에서 처음 실행

서버에 `uv` 실행 파일과 NVIDIA driver가 있으면 저장소를 받습니다. 데이터 출처는
**ImageNet-1K / ILSVRC2012 train**입니다. 전체 train TAR은 **147.9 GB**이며 이미지
1,281,167장이 1,000개 그룹에 들어 있습니다. 첫 실험에서 128만 장을 처리하는 시간을
줄이기 위해 **`./solo download`는 기본적으로 그룹당 100장, 총 100,000장만 준비합니다.**
[공식 ImageNet 서버](https://image-net.org/challenges/LSVRC/2012/2012-downloads)의
TAR에서 필요한 이미지만 읽고 연결을 닫아 147.9 GB 원본을 내려받지 않습니다. 실제 저장
크기와 다운로드량은 서버 실행 결과에 기록됩니다. ImageNet 그룹명은 파일 식별자로만
사용하고 class label과 annotation은 학습에 사용하지 않습니다. Index는
[`assets/imagenet_train_index.json`](assets/imagenet_train_index.json)에 포함되어 있어
서버에서 헤더 1,000개를 다시 조회하지 않습니다. [ImageNet 이용 조건](https://www.image-net.org/download.php)을
확인하세요. 공식 서버가 부분 요청을 제공하지 않으면 전체 파일을 대신 받지 않고 중단합니다.
중단 후 같은 명령을 다시 실행하면 이미 준비된 그룹은 재사용합니다.

```bash
git clone https://github.com/jeehoo0507/YOLO-67.git
cd YOLO-67
./solo download && ./solo benchmark && ./solo generate
```

`./solo benchmark`가 프로젝트 내부 `uv` 환경 준비 → checkpoint 검증 → 100장 smoke test →
worker/batch autotune → 10,000장 sustained benchmark → `reports/` 생성까지 실행합니다.
터미널 마지막에 report 경로와 평균 img/s, **10만 장 처리 예상 시간**, ImageNet-1K
전체 128만 장으로 확장했을 때의 예상 시간이 나옵니다. 24시간 PASS/FAIL은 현재 subset과
원래의 전체 ImageNet 목표를 각각 표시합니다.
`&&`는 benchmark가 성공적으로 끝났을 때만 전체 이미지 처리(`generate`)를 시작합니다.
**24시간 목표 FAIL은 속도 평가일 뿐 명령 실패가 아니므로, 이 경우에도 generate가 이어집니다.**

이미 클론한 서버라면 다시 클론하지 말고 최신 코드를 받은 뒤 같은 한 줄을 실행합니다.

```bash
cd YOLO-67
git pull --ff-only
./solo download && ./solo benchmark && ./solo generate
```

**라벨 생성이 끝난 뒤 바운딩 박스 예시 보기:**

```bash
./solo preview
```

결과 이미지는 `reports/pseudo_preview.jpg`에 저장됩니다. GitHub에서도 보려면 아래
`git add reports` → `git commit` → `git push` 명령으로 결과를 올리세요.

benchmark 리포트를 먼저 확인하고 전체 처리 여부를 결정하려면 `./solo benchmark`만 실행한 뒤
아래 `./solo generate`를 따로 실행하세요. 실험 결과를 Git으로 전달할 때:

```bash
git add reports
git commit -m "exp: ImageNet pseudo-label results"
git push
```

따로 진행하거나, 전체 처리가 중단되었을 때는 아래 명령을 실행합니다. 중단되면
같은 명령을 다시 실행해 완료된 이미지부터 이어갑니다.

```bash
./solo generate
```

환경 확인만 하려면 `./solo doctor`를 실행할 수 있습니다.

`./solo`는 환경 변수를 설정하고 `uv run --frozen`으로 필요한 환경을 자동 동기화합니다.
`.venv` 활성화나 별도 Python 명령은 필요 없습니다. Linux x86_64를 GPU 실행 대상으로
합니다. Python 3.11.13, PyTorch 2.7.1과 CUDA 12.6 user-space dependencies는
`uv.lock`으로 고정되고 저장소 안에 설치됩니다. Host CUDA toolkit 설치는 필요 없습니다.
NVIDIA driver는 변경하지 않습니다. DINO checkpoint는 SHA256 검증 후 사용합니다.
`./solo download`가 받은 이미지는 `data/imagenet/train/`에 준비하며, 저장소를 삭제하면
다운로드 데이터도 함께 삭제됩니다. 이미 그 폴더에 10만 장 이상이 있다면 재다운로드 없이
그룹당 100장을 선택합니다. 나머지 기존 파일은 자동 삭제하지 않지만 benchmark와 generate는
`data/imagenet/train/.solo-selection.txt`에 기록된 **선택 이미지 10만 장만 처리합니다.**
기존 외부 ImageNet 데이터는 읽기 전용으로
`./solo benchmark /실제/경로 && ./solo generate /실제/경로`처럼 쓸 수 있습니다.
이 경우 외부 데이터는 복사·수정하지 않습니다. 이미 받은 공식
`ILSVRC2012_img_train.tar`가 이미 **외부 경로**에 있다면
`./solo download --archive /실제/파일`로 같은 10만 장 sample을 준비할 수 있습니다.
원본 archive는 읽기만 합니다. 장수를 바꾸려면 `./solo download --images-per-group 200`
(총 20만 장), 전체를 원할 때만 `./solo download --mode full`을 사용합니다.
기존 100GB 또는 다운로드 속도 기준 자동 모드는 `--mode 100gb`, `--mode auto`로 남겨 두었습니다.
이전에 저장소 안에 정상적으로 받아 둔 전체 TAR이 있으면 재사용합니다. 완료되지 않은
`.part` 파일은 건드리지 않으며 새 부분 요청 방식에서는 사용하지 않습니다.

기본 설정은 [configs/baseline.toml](configs/baseline.toml)에 있습니다. 변경한 별도 TOML은
`./solo benchmark --config configs/custom.toml`로 사용합니다. `generate`에도 같은
config를 지정합니다. 알 수 없는 설정 이름은 에러로 처리합니다.

## Benchmark

1. 전체 이미지 목록을 읽고 relative path hash + seed로 deterministic subset을 고릅니다.
   class 폴더별로 첫 이미지들만 고르는 편향을 피합니다.
2. 실제 이미지 100장으로 decode, pretrained features, MaskCut, bbox, YOLO 파일·receipt
   재검증, JPEG preview를 확인합니다. Smoke 실패나 모든 이미지에서 zero-box면 중단합니다.
3. 같은 1,000장을 workers × batch 조합마다 **2회 새로 처리**합니다. 기존 라벨을 skip하지
   않습니다. worker 수는 physical/effective core, affinity, Linux cgroup quota와 RAM으로
   제한합니다. batch는 8/16/32/64/128을 순서대로 탐색하며 CUDA OOM 후 더 큰 batch를
   시도하지 않습니다. 후보별 성공 여부와 오류는 보존합니다.
4. 반복 throughput의 min/max가 0.85 이상인 후보 중 **최저 반복 throughput**이 가장 높은
   조합을 고릅니다. 최고 순간 GPU 속도는 선택 기준이 아닙니다.
5. 선택한 조합으로 10,000장을 다시 처리합니다. 평균 img/s, elapsed, 현재 subset 전체
   예상 시간과 ImageNet-1K 전체 외삽 시간, 각 24시간 PASS/FAIL을 출력합니다.
   속도 FAIL은 실행 실패가 아닙니다.
   이미지 처리 오류가 있으면 best config를 활성화하지 않습니다.

CPU만 있는 환경에서도 correctness 검증은 가능합니다. 축소된 테스트 config를 쓸 수 있지만
**10k confirmation과 2회 반복을 완료한 설정만** 현재 dataset 전체 generation의 기본 설정이 됩니다.
GPU가 없는 결과는 리포트에 CPU 테스트로 표시됩니다. 실제 A5000 성능은 서버 결과로 판단합니다.

timed wall은 image decode 시작부터 DINO, queue/IPC, MaskCut, bbox, label/receipt atomic
write, 마지막 worker drain까지 포함합니다. discovery, 프로세스 준비와 kernel warmup,
resume 검사는 별도 기록합니다. stage ms/image는 병렬 worker의 누적 service time이므로
합산하면 elapsed와 같지 않습니다. 파일시스템 cache를 강제로 비우지 않습니다. 10k
confirmation은 캐시를 포함한 실행 결과이며 더 큰 데이터셋의 sustained IO는 달라질 수 있습니다.

단일 GPU owner process가 batched DINO를 실행하고 CPU-only spawn worker pool에 feature를
보냅니다. decode thread prefetch와 완료 callback을 사용하며 feature backlog는
`workers + queue_batches × batch`로 제한합니다. CPU worker는 CUDA를 사용하지 않고 BLAS
thread 수를 1로 고정합니다. 무한 queue나 이미지별 GPU worker를 만들지 않습니다.

## 출력과 resume

```text
pseudo/imagenet/
├── labels/<relative-image-stem>.txt
├── receipts/<relative-image-path>.json
└── .generation.lock
```

예: `n01440764/abc.JPEG` → `labels/n01440764/abc.txt`. 같은 폴더에서 동일 stem의 JPG/PNG가
겹치면 discovery에서 충돌을 명시하고 중단하므로 라벨을 덮어쓰지 않습니다. Milestone 2에서
원본 이미지를 복사하지 않는 YOLO dataset adapter를 연결할 예정입니다.

```text
0 0.43120000 0.52210000 0.31500000 0.40120000
```

모든 클래스는 `0 = object`입니다. 이미지 전체를 384×384로 resize하고 mask의 patch 경계를
원본 이미지의 normalized 좌표로 환산합니다. aspect ratio 필터와 quality 통계는 원본 pixel
크기를 사용합니다. CRF, segmentation boundary refinement, DINO fine-tuning은 없습니다.

같은 `./solo generate ...` 명령으로 재시작합니다. 각 이미지의 source size/mtime,
dataset root, algorithm/config fingerprint, label SHA256, row 형식, box 개수를 검증합니다.
정상적인 zero-box 파일도 완료 receipt가 있어야 skip합니다. 빈 파일만 남거나 잘린 JSON,
checksum 불일치, 설정 변경은 해당 이미지만 다시 처리합니다. 라벨과 receipt는 같은 디렉터리의
temporary file에 쓰고 fsync + atomic rename합니다. receipt가 이미지별 완료 시점입니다.
단일 central state 파일의 손상 때문에 완료 작업을 잃지 않습니다.

첫 화면에서 완료된 이미지 수를 복구한 뒤 남은 이미지를 처리합니다. average/rolling img/s는
이번 실행에서 새로 성공한 이미지만 사용하므로 resume skip으로 부풀려지지 않습니다.
원본 크기 변경 검출은 size/mtime 기반이며 원본 내용 전체를 hashing하지 않습니다.
단일 output 디렉터리에는 advisory lock을 사용합니다. Ctrl+C / 종료 후 다시 실행하면 완료
receipt가 있는 이미지는 유지됩니다. 손상 이미지/CPU 오류는 성공으로 기록하지 않고 retry
대상으로 남깁니다. 오래된 `.tmp` 파일은 완료 검증에 사용하지 않습니다.

`generate`는 repository의 `work/best_config.json` 또는 추적된 `reports/*/best_config.json`에서
호환되는 10k 결과를 찾습니다. 모델·MaskCut·filter 설정, GPU/VRAM/CUDA runtime, 안전한 worker
limit, decode/queue/fsync 정책이 맞지 않으면 새 benchmark를 요구합니다.

## 작은 실험 결과만 Git으로 전달

```text
reports/<UTC-run-id>/
├── summary.md
├── benchmark.json
├── benchmark.csv
├── best_config.json
├── system_info.txt
├── subset.json
└── pseudo_preview.jpg
```

리포트에는 Git SHA/dirty 여부, command, 전체 configuration, system/model 정보, 반복별
throughput, DINO/MaskCut/IO/bbox timing, queue/backpressure, peak allocated VRAM, best 설정,
품질 histogram, 대표 JPEG grid, errors/warnings가 들어갑니다. 병목은 누적 worker service /
workers와 producer wall의 비교로 **추정**하며 GPU utilization 측정치로 오해하지 않도록 표시합니다.
preview는 기본 64장으로 하나의 JPEG에 합칩니다. 대용량 라벨과 원본 이미지는 report에 복사하지
않습니다. 실패한 실행도 가능한 범위에서 partial report를 남깁니다.
`./solo generate`가 끝나면 터미널에 `Bounding-box preview (64 images): .../pseudo_preview.jpg`
경로를 표시합니다. 이 파일은 실제 의사 라벨 박스를 원본 이미지 위에 그린 64장짜리 grid이며,
박스가 있는 예시를 우선 담습니다. `summary.md`에도 같은 이미지가 표시됩니다. 서버에서
아래처럼 `reports/`만 GitHub에 올리면 웹에서 preview를 바로 볼 수 있습니다.

생성이 끝난 뒤 예시만 다시 만들려면 다음 명령을 실행합니다. DINO와 MaskCut은 다시 돌리지
않고 저장된 라벨을 읽어 `reports/pseudo_preview.jpg`를 만듭니다. 외부 이미지 디렉터리나
별도 설정을 사용했다면 `./solo preview /실제/경로 --config configs/custom.toml`처럼
`generate`와 동일하게 지정합니다.

```bash
./solo preview
```

서버에서는 source code를 편집할 필요 없이 다음과 같이 결과만 공유합니다.

```bash
git add reports
git commit -m "exp: A5000 ImageNet pseudo benchmark"
git push
```

후속 분석 시 `summary.md`와 `benchmark.json`을 먼저 확인합니다. 현재 subset의 실제
이미지 수·예상 시간과 ImageNet 전체 1,281,167장에 대한 외삽 시간을 모두 기록합니다.

## 프로젝트 격리

환경 `.venv`, uv cache `.uv-cache`, 관리 Python `.uv-python`, Torch/HF cache `.cache`,
Ultralytics/XDG config `.config`, weight `weights`, temporary files `work/tmp`, pseudo labels
`pseudo`가 모두 repository 아래입니다. wrapper는 XDG data/state, uv tools, matplotlib와
Python bytecode cache, NVIDIA CUDA JIT cache, PyTorch/Triton compile cache도 redirect합니다.
활성화된 외부 Python/conda 환경 변수는 wrapper 안에서 해제합니다.
`OMP/MKL/OPENBLAS_NUM_THREADS=1`이 기본입니다.
global package 설치, global model cache, conda, sudo는 사용하지 않습니다. benchmark와
generate에 전달한 외부 ImageNet 경로는 읽기만 합니다. 기존 uv executable은 관리하지
않습니다. ImageNet 다운로드는 Python
표준 라이브러리만 사용하며 별도 downloader 설치나 전역 인증 설정이 필요하지 않습니다.

## 검증 및 다음 milestone

```bash
./solo test -q
./solo check src tests
```

테스트는 spectral discovery, key-feature equivalence, normalized bbox, filter, single-class row,
atomic write interruption, 손상/빈 receipt, parallel batching, resume, OOM, worker scaling과
안정성 선택을 확인합니다. 실제 checkpoint 테스트는 이미 다운로드된 weight가 있을 때 실행하며
테스트만으로 weight를 다운로드하지 않습니다.

root `./solo` 파일과 Python package 디렉터리 이름 충돌을 피하기 위해 module은 `src/solo/`에
둡니다. DINO는 `DenseBackbone` 계약, CPU discovery는 `discover_masks`, 저장은 `LabelStore`로
분리되어 있습니다.

`./solo train`과 `./solo infer --support ... --query ...`는 다음 milestone을 위한 예약 명령입니다.
Milestone 1의 실제 서버 benchmark와 품질이 확인된 뒤 YOLO11n (640, single-class object)를
학습하고, query당 한 번의 dense DINO forward + ROI pooling + prototype cosine matching을
추가합니다. 현재 명령은 미구현임을 명시하고 종료하며 가짜 학습/추론 결과를 만들지 않습니다.

공식 소스와 라이선스는 [THIRD_PARTY.md](THIRD_PARTY.md)에 기록합니다.
