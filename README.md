# SOLO

**COCO 이미지 → frozen DINO ViT-S/16 → MaskCut 비지도 박스 → YOLO11n 사전학습**

COCO의 기존 바운딩 박스와 클래스 주석은 다운로드하거나 사용하지 않습니다.
DINO/MaskCut이 만든 모든 박스를 **`0 = object`**로 저장합니다. YOLO11n은
**무작위 초기화부터** 이 박스로 물체의 위치를 학습합니다. DINO는 이미 사전학습된
ViT-S/16을 frozen 상태로 사용합니다.

최종 목표는 `입력 이미지 → YOLO 객체 후보 + DINO dense 특징 → 예시 이미지와 few-shot 매칭`
입니다. 추후 query 전체에서 DINO를 한 번만 실행하고 각 후보 영역의 특징을 pooling해
support prototype과 비교합니다. 현재 구현 범위는 데이터 준비·박스 생성·YOLO 학습이며,
few-shot `infer`는 다음 단계입니다.

## 서버에서 실행

서버에 기존 `uv` 실행 파일과 NVIDIA driver만 있으면 됩니다. 처음 받는 경우:

```bash
git clone https://github.com/jeehoo0507/YOLO-67.git
cd YOLO-67
./solo download-coco && ./solo benchmark && ./solo generate && ./solo train
```

이미 클론했다면 삭제하지 않고 최신 코드를 받아도 됩니다:

```bash
cd YOLO-67
git pull --ff-only
./solo download-coco && ./solo benchmark && ./solo generate && ./solo train
```

위 한 줄은 **COCO 이미지 다운로드 → 실제 서버 병렬 benchmark/autotune → 전체 DINO 박스
생성 → YOLO 학습** 순서로 실행됩니다. 한 단계가 실패하면 다음 단계는 실행하지 않습니다.
24시간 throughput 목표의 FAIL은 속도 평가이므로 명령 실패로 처리하지 않습니다.
다시 클론하려고 기존 폴더를 지우면 그 안의 이미지·환경·가중치·미공유 결과도 함께 삭제됩니다.

단계를 따로 실행할 수도 있습니다:

```bash
./solo download-coco
./solo benchmark
./solo generate
./solo preview
./solo train
```

`preview`는 저장된 DINO 박스를 원본 이미지 위에 그린 `reports/pseudo_preview.jpg`를
만듭니다. DINO나 YOLO를 다시 실행하지 않습니다. `generate`도 완료 시 대표 grid를 저장합니다.

## 데이터와 학습 설정

- COCO 2017 train **118,287장**, val **5,000장**: 총 **123,287장**.
- [공식 COCO 이미지](https://cocodataset.org/#download)만 사용합니다. 공식 S3 bucket의
  HTTPS 주소에서 받아 `data/coco/images/train2017`, `val2017`에 풉니다.
- 다운로드 약 **20.15 GB**. 완료된 zip은 압축 해제 후 삭제합니다. 압축 해제 중에는
  zip과 이미지가 함께 있으므로 환경·cache까지 고려해 **여유 공간 60 GB 이상**을 권장합니다.
- 다운로드는 `.part`에서 이어받고, 추출은 CRC 검사와 이미지별 atomic rename을 사용합니다.
- 기본 DINO 384px, MaskCut 최대 2개 객체, CRF OFF. 입력 이미지에 객체가 많아도
  첫 baseline은 이미지당 최대 2개 pseudo box만 생성합니다.
- YOLO11n: **무작위 가중치**, `nc=1`, 640px, batch 16, 최대 100 epochs.
  patience 30으로 개선이 없으면 일찍 종료합니다. 설정은
  [configs/baseline.toml](configs/baseline.toml)의 `[train]`에서 바꿉니다.
- 첫 학습 baseline은 FP32입니다. Ultralytics의 AMP 사전 검사가 COCO 사전학습 가중치를
  자동 다운로드하는 경로를 피하도록 AMP를 껐습니다.
- 공식 train/val 이미지 분할을 유지하지만 **양쪽 모두 DINO가 만든 라벨**로 학습·검증합니다.
  검증 점수는 pseudo box와의 일치도이며 COCO 정답 기준 mAP가 아닙니다.

학습 전에 모든 pseudo-label receipt를 검사합니다. 학습용 이미지 링크와 라벨 링크는
`work/coco_yolo_dataset/`에 만들며 원본 이미지를 복사하지 않습니다.
터미널에 epoch, batch 진행률, elapsed, **예상 남은 시간(ETA)**을 10초마다 표시합니다.
ETA는 현재까지 관측한 속도와 최대 epoch 수에 따른 추정치이며 초반에는 변동이 큽니다.

```text
outputs/coco-yolo11n/weights/best.pt    # 학습 결과
outputs/coco-yolo11n/weights/last.pt    # 재시작용
reports/coco-yolo11n-training/          # 작은 학습 요약과 metrics
```

학습 중단 후 `./solo train`을 다시 실행하면 `last.pt`의 완료 epoch부터 이어갑니다.
첫 checkpoint가 저장되기 전에 중단되면 첫 epoch부터 다시 시작합니다. 완료된 학습은
같은 명령으로 다시 학습하지 않습니다. Dataset이나 주요 학습 설정을 바꾸어 새 실험을
시작하려면 기존 `outputs/coco-yolo11n/`, `work/coco-train-state.json`을 먼저 보관하고
해당 경로를 비웁니다.

`./solo`는 `uv run --frozen`으로 저장소 내부 환경을 자동 준비합니다. `.venv` 활성화나
별도 Python 명령은 필요 없습니다. Python 3.11.13, PyTorch 2.7.1 및 dependency는
`uv.lock`으로 고정합니다. Host CUDA toolkit 설치나 NVIDIA driver 변경은 없습니다.
환경만 확인하려면 `./solo doctor`를 실행합니다.

별도 config는 `--config configs/custom.toml`로 지정하고 benchmark/generate/train에
동일하게 적용합니다. 외부 COCO 이미지는 각 명령의 경로 인자로 지정할 수 있으며,
`train2017/`, `val2017/`를 포함하는 이미지 root를 전달합니다.

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
pseudo/coco/
├── labels/<relative-image-stem>.txt
├── receipts/<relative-image-path>.json
└── .generation.lock
```

예: `train2017/000000000009.jpg` → `labels/train2017/000000000009.txt`. 같은 폴더에서 동일 stem의 JPG/PNG가
겹치면 discovery에서 충돌을 명시하고 중단하므로 라벨을 덮어쓰지 않습니다. `./solo train`은
원본 이미지를 복사하지 않는 YOLO dataset adapter를 사용합니다.

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
git commit -m "exp: COCO DINO boxes and YOLO training"
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
generate/train에 전달한 외부 이미지 경로는 읽기만 합니다. 기존 uv executable은 관리하지
않습니다. 이미지 다운로드는 Python
표준 라이브러리만 사용하며 별도 downloader 설치나 전역 인증 설정이 필요하지 않습니다.

## 검증 및 다음 milestone

```bash
./solo test -q
./solo check src tests
```

테스트는 spectral discovery, key-feature equivalence, normalized bbox, filter, single-class row,
atomic write interruption, 손상/빈 receipt, parallel batching, resume, OOM, worker scaling과
안정성 선택, COCO 다운로드/추출 재시작과 학습 데이터 연결을 확인합니다. 실제 checkpoint 테스트는 이미 다운로드된 weight가 있을 때 실행하며
테스트만으로 weight를 다운로드하지 않습니다.

root `./solo` 파일과 Python package 디렉터리 이름 충돌을 피하기 위해 module은 `src/solo/`에
둡니다. DINO는 `DenseBackbone` 계약, CPU discovery는 `discover_masks`, 저장은 `LabelStore`로
분리되어 있습니다.

`./solo train`은 무작위 초기화 YOLO11n (640, single-class object) 사전학습을 실행합니다.
`./solo infer --support ... --query ...`는 다음 milestone을 위한 예약 명령입니다.
이후 query당 한 번의 dense DINO forward + ROI pooling + prototype cosine matching을
추가합니다. 현재 `infer`는 미구현임을 명시하고 종료합니다.

공식 소스와 라이선스는 [THIRD_PARTY.md](THIRD_PARTY.md)에 기록합니다.

## 기존 ImageNet 실험

기존 ImageNet downloader는 `./solo download`로 유지합니다. COCO workflow에서는
`./solo download-coco`를 사용하세요. ImageNet용 설정과 경로는 명시적으로 지정합니다:

```bash
./solo download
./solo benchmark data/imagenet/train --config configs/imagenet.toml
./solo generate data/imagenet/train --config configs/imagenet.toml
```

ImageNet class label은 사용하지 않습니다. 이미 생성한 ImageNet 데이터와 결과는 COCO
명령이 자동 삭제하지 않습니다. `./solo train`은 현재 COCO train/val 구조를 대상으로 합니다.
