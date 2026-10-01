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

## 여러 객체가 모인 장면: crowded 설정

**현재 품질 확인:** [COCO 20장 추가 비교](reports/20261001T122548Z-instance-ablation-b31dde/summary.md)에서
보수적 설정은 박스 수를 75개에서 38개로 줄였지만, 붙어 있는 아이·얼룩말은 여전히 합쳐졌습니다.
추가 분할은 머리·몸 조각을 늘렸고, 보수적 설정에도 객체 누락이 있어 두 후보 모두 기본값으로
적용하지 않았습니다. `--crowded`는 개별 객체 분리의 해결책으로 검증된 설정이 아닙니다.
현재는 분리 방법을 더 검증한 뒤 대규모 라벨 재생성·학습 여부를 결정하는 것이 좋습니다.

기존 baseline의 `384px / 최대 2개` 설정 외에 `--crowded`를 추가했습니다.
**클래스는 계속 `0 = object`이며, 각 객체에 별도 박스를 만드는 것이 목표**입니다.

- 같은 frozen DINO ViT-S/16을 512px에서 실행해 24×24 대신 **32×32 패치**를 사용합니다.
- 최대 12개 마스크를 찾고, 전경 안에서 **떨어진 연결 영역을 각각 추출**합니다.
- 작은 객체를 덜 버리도록 최소 mask/box 면적을 낮춥니다.
- 특징 차이로 연결된 마스크를 추가 분할하는 옵션도 있지만 기본은 OFF입니다.
  작은 공개 예제 비교에서 몸 일부가 별도 박스로 잘리는 사례가 있어 기본 적용하지 않았습니다.
  완전히 붙거나 가려진 사람을 항상 개별 분리하는 방법은 아니며, 실제 군중 데이터의 검증이 필요합니다.

먼저 **전체 데이터를 다시 만들기 전에** 같은 이미지의 전후 박스를 비교하세요:

```bash
git pull --ff-only
./solo compare
```

기본 COCO val2017에서 deterministic 16장을 선택합니다. 문제가 나온 이미지들을
`data/check/` 같은 폴더에 모으면 그 폴더로 비교할 수 있습니다:

```bash
./solo compare data/check --limit 16
```

`reports/<run-id>-compare-*/comparison.jpg`의 **왼쪽은 기존 baseline, 오른쪽은 crowded**입니다.
JSON/CSV/summary에는 각 설정, 박스 수·면적 분포, 처리 시간, 하드웨어를 기록합니다.
이 검사는 기존 pseudo-label이나 학습 가중치를 변경하지 않으며, 추가 다운로드는 기존 DINO
가중치가 없는 경우에만 필요합니다. 작은 비교 실행의 속도는 10k benchmark를 대체하지 않습니다.
박스가 많다는 것만으로 성공은 아닙니다. **사람별 박스, 몸 일부의 중복 검출, 배경 박스, 누락**을
함께 확인하세요. `--limit`은 1~64입니다.

분리 품질을 확인한 뒤 새 설정으로 benchmark → 전체 라벨 생성 → 새 YOLO 학습을 실행합니다:

```bash
./solo benchmark --crowded && ./solo generate --crowded && ./solo train --crowded
./solo preview --crowded
./solo predict --crowded
```

이미 다운로드한 COCO 이미지를 그대로 사용합니다. `--crowded`는
[configs/crowded.toml](configs/crowded.toml)의 별칭이고, 생략하면 기존 baseline을 사용합니다.
새 설정은 처리 비용이 더 크므로 기존 benchmark의 batch/worker 설정을 재사용하지 않습니다.
라벨이 달라지므로 **기존 YOLO를 이어 학습하지 않고 별도 무작위 초기화 학습**을 시작합니다.

```text
pseudo/coco-crowded/                             # 새 학습 라벨
outputs/coco-yolo11n-crowded/weights/best.pt        # 새 YOLO 가중치
reports/coco-yolo11n-crowded-training/             # 새 학습 리포트
reports/pseudo_preview_coco-yolo11n-crowded.jpg     # 새 라벨 프리뷰
```

기존 `pseudo/coco/`, `outputs/coco-yolo11n/` 및 학습 재개 state는 보존합니다.
새 학습 중단 후에는 `./solo train --crowded`로 이어갑니다.
세부 옵션은 `[maskcut]`의 `max_objects`, `min_mask_area`, `all_components`입니다.
실험용 재귀 분할은 `split_depth = 1` 이상으로 켜며 `split_max_ncut`과
`split_min_cosine_distance`로 분할 허용 기준을 조절합니다. 영역이 충분히 크고,
분할 비용이 낮고, 특징 차이가 있으며, 각 자식이 하나의 연결 영역일 때만 분할하지만
**신체 부위 분할을 완전히 막지는 못합니다.** 변경 후에는 compare와 benchmark를 다시 실행하세요.
실험마다 `[train].run_name`과 `[pipeline].output_dir`를 다르게 지정하면 결과를 보존할 수 있습니다.

이 개선은 few-shot 매칭 전에 필요한 객체 후보를 만드는 단계입니다. 개인별 신원 구분이나
support 예시와의 최종 매칭 성능은 아직 구현·검증되지 않았습니다.

## DINO key 특징에서 박스까지 5장 시각화

```bash
./solo key-preview --crowded
# 원하는 이미지 폴더:
./solo key-preview data/check --crowded --limit 5
```

기본 COCO val2017에서 5장을 선택하여 `reports/<run-id>-key-preview-*/key_pipeline.jpg`에
**원본 → key 특징 PCA RGB → MaskCut 마스크 → 필터링된 박스**를 저장합니다.
이미지별 큰 그림은 `sample-01.jpg` 등이며, 실행한 모델·checkpoint SHA·설정·입력 SHA·박스
좌표는 `features.json`에 기록합니다. 저장된 학습 라벨이나 YOLO 가중치는 바꾸지 않습니다.

현재 backbone은 **DINO v1 ViT-S/16**입니다. 마지막 attention block의 key에서 CLS를 제외한
384채널을 사용합니다. RGB 그림은 이미지마다 PCA 3성분으로 압축한 설명용 그림이며,
**그 색을 이용해 박스를 만드는 것은 아닙니다.** MaskCut은 원래 384채널을 사용합니다.
색은 클래스·신원·confidence가 아니며 이미지 간 같은 색이 같은 객체를 뜻하지 않습니다.
마스크와 최종 박스의 색은 해당 이미지 안에서만 대응하고, 필터링된 마스크는 회색입니다.

모델 선택: 지금의 key/MaskCut 기준선은 v1을 유지합니다. [DINOv2](https://github.com/facebookresearch/dinov2)는
대규모 학습으로 다양한 도메인에서의 범용 특징을 강화했고,
[DINOv3](https://ai.meta.com/research/publications/dinov3/)는 Gram anchoring 등으로 dense feature
품질을 강화했습니다. 향후 few-shot ROI 특징 비교에서는 v3를 우선 비교할 후보로 봅니다.
하지만 논문의 일반적인 dense 성능이 이 프로젝트의 **개별 객체 박스 품질**을 보증하지는
않습니다. 아래 `compare-backbones`로 세 버전을 같은 이미지에서 비교할 수 있지만,
v1이 최종 개인화 목표에 충분하다고 확정한 것은 아닙니다. 특히 `crowded`는 배경·신체 일부를 추가 박스로 잡을 수
있으므로 큰 학습을 시작하기 전에 실패 장면도 확인해야 합니다.

## 일상 물체 few-shot 샘플: 개의 외형 구분

```bash
git pull --ff-only
./solo pet-demo
```

YOLO가 기본적으로 다루는 **개**를 대상으로 비글 / 퍼그 / 사모예드를 비교합니다.
Oxford-IIIT Pet의 train 예시 각 1장과 test 사진 각 3장을 고정하여 다운로드합니다.
필요한 사진 12장만 받으며 데이터셋 전체나 별도 주석은 받지 않습니다.
처음 실행할 때 필요한 DINO v1/v2/v3와 공식 COCO YOLO11n 가중치도 레포 내부에 준비합니다.

YOLO는 모든 기본 클래스를 예측하고, 그중 `dog` 후보에서만 DINO 특징을 pooling해
세 예시와 비교합니다. DINO는 이미지 전체에 버전별 한 번만 실행합니다. 비교 결과와
대표 그림은 `reports/<run-id>-pet-demo-*/`에 생성됩니다.

이 실험은 **품종의 외형을 구분**하며 같은 강아지 개체를 재식별하는 검증은 아닙니다.
이미지별 가장 confidence가 높은 dog 후보의 품종을 비교하며, 정답 박스를 사용한
IoU 검출 평가는 아닙니다. 알려지지 않은 품종을 거부하는 기준도 검증하지 않았습니다.
테스트 사진 3장을 나란히 붙인 추가 입력은 `collage.jpg`로 명시하며, 자연 사진 9장
수치에서 제외합니다. 일반 COCO YOLO 대조군이며 SOLO 전용 detector 학습 결과가 아닙니다.

## 농작물 상태 few-shot 샘플 실험

[실제 실행 결과와 3상태 비교 그림](reports/20261001T130459Z-crop-demo-f0d92b/review.md):
9장·참조 잎 43개에서 YOLO+DINO v3는 위치와 상태를 함께 9개 맞혔습니다.
정답 위치를 알려준 별도 상태 분류는 v1 60.5%, v2 76.7%, v3 79.1%지만,
v3도 건강한 잎은 3/12만 맞혀 **현장 적용에 충분하지 않은 결과**입니다.

```bash
git pull --ff-only
./solo crop-demo
```

고정된 PlantDoc 토마토 자료에서 **건강 / Early blight / Yellow virus** 예시를 각 1장,
테스트를 각 3장 사용합니다. 실외·화분·재배 배경이 있는 이미지를 실행 결과를 보기 전에
골랐으며, 같은 농장의 넓은 밭 사진을 검증한 데이터는 아닙니다. 자료의 상태 이름은 데이터셋
주석이며 병해의 확진으로 해석하면 안 됩니다. 출처·고정 revision·SHA256·선정 사유는
`configs/crop-demo.json`에 있습니다. 지원 예시는 cropped train, 테스트는 원본 TEST split입니다.

명령 하나가 필요한 샘플·가중치를 레포 내부에 준비하고 다음을 비교합니다:

- 후보 생성: 일반 COCO 사전학습 YOLO11n / DINO v1 MaskCut / SAM 2.1 Tiny 자동 마스크.
- 상태 비교: frozen DINO v1 / v2 / v3의 최종 patch 특징으로 같은 후보를 ROI pooling하여
  support prototype과 cosine 비교. 각 이미지에서 backbone당 forward는 한 번뿐입니다.
- 위치 평가: 공개 PlantDoc 박스는 **평가에만** 사용하여 IoU ≥ 0.5인 잎을 셉니다.
- 별도 진단: 정답 잎 위치를 알고 있다고 가정한 ROI 상태 분류도 표시합니다.
  이 수치는 실제 검출 성능이 아닙니다.

**현재 컴퓨터에 학습된 SOLO 전용 YOLO 가중치가 없어서 COCO YOLO를 비교 대조군으로 씁니다.**
클래스 이름을 무시해도 COCO의 잎 검출 능력이 생기는 것은 아닙니다. 전용 `0=object` YOLO를
학습하거나 검증한 결과로 해석하지 마세요. 모델 학습·feature 주입은 하지 않습니다.
이번 실험은 세 상태 중 하나를 고르는 분류이며 `unknown`/배경 거부 기준은 검증하지 않았습니다.

`reports/<run-id>-crop-demo-*/support.jpg`, 상태별 비교 이미지, `summary.md`, `results.json`을
만듭니다. 박스 색은 초록=healthy, 빨강=early blight, 노랑=yellow virus이고 숫자는 cosine
유사도입니다(정확도/확률 아님). `v2-`, `v3-` 그림은 후보 박스는 같고 특징 비교 모델만 다릅니다.
기존 학습 라벨·가중치는 유지합니다. 모든 다운로드는 레포 내부이며, CPU 실험은 처리량
benchmark를 대체하지 않습니다.

## SAM 2와 DINO/MaskCut 비교

실측 결과: [같은 COCO 20장 비교](reports/20261001T123557Z-sam-comparison-1953d6/review.md),
[핵심 5장 32×32 탐색 재확인](reports/20261001T123801Z-sam-comparison-4c2d5f/review.md).
이번 SAM 2.1 Tiny 자동 마스크 설정은 사과·일부 사람의 경계를 더 잘 나누지만,
두 아이를 온전한 사람 박스 2개로 정리하지 못했습니다. 20장 박스는 DINO 75개,
SAM 180개였으며, 부위·배경 박스도 포함됩니다. SAM을 검증된 대체 방법으로 기본 적용하지는
않았습니다. 더 큰 모델이나 다른 필터·프롬프트 방식의 결과까지 대표하는 비교는 아닙니다.

```bash
git pull --ff-only
./solo compare-sam
# 같은 개인 이미지 최대 20장으로 비교:
./solo compare-sam data/check --limit 20
```

같은 이미지에서 **DINO v1 ViT-S/16 + MaskCut**과 공식 **SAM 2.1 Hiera-Tiny**의
자동 마스크를 비교합니다. 사람이 클릭하거나 정답 박스를 제공하지 않고, SAM의 박스 개수도
고정하지 않습니다. 기본 입력은 `data/coco/images/val2017`이며, SAM 패키지는 별도 uv 그룹,
가중치 약 156MB는 `weights/`에만 설치·저장합니다. CUDA extension을 빌드하지 않으므로
호스트에 CUDA toolkit을 추가 설치할 필요가 없습니다.

`reports/<run-id>-sam-comparison-*/`에 이미지별 그림과 5장씩 묶은 비교 그림,
좌표·마스크 품질 점수·처리 시간·설정·모델 SHA256을 저장합니다. 그림은 왼쪽부터
**DINO 박스 / SAM 박스 / SAM 마스크**입니다. DINO는 crowded 512px, SAM은 학습된
1024px 입력을 사용하므로 인코더 구조 하나만의 우열을 비교하는 실험은 아닙니다.
두 모델에 같은 최종 면적·종횡비 필터를 적용하며 SAM에는 자체 마스크 품질·중복 필터도 있습니다.

CPU에서도 비교할 수 있도록 기본 탐색은 16×16 자동 점, PyTorch CPU 스레드는 4개입니다.
공식 기본 밀도인 32×32로 확인하려면 `--points-per-side 32`를 추가합니다.
`--cpu-threads 1`로 CPU 사용량을 줄일 수 있습니다. 촘촘한 탐색은 더 느리고 작은 객체
검출에 영향을 줄 수 있습니다. crop 재탐색과 CUDA 연결 영역 후처리는 비활성화합니다.
시간은 워밍업 후 이미지당 한 번 측정한 값으로, 서버의 sustained throughput은 아닙니다.

**ViT는 모델 구조, DINO는 자기지도 학습 방식, SAM은 분할 모델입니다.**
SAM 1은 일반 ViT 이미지 인코더를, SAM 2는 **Hiera라는 계층형 Vision Transformer**를
사용합니다. 따라서 SAM도 Vision Transformer 계열을 사용합니다.
주요 차이는 DINO의 자기지도 특징으로 MaskCut을 수행하는 것과, SAM의 마스크 지도학습으로
얻은 분할 능력을 사용하는 것입니다. COCO 정답 주석을 이번 실행에 쓰지는 않지만,
SAM 사용 시 원래의 엄격한 비지도 사전학습 조건은 달라집니다.
SAM의 마스크 품질 점수도 ‘완전한 객체일 확률’이 아니므로 신체 일부·배경·중첩 박스가
남을 수 있습니다. 비교 명령은 기존 생성 라벨이나 YOLO 가중치를 변경하지 않습니다.

출처: [SAM 2 공식 코드](https://github.com/facebookresearch/sam2),
[Hiera 논문](https://arxiv.org/abs/2306.00989).

## DINO v1 / v2 / v3 간단 비교

```bash
./solo compare-backbones
# 같은 개인 이미지 5장으로 비교:
./solo compare-backbones data/check --limit 5
```

비교 전용 명령이며 production backbone이나 기존 pseudo-label은 바꾸지 않습니다.
필요한 timm 및 가중치도 `.venv`, `.cache/huggingface/` 등 **레포 내부**에만 준비합니다.
v2/v3는 공개된 timm ViT-S 변환 가중치의 고정 revision을 사용하고 SHA256을 기록합니다.
패키지는 별도 `comparison` dependency group과 `uv.lock`으로 고정되어 있습니다.

- 동일 이미지·448px·FP32·CPU 1 thread·batch 1·명시적 attention 연산을 사용합니다.
  다운로드·모델 초기화 후 1회 warmup, 이미지마다 기본 3회 반복합니다.
- v1/v3는 patch 16(28×28), v2는 patch 14(32×32)입니다. 입력 크기는 같지만 token 수가
  달라 v2가 더 많은 연산을 합니다. **서버 GPU throughput 벤치마크가 아닙니다.**
- 한 번의 forward에서 `keys`(마지막 QKV의 K, v3는 RoPE 적용 전)와 `tokens`(최종 LayerNorm
  patch 특징)를 함께 얻습니다. CLS/register token은 제외합니다.
- 동일 MaskCut 설정으로 비교한 뒤, **최종 patch 특징만** 임계값 0.5·0.7·0.9를 모든 모델에
  똑같이 적용합니다. 모델별 특징 분포가 다르므로 기존 tau=0.15의 단순 교체 결과만으로
  모델의 일반적인 우열을 판단하지 않습니다. sweep은 특징을 재사용합니다.

```text
reports/<run-id>-backbones-*/
  summary.md / comparison.json       # 조건, 모델/가중치, 반복 시간, 박스 좌표
  keys_boxes.jpg / tokens_boxes.jpg  # 원본 | v1 | v2 | v3 (기존 tau)
  keys_maps.jpg / tokens_maps.jpg    # PCA 특징 비교
  tokens_tau_0.5.jpg                 # 추가 임계값 비교 (0.7, 0.9도 저장)
  v2_keys_pipeline.jpg              # 원본 → 특징 → 마스크 → 박스
  v3_tokens_pipeline.jpg            # 각 버전/특징에 대해 별도 그림
```

시간은 이미지별 반복의 median을 구한 뒤 이미지들에 대해 평균한 값입니다. forward와
MaskCut/박스/파일쓰기 시간을 따로 기록하며, PCA·그림 저장·다운로드는 제외합니다.
박스 개수는 정확도가 아닙니다. 정답 주석이 없는 이 비교에서는 mAP나 정확도 점수를 만들지
않습니다. 특히 5장으로 고른 threshold는 다른 군중 이미지에서 독립적으로 검증해야 합니다.

## 학습한 YOLO 테스트

학습이 끝나면 아래 명령으로 COCO 검증 이미지의 첫 16장에 **학습한 YOLO가 예측한 박스**를
그립니다. `outputs/coco-yolo11n/weights/best.pt`를 자동으로 사용합니다.

```bash
./solo predict
```

내 사진 한 장 또는 사진 폴더도 지정할 수 있습니다. 원본은 수정하지 않습니다.

```bash
./solo predict /path/to/photo.jpg
./solo predict /path/to/photos
./solo predict /path/to/photos --limit 0
```

폴더는 하위 폴더를 포함해 경로순으로 기본 16장만 처리합니다. `--limit 0`은 전부,
`--limit 100`은 최대 100장을 처리합니다. 박스가 너무 적으면 `--conf 0.1`로 confidence
threshold를 낮춰 확인할 수 있습니다(기본 0.25). 별도 학습 설정을 사용했다면
`--config configs/custom.toml`을 같이 전달합니다.

결과는 실행할 때마다 새 폴더에 저장하고 터미널에 그 위치와 진행률·ETA를 출력합니다:

```text
outputs/predictions/<run-id>/images/       # 박스와 object confidence를 그린 이미지
outputs/predictions/<run-id>/boxes/        # 이미지별 박스 좌표·confidence JSON
outputs/predictions/<run-id>/summary.json  # 사용한 가중치·설정·처리 수
```

`images/`의 JPG를 열면 됩니다. 객체를 찾지 못한 이미지도 박스 없이 저장합니다.
이 명령은 YOLO의 `object` 탐지 결과를 눈으로 확인하는 용도이며 COCO 정답 기준 mAP 평가나
few-shot 분류는 수행하지 않습니다. DINO 학습 라벨 확인은 `./solo preview`, 학습된 YOLO
결과 확인은 `./solo predict`입니다. `./solo test`는 개발용 코드 테스트 명령입니다.

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
