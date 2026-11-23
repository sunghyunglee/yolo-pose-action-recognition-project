# yolo-pose-action-recognition-project — YOLO Pose 기반 동작인식

<p align="right"><b>한국어</b> | <a href="README.en.md">English</a></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Ultralytics-YOLO11_Pose_8.4-111F68?logo=ultralytics&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-2.15-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-5.0-5C3EE8?logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-3.1-150458?logo=pandas&logoColor=white">
  <br>
  <img src="https://img.shields.io/badge/uv-Package_Manager-DE5FE9?logo=uv&logoColor=white">
  <img src="https://img.shields.io/badge/Ubuntu-WSL2-E95420?logo=ubuntu&logoColor=white">
  <img src="https://img.shields.io/badge/VS_Code-Editor-007ACC?logo=visualstudiocode&logoColor=white">
  <img src="https://img.shields.io/badge/IP_Webcam-Phone_Camera-3DDC84?logo=android&logoColor=white">
</p>

## 프로젝트 소개

### Architecture

사전학습 **YOLO Pose(`yolo11n-pose.pt`)** 로 사람 관절 17개를 찾고, 그 좌표를 **38-D Frame Feature → 30 Step Sequence** 로 바꿔 **Random Forest 기준선**으로 동작을 분류하고, 같은 모델을 **저장 영상 · IP Webcam 실시간 입력**에 연결한 뒤, **ByteTrack 으로 여러 사람을 Track ID 별로 따로** 판단합니다. 데이터는 **사람 단위(S01 Train · S02 Validation · S03 Test)** 로 나누고, 조건 비교는 Validation 으로만 한 뒤 Test 는 **한 번만** 확인합니다.

![YOLO Pose 동작인식 Architecture](docs/images/architecture.png)

*위 그림은 학습 Pipeline(1 ~ 6일차)입니다. 7일차 실시간 Pipeline 은 [실시간 Pipeline](#실시간-pipeline-7일차), 8일차 다중 인물은 [다중 인물 Tracking](#다중-인물-tracking-8일차) 에 있습니다.*

<br>

### 개요

교과 10 · 자세추정(Pose Estimation) 실습 프로젝트입니다. Pose 모델을 새로 학습하지 않고, **사전학습 모델이 찾은 관절을 동작인식용 데이터로 바꾸는 과정**을 1일차부터 단계별로 구현했습니다.

```text
Pose → Keypoint → Feature → Sequence → Dataset → Action Model → (Realtime → Tracking → Augmentation → Failure Analysis)
```

목표는 높은 점수가 아니라 **결과가 틀렸을 때 어느 단계에서 처음 틀렸는지 따라갈 수 있는 Pipeline** 입니다. 단계마다 품질을 확인하고, 같은 사람의 데이터가 Train · Test 에 섞이지 않게 나누고, Test 를 본 뒤에는 기준을 고치지 않았습니다.

> 현재 **1 ~ 8일차**까지 반영했습니다. 9 ~ 10일차(보강 · 실패분석)는 진행하면서 추가합니다.

<br>

### 작업 결과

| 단계 | 방법 | 핵심 결과 |
|---|---|---|
| 1일차 · Pose 입력 | 이미지 → MP4 → IP Webcam 순서로 같은 Pose 코드 검증 | 입력 Pipeline `00` ~ `05` · 폰 카메라 Frame 캡처 |
| 2일차 · Keypoint 품질 | 관절별 Confidence → VALID ≥ 0.50 · WARNING · LOW < 0.30 | 품질 비교 CSV · Skeleton 품질 그림 · **Person BBox Confidence ≠ Keypoint Confidence** |
| 3일차 · Frame Feature | Hip Center 위치 + Shoulder Width 크기 정규화 · 팔꿈치 · 무릎 각도 | **38-D = 좌표 34 + 각도 4** · 규칙 판정(17번) 팔 들기 **True** (최소 Confidence 0.971) · 앉은 자세도 **True** → 규칙 하나로는 구분 못 함 |
| 4일차 · Action Sequence | 6개 Action Clip 의 38-D 를 시간순 연결 | Clip 별 Sequence CSV · Keyframe · 단계별 점검 `19` ~ `24` |
| 5일차 · Dataset · Split | 3명 × 6동작 = **MP4 18개** → Keypoint CSV · 자동 QA + Human QA · 사람 단위 Split | AUTO_PASS **13** · CHECK **5** → 사람 확인 PASS · 승인 Clip **18** · Train / Val / Test **6 / 6 / 6** · 누수 검사 OK |
| 6일차 · RF 기준선 | 30 Step × 38-D = **1,140 Feature** · Random Forest (Tree 300 · seed 42) | Validation(S02) 한 장면 38 Feature **0.6667 (4/6)** vs 30 Step **0.5000 (3/6)** · Macro F1 0.5556 vs 0.4444 |
| 7일차 · 실시간 연결 | 최근 **5초** Sliding Window → 6일차와 같은 30 Step 으로 Resampling → 같은 Random Forest · Confidence **0.55** 미만은 UNKNOWN · 최근 **5개 중 3표** 안정화 | 저장 MP4 → IP Webcam 순서로 같은 Runtime 실행 · Mini Challenge 로 Runtime 변수 **하나만** 바꿔 같은 MP4 재실행 비교 |
| 8일차 · 다중 인물 Tracking | `model.track(persist=True)` + **ByteTrack** 으로 Frame 마다 사람에게 Track ID 를 붙여 이어 감 · Track ID 별로 Feature Buffer · 안정화를 **따로** 둠 · **2초** 동안 안 보이면 그 사람 상태 삭제 | SBU 영상 3개(악수 · 포옹 · 발차기)로 기본 · 가림 · 빠른 상호작용 Tracking 비교 · 같은 Random Forest 를 사람마다 따로 적용 · IP Webcam 다중 인물 Runtime Log |

6일차 숫자는 Validation **6 Clip** 결과입니다. 7 · 8일차 Runtime 값은 정식 Test Accuracy 가 아니라 실시간 동작 확인용이라 성능 숫자로 쓰지 않습니다. 이번 데이터에서는 30 Step Sequence 가 한 장면보다 1개 더 틀렸지만, Clip 수가 적어 일반화하지 않습니다(강사님 자료는 둘 다 0.6667 / 0.5556).

**AI 결과를 그대로 믿지 않은 이유**

- **CSV 는 예측이지 정답이 아님** — YOLO Pose 가 만든 좌표 · Confidence 는 예측 결과입니다. Confidence 0.99 를 "정확도 99%" 로 읽지 않고, 원본 이미지 · 영상에 다시 그려서 확인했습니다.
- **규칙은 쉽게 무너짐** — 3일차 "손목이 어깨보다 위" 규칙은 팔 들기를 맞혔지만 앉은 자세도 True 로 판정했습니다. 그래서 한 장면 규칙이 아니라 **여러 Frame 을 이은 Sequence** 와 학습 모델로 넘어갔습니다.
- **정답은 기준표에서만** — 동작 이름(`action_label`)은 YOLO 가 예측하지 않고 `action_classes.csv` 정답표에서 붙입니다. 학습 때는 **y 로만** 쓰고 입력(X)에는 넣지 않았습니다.

<br>

### 지킨 원칙

| 원칙 | 내용 |
|---|---|
| 같은 처리 기준 | 학습과 추론에 같은 정규화 · Feature · Sequence 길이 사용 (`src/feature_utils.py` · `src/sequence_utils.py`) |
| 단계마다 품질 검사 | 입력 → Keypoint Confidence → 자동 QA → Human QA(PASS · RESHOOT · EXCLUDE) |
| 사람 단위 Split | S01 Train · S02 Validation · S03 Test · Sequence 만들기 전에 배정(`configs/day05_subject_split.json`) |
| Validation 으로만 선택 | 38 · 39번 비교는 Validation 으로 · Test(40번)는 고른 뒤 1회 |
| 원본과 Feature 분리 | 5일차에 원본 Keypoint 저장 → 6일차에 Feature 다시 계산 · Feature 를 바꿔도 Pose 를 다시 돌리지 않음 |
| 설정은 파일로 | 모델 · Confidence · Sequence 길이 · RF · Runtime 설정은 `configs/*.json` 에서 읽고 코드에 하드코딩하지 않음 |
| 사람마다 따로 기억 | 여러 사람의 Frame 을 한 Buffer 에 섞지 않고 Track ID 별 Buffer · 안정화 상태를 따로 관리 (`src/multi_person_utils.py`) |
| 지연은 나눠서 본다 | 느리면 모델부터 의심하지 않고 카메라 · 네트워크 / AI Processing / Sliding Window · 안정화 대기 중 어디인지 구분 |

<br>

### 실시간 Pipeline (7일차)

6일차에 학습한 **같은 Random Forest** 를 끝없이 들어오는 Frame 에 연결했습니다. 모델이 바뀐 것이 아니라, **실시간 Stream 에서 모델 입력을 만드는 방식**이 추가된 것입니다.

```text
Saved MP4 / IP Webcam
        ↓
YOLO Pose → 38-D Frame Feature (학습 때와 같은 계산)
        ↓
Timed Sliding Window (최근 5초 · 80% 이상 채워지면 예측)
        ↓
6일차 Sequence Step(30)으로 Resampling → 1,140 Feature
        ↓
Random Forest → Raw Prediction
        ↓
Confidence Threshold 0.55 → Action / UNKNOWN
        ↓
Prediction Stabilization (최근 5개 중 3표) → Stable Action
```

| 배운 점 | 내용 |
|---|---|
| Window Seconds ≠ Sequence Steps | Window 는 "최근 몇 초를 볼까"(5초), Sequence Step 은 "모델에 몇 칸으로 넣을까"(30) · 5초 안의 Frame 을 30 Step 으로 다시 맞춤 |
| 처음 몇 초는 WARMING_UP | 5초 × 0.8 = 약 4초가 쌓여야 첫 예측 · 오류가 아님 |
| UNKNOWN 은 보류 | Confidence 가 낮으면 억지로 고르지 않고 UNKNOWN 으로 남김 |
| 안정화의 대가 | Label 흔들림은 줄지만 동작이 바뀐 뒤 반응이 늦어짐 |
| Runtime Match ≠ Test Accuracy | 실시간 확인용 값이라 6일차 Test 와 같은 성능 지표로 보지 않음 |
| 지연 구분 | 영상 자체가 늦음 = 카메라 · 네트워크 / 첫 결과가 늦음 = Sliding Window / Label 이 늦게 바뀜 = 안정화 |

**Mini Challenge** — Action 하나를 골라 새 MP4 를 직접 찍고, 기본 Runtime → **변수 하나만** 바꿔 같은 MP4 재실행 → Baseline vs Experiment 비교 → 같은 Action 을 IP Webcam 에서 수행해 Runtime Log 를 분석했습니다. 이 영상은 Train 데이터가 아니라 **학습된 모델을 확인하는 Runtime 입력**입니다.

<br>

### 다중 인물 Tracking (8일차)

7일차 Runtime 은 화면에 **한 사람**만 있다고 가정했습니다. 두 사람이 있으면 두 사람의 관절이 한 Buffer 에 섞여 Sequence 가 엉망이 됩니다. 8일차는 모델은 그대로 두고, **사람을 구분해서 사람마다 따로 Sequence 를 쌓는 부분**을 추가했습니다.

```text
Saved MP4 / IP Webcam
        ↓
YOLO Pose + ByteTrack (model.track · persist=True) → 사람마다 Track ID
        ↓
Track ID 별 38-D Frame Feature (학습 때와 같은 계산)
        ↓
Track ID 별 Sliding Window (5초) · 안정화 (5개 중 3표)   ← 7일차 설정 그대로
        ↓
Track ID 별 Random Forest → Action / UNKNOWN
        ↓
2초 동안 안 보인 Track ID 는 상태 삭제 (lost_timeout_seconds)
```

| 코드 | 하는 일 | 결과 |
|---|---|---|
| `46` | 8일차 입력 점검 (SBU 영상 3개 · 6일차 모델 · 7일차 설정) | 입력 OK 여부 |
| `47` | SBU 영상에 YOLO Pose + ByteTrack | `reports/day08/tracking/*_tracked.mp4` · `reports/day08/logs/*_tracking.csv` |
| `48` | Tracking Log 요약 | 고유 Track ID 수 · 추적된 Frame 비율 · 두 사람이 다 잡힌 Frame 비율 · 한 Frame 최대 ID 수 · 남는 ID 수 |
| `49` | IP Webcam 다중 인물 실시간 동작인식 (`Q` 종료 · `R` 모든 사람 상태 초기화) | `reports/day08/ip_webcam/ip_multi_<시각>.csv` |
| `50` | 49번 Log 를 Track ID 별로 요약 | 처음 · 마지막 등장 시각 · UNKNOWN 비율 · 가장 많은 Stable Label · 평균 Confidence · 처리 FPS |
| `51` | SBU 시나리오 3개 Tracking 결과 비교 | `reports/day08/experiments/tracking_scenario_summary.csv` |

| SBU 영상 | 시나리오 | 확인할 것 |
|---|---|---|
| `01_sbu_shaking_hands.mp4` | BASIC_TRACKING | 두 사람이 떨어져 있을 때 ID 가 끝까지 유지되는가 |
| `02_sbu_hugging.mp4` | OCCLUSION_TRACKING | 몸이 겹칠 때 ID 가 끊기거나 새 ID 가 생기는가 |
| `03_sbu_kicking_challenge.mp4` | FAST_INTERACTION_TRACKING | 빠른 동작에서 ID 가 바뀌는가 |

| 배운 점 | 내용 |
|---|---|
| Detection ≠ Tracking | Detection 은 Frame 마다 "사람이 어디 있나", Tracking 은 "앞 Frame 의 그 사람이 이 사람인가"까지 이어 줌 |
| Buffer 는 사람 수만큼 | Track ID 별로 Buffer · 안정화를 두어야 두 사람의 동작이 섞이지 않음 |
| ID 가 바뀌면 처음부터 | 가림 · 빠른 동작으로 새 ID 가 생기면 그 사람의 Buffer 가 비어 다시 WARMING_UP |
| 사라진 사람 정리 | `lost_timeout_seconds` 2초가 지나면 상태를 지워 오래된 Buffer 가 남지 않게 함 |

<br>

### 학습 파이프라인

4 ~ 6일차 데이터가 어떤 파일 · 코드를 거쳐 모델과 평가까지 가는지 **직접 따라가며 그린 흐름도** 입니다. 정답(`action_label`)이 정답표에서 평가까지 어떻게 이어지는지 함께 표시했습니다.

<p align="center"><img src="docs/images/shl_pose_action_pipeline.png" width="560" alt="4~6일차 Action 학습 파이프라인"></p>

*설정 파일 · 원본 영상 · Python 처리 · 자동 생성 CSV · 모델 · 평가를 색으로 구분했습니다.* 자세한 설명 → [파일 추적표 · 설명](docs/shl_pose_action_pipeline_and_explanation.md)

<br>

### 바로가기

| 폴더 | 무엇인가 | 핵심 |
|---|---|---|
| [**scripts**](scripts) | 일차별 실습 코드 (`00` ~ `51`) | 번호 순서대로 프로젝트 루트에서 실행 |
| [**src**](src) | 공통 함수 | `common.py` 설정 · 경로 · `pose_utils.py` COCO 17 · `feature_utils.py` 38-D · `sequence_utils.py` Sequence · `realtime_utils.py` Sliding Window · 안정화 · FPS · `multi_person_utils.py` Track ID 별 상태 관리 |
| [**configs**](configs) | 설정 파일 | `project.json` Pose 기준 · `day05_subject_split.json` 사람 배정 · `day06_baseline.json` RF 설정 · `day07_realtime.json` Runtime 설정 · `day08_multi_person.json` Tracker · 사라진 사람 정리 시간 |
| [**docs**](docs) | 문서 · 그림 | Architecture · 4 ~ 6일차 파이프라인 흐름도 · 파일 추적표 |

| 일⁠차 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 주제 | Pose 입력 | Keypoint 품질 | 38-D Feature | Action Sequence | QA · Split | RF 기준선 | 실시간 | 다중 인물 |
| 코드 | `00~05` | `06~10` | `11~17` | `19~24` | `25~32` | `33~40` | `41~45` | `46~51` |

<br><br>

## 빠른 시작

준비물: Windows 11 PC · WSL2 Ubuntu · [uv](https://docs.astral.sh/uv/) · GPU 가 없으면 `device: "auto"` 로 CPU 사용

**1단계 · 받기** — 프로젝트는 WSL 안(`~/ai_vision/`)에 둡니다

```bash
mkdir -p ~/ai_vision && cd ~/ai_vision
git clone https://github.com/sunghyunglee/yolo-pose-action-recognition-project.git subject10_pose_action
cd subject10_pose_action
```

**2단계 · 설치**

```bash
uv sync
source .venv/bin/activate
python scripts/00_check_environment.py
```

**3단계 · 데이터 넣기** — 영상 · 이미지는 Git 에 없습니다. 수업 데이터를 일차별 폴더에 넣습니다

```text
data/day01/  data/day02/  data/day03/  data/day04/  data/day05/videos/  (S01_A001_T01.mp4 ~ S03_A006_T01.mp4)
data/day08/videos/  (01_sbu_shaking_hands.mp4 · 02_sbu_hugging.mp4 · 03_sbu_kicking_challenge.mp4)
```

**4단계 · 실행** — 터미널 위치는 프로젝트 루트, 코드는 번호 순서대로

```bash
python scripts/26_extract_keypoint_dataset.py
python scripts/30_build_subject_split.py
python scripts/31_check_split_leakage.py
python scripts/34_build_sequence_dataset.py
python scripts/35_train_rf_baseline.py
python scripts/36_evaluate_validation.py
python scripts/38_compare_static_vs_sequence.py
python scripts/40_evaluate_test.py
```

실시간 (7일차) — 6일차 모델이 있어야 합니다

```bash
python scripts/41_check_day07_inputs.py
python scripts/42_realtime_action_video.py --source data/day05/videos/S02_A006_T01.mp4
python scripts/44_ip_webcam_action_realtime.py --url http://192.168.0.10:8080/video
```

다중 인물 (8일차) — 6일차 모델 · 7일차 설정이 있어야 합니다

```bash
python scripts/46_check_day08_inputs.py
python scripts/47_track_sbu_video.py --source data/day08/videos/01_sbu_shaking_hands.mp4
python scripts/48_analyze_tracking_log.py --log reports/day08/logs/01_sbu_shaking_hands_tracking.csv
python scripts/51_compare_tracking_scenarios.py
python scripts/49_ip_webcam_multi_person_action.py --url http://192.168.0.10:8080/video
python scripts/50_analyze_multi_person_log.py
```

51번은 SBU 영상 3개 모두 47 → 48번을 돌린 뒤 실행합니다. `--url` 은 폰 IP Webcam 앱에 표시되는 주소로 바꿉니다. IP Webcam 화면에서 숫자키 `1` ~ `6` 으로 지금 수행할 목표 Action 을 고릅니다.

5일차 Human QA(28번)는 실행 전에 `keypoint_clip_review.csv` 의 `human_decision` 칸에 PASS · RESHOOT · EXCLUDE 를 직접 적습니다. 결과는 `reports/dayXX/`, 모델은 `artifacts/day06/rf_action_baseline.joblib` 에 저장됩니다.

<br><br>

## 주의사항

- **원본 영상 · 이미지 · `data/` · `reports/` · `artifacts/` · `*.pt` · 생성 CSV 는 Git 에 올리지 않습니다** — 코드 · 설정 · 문서만 올립니다.
- 촬영한 사람의 얼굴이 보이는 이미지 · 영상은 공개하지 않습니다.
- Confidence 기준(0.50 / 0.30)은 교육용 기준이며 YOLO 공식 기준이 아닙니다. 실제 프로젝트에서는 검증 데이터로 다시 정합니다.
- `scripts/18_video_frame_features.py` 는 수업자료 코드에 오류가 있어 확인 후 추가할 예정입니다.
