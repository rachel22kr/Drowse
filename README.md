# 눈 개폐 분류와 졸음 알람

MRL 눈 사진으로 신경망과 PCA 분류기를 만들고, 사진·웹캠에서 눈 개폐를 판단합니다. 레이블은 **`0=감음(closed)`, `1=뜸(open)`**입니다. 신경망은 무작위 초기화와 교차 엔트로피로 직접 학습하며, 졸음은 별도 시간 조건으로 판단합니다.

## 파일 구성

| 학습 | 추론 |
| --- | --- |
| `train_preprocess.py`: 설정·전처리·분할·캐시·눈 사진 수집 | `inf_preprocess.py`: 웹캠 양안 ROI·4초 프레임 버퍼 |
| `train.py`: 학습·조기 종료·베스트 모델·평가·비교 플롯 | `inf.py`: 모델 구조·신경망/PCA 추론·웹캠 실행 |
| `train_pca.py`: PCA 계산·저장·평가·판정 방향 시각화 | `inf_alarm.py`: 졸음 조건·이력·비차단 경고음 |

신경망 구조는 `inf.py`, 공통 전처리와 경로는 `train_preprocess.py`에서 공유합니다. SmallCNN과 DenseNet 모두 사전학습 가중치 없이 학습합니다.

## 설치와 실행

현재 데이터셋은 `/home/seungmin/projects/openCV/datasets`에 있습니다.

```bash
cd /home/seungmin/projects/openCV
source ../.venv/bin/activate

python train.py                  # 최대 300 epochs, 조기 종료 사용
python inf.py --image datasets/s0001/s0001_00001_0_0_0_0_0_01.png --device cpu
python train.py --evaluate-only
```

새 환경에서는 여섯 파일이 있는 폴더에서 가상환경을 만들고 설치합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision numpy pandas scikit-learn matplotlib opencv-contrib-python mediapipe
```

모든 결과는 기본 **`results/` 바로 아래**에 저장합니다. CLI 상대 경로는 실행 위치와 관계없이 코드 폴더 기준이며, CSV의 이미지 상대 경로는 해당 CSV 폴더 기준입니다.

```bash
python train.py --results-dir results/run2
python inf.py --results-dir results/run2 --image datasets/s0001/s0001_00001_0_0_0_0_0_01.png
python train.py --evaluate-only --results-dir results/run2
```

## 데이터와 전처리

`datasets/`는 MRL Eye 형식 **84,898장·37명**입니다. `annotation.txt`에 따라 파일명 5번째 필드를 눈 상태로 읽습니다. 감음은 41,946장, 뜸은 42,952장입니다.

```text
datasets/
├── annotation.txt
├── s0001/s0001_00001_0_0_0_0_0_01.png
└── s0002/...
```

기본 seed 42는 이미지 기준 학습용 67,918장과 최종 테스트 16,980장으로 **80:20** 분할합니다. 신경망은 학습용 80% 중 10%를 검증에 사용하므로 전체 비율은 약 **학습 72% / 검증 8% / 테스트 20%**입니다. PCA는 원래 학습용 80%로 계산하고 두 방법을 동일 테스트 20%로 비교합니다.

`--split-by subject`는 사람을 80:20으로 나눠 같은 사람의 사진이 최종 테스트에 섞이지 않게 합니다. 사람별 사진 수가 달라 이미지 비율은 달라질 수 있습니다. 신경망 내부 검증은 학습용 사람들의 이미지에서 나눕니다.

전처리는 **흑백 → 64×64 → [-1,1] 정규화**이며 히스토그램 평활화를 사용하지 않습니다. 원본에서 만든 uint8 메모리맵 캐시를 두 방법이 공유하고, 신경망 학습 부분에만 반전·밝기·회전 등의 증강을 적용합니다.

## 학습과 조기 종료

```text
train_preprocess.py → train.py의 학습·검증 → 베스트 모델 테스트
                                      → train_pca.py → 같은 테스트셋 성능 비교
```

```bash
python train.py --patience 10 --min-delta 0.0001 --val-ratio 0.1
python train.py --epochs 1 --device cuda      # 짧은 실행 확인
python train.py --model densenet
python train.py --folder /data/eyes
python train.py --csv data/eyes.csv
python train.py --mrl datasets --split-by subject
```

조기 종료는 **검증 손실 `val_loss`**를 기준으로 합니다. `--min-delta` 이상의 개선이 `--patience` epoch 동안 없으면 학습을 종료합니다. 작은 개선이 누적되어 지정한 개선폭에 도달하면 대기를 초기화합니다.

실제 검증 손실이 가장 낮을 때마다 **`results/best_model.pt`**를 저장합니다. 학습 종료 후 이 모델을 다시 읽어 최종 테스트 지표와 예측을 계산하고, 추론에서도 같은 파일을 사용합니다. 작은 개선도 베스트 모델 저장에는 반영합니다.

테스트 정확도는 매 epoch 플롯용으로 기록하지만 조기 종료나 베스트 모델 선택에는 사용하지 않습니다. `nn_validation_split.csv`에 내부 학습·검증 배정을 기록하며, `training_summary.json`에 베스트 epoch·실행 epoch·종료 여부를 저장합니다.

PCA는 학습 이미지의 분산이 큰 축을 계산한 뒤, PCA 공간의 **뜬 눈 평균 − 감은 눈 평균** 방향과 두 평균의 중간 경계로 분류합니다. 평균·축·방향·경계는 학습용 데이터로만 계산하며, 성분 수는 `--pca-components`로 지정합니다.

## 결과와 성능 비교

```text
results/
├── split.csv, split.npy, split_counts.csv
├── nn_validation_split.csv, history.csv, training_summary.json
├── best_model.pt, pca_model.npz, alarm_config.json
├── nn_metrics.json, pca_metrics.json, comparison.csv, evaluation.csv
├── nn_predictions.csv, pca_predictions.csv
├── image_prediction.json                # 사진 추론 결과
├── training_curves.png, test_accuracy.png
├── accuracy_comparison.png, performance_comparison.png
├── roc_comparison.png, pr_comparison.png, confusion_matrices.png
├── pca_projection.png, pca_direction.png, pca_components.png
├── face_landmarker.task                 # 웹캠·사진 수집 시 다운로드
└── collected_eyes/                      # 직접 수집한 학습 사진
```

| 플롯 | 내용 |
| --- | --- |
| `training_curves.png` | 학습·검증 손실과 정확도, 테스트 정확도, 베스트 epoch |
| `test_accuracy.png` | epoch별 테스트 정확도와 검증 기준 베스트 epoch |
| `accuracy_comparison.png` | 신경망 베스트 모델과 PCA의 테스트 정확도 |
| `performance_comparison.png` | Accuracy·macro-F1·감음 Precision/Recall·ROC-AUC·Average Precision |
| `roc_comparison.png`, `pr_comparison.png` | 감음을 양성으로 한 ROC·Precision–Recall 곡선 |
| `confusion_matrices.png` | 두 방법의 감음·뜸 오분류 수 |
| `pca_projection.png`, `pca_direction.png`, `pca_components.png` | 클래스 분포·픽셀 판정 방향·주성분 |

`python train.py --evaluate-only`는 베스트 모델과 PCA를 다시 평가해 `evaluation.csv`와 성능 비교 결과를 저장합니다. PCA 감음 점수는 경계 기준 값으로, 보정된 확률은 아닙니다.

## 사진·웹캠 추론과 알람

```bash
python inf.py                    # 신경망과 PCA, 알람은 신경망 기준
python inf.py --method nn
python inf.py --method pca        # PCA 기준 알람
python inf.py --preview-only     # 모델 없이 웹캠 전처리 확인
python inf.py --camera 1
```

현재 서버에는 `/dev/video*` 장치가 없어 `--image`로 개별 눈 사진을 확인합니다. 사진 판정 결과는 `results/image_prediction.json`에 저장하며, 사진 모드는 얼굴 검출·알람을 실행하지 않습니다. 웹캠이 연결된 환경에서는 `python inf.py`로 실행하고 `q` 또는 `Esc`로 종료합니다.

기본 알람은 **감음 0.5초 이상 AND 감기 시작 간격 3초 이하**가 동시에 성립할 때입니다. 양안 감음 점수의 최솟값을 최근 7회 평균해 0.5 이상이면 감음으로 봅니다. 첫 감음은 주기 미확정으로 제외하며, 감고 있는 동안 새 시작으로 세지 않습니다.

원본 프레임·점수·판정 결과를 최근 **4초 동안 RAM에 보관**하고 시간 기준으로 제거합니다. 얼굴·눈 검출 실패나 0.5초를 넘는 관측 공백에는 감음·주기 이력을 초기화합니다. 설정은 `results/alarm_config.json`에서 변경합니다. Windows는 `winsound`, Linux/macOS는 터미널 벨을 사용합니다.

사진·영상에서 학습용 눈 ROI를 수집할 수도 있습니다.

```bash
python train_preprocess.py --input videos/person_open.mp4 --subject person --label open
python train.py --folder results/collected_eyes
```

수집 입력은 한 사람·한 상태로 준비합니다. 원본 눈 ROI를 저장하며 기존 파일은 덮어쓰지 않습니다. 원본 데이터셋과 생성된 `results/`는 코드 업로드 대상에서 제외합니다. 웹캠·스피커 동작과 실제 활용 성능은 사용 환경에서 확인해야 합니다.
